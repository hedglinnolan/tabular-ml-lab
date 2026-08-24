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
    #: The features that SURVIVED selection, read off the fitted selector.
    #: `None` where no selection was recorded — which is not the same as an
    #: empty list, and a consumer that could not tell those apart would report
    #: *selection kept nothing* about a project that never selected.
    selected_features: Optional[List[str]] = None
    #: How many columns the selector was OFFERED — the recorded candidate pool
    #: as it survived to the selector — so the run can say *kept 3 of 6* rather
    #: than *kept 3*. A count with no denominator is a number a reader cannot
    #: judge.
    n_candidates: Optional[int] = None
    #: Columns that reached the model WITHOUT being offered to selection.
    #: Counted separately and never folded into the kept total: a run that said
    #: *kept 241 of 244* about a user who nominated six columns would be
    #: describing the encoder's output as if it were the user's choice.
    n_passthrough: Optional[int] = None

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
            "selected_features": (None if self.selected_features is None
                                  else list(self.selected_features)),
            "n_selected": (None if self.selected_features is None
                           else len(self.selected_features)),
            "n_candidates": self.n_candidates,
            "n_passthrough": self.n_passthrough,
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
    #: The invalidation log's length when this run was produced (`GUIDED-094`).
    #: Anything appended after it is a change these numbers do not account for,
    #: and `project.stale_since(mark)` says WHICH. `None` means the run
    #: predates the watermark and its staleness is undetermined — which is a
    #: third state, not a synonym for fresh.
    mark: Optional[int] = None
    #: **What a model that always answers the more common level scores.**
    #: `DRIVE-048`. The engine's own imbalance finding says *"Accuracy can be
    #: misleading; use F1, PR-AUC, or balanced accuracy instead"*, and the
    #: held-out table then leads with Accuracy 0.88 — sitting AT the 87.77%
    #: base rate. Both true, and a reader has to hold two screens in their head
    #: to see it. Carried so the number a metric must BEAT travels with the
    #: metrics rather than being a fact about the data three cards away.
    #:
    #: `None` on regression, and on any classification run whose held-out rows
    #: produced no usable outcome — never `0.0`, which is a score.
    majority_class_rate: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type, "target": self.target,
            "n_train": self.n_train, "n_test": self.n_test,
            "seal_basis": self.seal_basis, "exploratory": self.exploratory,
            "features": list(self.features), "notes": list(self.notes),
            "mark": self.mark,
            "majority_class_rate": self.majority_class_rate,
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


def feature_frame(project: Any, table: Optional[pd.DataFrame] = None
                  ) -> pd.DataFrame:
    """**The one place that decides what a model is fed.**

    `_feature_frame` above takes three loose arguments and four call sites in
    this package pass them — which is four places that have to remember the
    same rule, and `GUIDED-108` is what happens when one of them forgets.
    Identifier exclusion is the third rule now, after the target and the
    grouping key, and adding it as a fourth argument would have made the
    problem worse rather than better.

    So this is the project-aware door and the call sites use it.
    `test_every_path_that_feeds_a_model_goes_through_one_door` asserts that,
    because *somebody will pass the loose one next time* is the prediction this
    codebase has been right about repeatedly.
    """
    from turbotab import identifiers as _ids

    frame = project.working_table if table is None else table
    group_col = (getattr(project, "grain", None) or {}).get("group_col")
    out = _feature_frame(frame, str(project.target), group_col)
    excluded = set(_ids.excluded(project))
    if not excluded:
        return out
    return out[[c for c in out.columns if str(c) not in excluded]]


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


def _surviving_features(pipe: Any, plan: Any):
    """Which features the fitted selector kept, READ OFF THE FIT.

    `None` where no selection was recorded — distinct from `[]`, which would
    say the selector kept nothing.

    Read rather than re-derived: the selected set is the thing a reader most
    wants to check, and a second derivation of it is a second answer to *which
    columns did the model see*. The names come from the step BEFORE the
    selector, so a one-hot expansion is named as the columns the model actually
    got rather than as the categorical column they came from.
    """
    if getattr(plan, "selector_step", None) is None:
        return None, None, None
    try:
        selector = pipe.named_steps[plan.selector_step]
        # WHAT THE SELECTOR RANKED, asked of the selector. The pool is the
        # recorded candidates that reached it; everything else passed through
        # unranked and is counted separately, because a run that folded the
        # encoder's 238 one-hot columns into the kept total would be describing
        # the encoding as if it were the user's choice.
        pool = [str(c) for c in selector.pool_]
        kept = [str(c) for c in selector.kept_]
        passthrough = [str(c) for c in selector.passthrough_]
    except Exception:
        # A selector that cannot say what it kept says nothing, rather than a
        # list somebody would read as the answer.
        return None, None, None
    return kept, len(pool), len(passthrough)


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
    unchosen = event_not_chosen(project)
    if unchosen:
        raise TrainingRefusal(unchosen)


def event_not_chosen(project: Any) -> Optional[str]:
    """The refusal owed when the outcome has two levels and nobody said which
    is the event — or `None` when there is nothing to refuse.

    ## Why the fit and not the seal

    `DRIVE-032`, and `L60` had two defensible places to put this. The seal was
    the other one, and it is the wrong one: `engine.draw_holdout` does not
    stratify by class — it draws from `df.index[y.notna()]` and knows nothing
    about levels — so **the split is byte-identical whichever level is the
    event.** Gating the seal would refuse a step that cannot use the answer.

    Where the answer IS used is here and downstream: `positive_label` sets what
    sensitivity and specificity are the sensitivity and specificity *of*, and
    `figure_bundle.predictions_for` builds its entire binary vector from it. So
    the refusal lands where the value is consumed, which is also where
    `api.py`'s repair handler already refuses: *"There is no default: whether
    the event is (say) death or survival is the research question, not
    something the file can say."*

    ## What "chosen" means, and why it is read off the record

    Applying the `set_positive_class` repair rewrites the column so the chosen
    level is `1`. So a target whose event was chosen is `0`/`1` with the event
    as `1`, and `classes_[1]` is then genuinely correct rather than a guess.
    **The defect was never the encoding — it was running at all with the
    question open.** The decision is therefore the thing consulted, not the
    dtype: a `0`/`1` column can be one a user chose or one that arrived that
    way, and only the record can tell them apart.
    """
    from ml import binary_text as _bt

    target = str(project.target or "")
    if not target or (project.task_type or "") != "classification":
        return None
    table = project.working_table
    if target not in table.columns:
        return None
    if _bt.two_level_plan(table[target]) is None:
        return None            # not two-level: multiclass has no single event
    if event_decision(project) is not None:
        return None
    return (
        f"Which level of {target!r} is the event has not been recorded, and it "
        f"decides what every score means — sensitivity and specificity are of "
        f"the event, and the curves are drawn against it. There is no default: "
        f"whether the event is (say) death or survival is the research "
        f"question, not something the file can say. Answer "
        f"“Which of these is the event you are predicting?” on the "
        f"outcome, then fit.")


def event_decision(project: Any) -> Optional[Any]:
    """The recorded answer to *which level is the event*, or `None`.

    One lookup, read by the refusal above and by `chosen_event_level` below.
    They used to be one place because there was one reader; the moment
    `DRIVE-040` added a second, *is it answered* and *what was the answer* had
    to come from the same scan or they could disagree about a project — which
    is how a refusal ends up firing on a run whose figure is already naming the
    level, or the reverse.
    """
    target = str(project.target or "")
    if not target:
        return None
    for decision in reversed(list(project.decisions)):
        if (getattr(decision, "kind", "") == "apply"
                and str(getattr(decision, "subject", "")) == f"positive_class__{target}"):
            return decision
    return None


def chosen_event_level(project: Any) -> Optional[str]:
    """The LEVEL the user named, as it is spelled in their column.

    **`DRIVE-040`.** The repair encodes the chosen level as `1`, so from the
    fit onward the only thing visible is the encoded value and the figure said
    `event: "1.0"` — a value the user never typed, appearing nowhere in their
    column, decodable only from one sentence in the transcript.

    `None` where nothing was recorded, or where the record predates this field,
    and the caller then falls back to the encoded value. **Not a guess and not
    an empty string**: a project whose decision carries no level is one this
    cannot name, and saying so is the branch that keeps the figure honest.
    """
    decision = event_decision(project)
    if decision is None:
        return None
    payload = getattr(decision, "payload", None) or {}
    level = payload.get("event_level")
    return str(level) if level not in (None, "") else None


#: What the event's VALUE is in the working table once it has been recorded.
#:
#: `apply_positive_class` rewrites the outcome so the chosen level is `1`, so
#: after the repair the event is `1` **by construction**. This constant is the
#: one place that says so. Three call sites each knowing it would be three
#: copies of an encoding decision that lives in `ml/binary_text.py`.
EVENT_VALUE = 1


def outcome_level_names(project: Any) -> Dict[Any, str]:
    """`{1: "True", 0: "False"}` — the encoded values back to what a user typed.

    **`DRIVE-040`, and this is the half `L61` left.** That loop carried the
    EVENT's name, which is enough for a figure caption — *"829 events of
    True"*. It is not enough for anything that renders **both** levels, and
    three surfaces do: Table 1's column headers (`0 (n=770)` / `1 (n=5527)`),
    the PCA group annotation (`<NA> 15,552, 1 5,527, 0 770`), and the event
    noticing card, which flips from `False`/`True` to `0.0`/`1.0` the moment
    the answer is recorded.

    **Measured before it was built**: with the decision recorded, the payload
    carried `event_level` and nothing for the comparison, and the live
    finding's own `spellings` had been recomputed to `{'0': '0', '1': '1'}` —
    so after the repair the original words survived nowhere but the decision's
    prose. `engine.record_fix` records both now.

    **Empty where the record cannot say**, which includes every project sealed
    before `L62`. A renderer then shows the encoded value, which is what it
    showed before — silent rather than guessing, and a guess here would put a
    word in a user's column that they never typed.
    """
    decision = event_decision(project)
    if decision is None:
        return {}
    payload = getattr(decision, "payload", None) or {}
    event = payload.get("event_level")
    comparison = payload.get("comparison_level")
    names: Dict[Any, str] = {}
    if event not in (None, ""):
        names[EVENT_VALUE] = str(event)
    if comparison not in (None, ""):
        names[1 - EVENT_VALUE] = str(comparison)
    return names


def recorded_event_value(project: Any) -> Optional[Any]:
    """The value to compare a column against to count events, or `None`.

    **`DRIVE-043`.** `resolution.statement` counted events against
    `counts.index[-1]` — the LEAST FREQUENT level, never the recorded decision.
    When the event is the minority that is accidentally right, which is the
    ordinary clinical case and every fixture this repository had; when the user
    names the majority as the event it reports the non-event count under an
    events label. Run 5 named `True` on a column that is 87.77% `True` and the
    Methods section printed 116 where the figures printed 829, on the same 945
    rows.

    `None` means *nobody has said which level is the event*, and the caller
    must then say what it is actually counting rather than calling it an event.
    That branch is real and reachable: `set_positive_class` is not in
    `PRE_BARRIER_ONLY_FIXES`, so a user may seal first and answer afterwards,
    and the resolution is computed once at the seal and never recomputed.
    """
    if event_decision(project) is None:
        return None
    return EVENT_VALUE


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

    features = feature_frame(project, table)
    X_train = features[has_y & ~is_test]
    X_test = features[has_y & is_test]
    y_train = table.loc[X_train.index, target]
    y_test = table.loc[X_test.index, target]

    disclosures = getattr(project, "lockbox", {}) or {}
    # STAMPED BEFORE THE FIRST FIT, so a change made while the job runs counts
    # as a change this run does not account for. Stamping afterwards would
    # silently absorb it.
    stamped = project.mark() if hasattr(project, "mark") else None
    run = TrainingRun(
        task_type=task_type, target=target,
        n_train=int(len(X_train)), n_test=int(len(X_test)),
        seal_basis=disclosures.get("seal_basis"),
        exploratory=bool(_is_exploratory(project)),
        mark=stamped,
        # `DRIVE-048`. The score to beat, computed from the HELD-OUT rows —
        # the same rows every metric beside it is computed on, so the
        # comparison is like for like. `None` on regression and on an empty
        # holdout: returning `0.0` from ignorance would be a score.
        majority_class_rate=(
            float(y_test.value_counts(normalize=True).iloc[0])
            if task_type == "classification" and len(y_test) else None),
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
        # `AUDIT-028`. This read "Every statistic in it is fitted inside the
        # training folds." — over the fit thirty lines below, which is
        # `pipe.fit(X_train, y_train)`, once, per model. There are no folds in
        # this door: nothing under `turbotab/` imports `KFold`,
        # `cross_val_score` or `cross_validate`. The guarantee the note exists
        # to give is that no held-out row informs a fitted statistic, and THAT
        # is true and is what it now says — the claim is corrected to the one
        # the door can keep, not dropped.
        # `X_train` is `features[has_y & ~is_test]` — the analysis population,
        # not the training one. `DRIVE-050`'s class: the number is right and
        # the label named a wider set of rows than the number counts.
        f"Every statistic in it is fitted once over the {int(len(X_train)):,} "
        f"training rows with an outcome, and the held-out rows inform none of "
        f"them.")

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
            (result.selected_features, result.n_candidates,
             result.n_passthrough) = _surviving_features(pipe, plan)
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
    # HOW MANY FEATURES SURVIVED, AND WHICH. Named rather than counted: a
    # selection that reports only a count is a decision a reader cannot check,
    # and the surviving set is the one thing a methods section must carry.
    for result in run.results:
        if result.selected_features is None:
            continue
        kept = result.selected_features
        shown = ", ".join(f"`{c}`" for c in kept[:10])
        if len(kept) > 10:
            shown += f", and {len(kept) - 10} more"
        offered = result.n_candidates or len(kept)
        passed = result.n_passthrough or 0
        run.notes.append(
            f"{result.name}: feature selection kept {len(kept)} of {offered} "
            f"candidate column(s) — {shown}."
            + (f" {passed} further column(s) were not offered to selection and "
               f"reached the model unchanged." if passed else ""))
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

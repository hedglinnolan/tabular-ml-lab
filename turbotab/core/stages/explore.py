"""The ``explore`` stage: Explore after the seal (MODELING_SEQUENCE §0 ruling 3; §1 row 1; §4 "Outcome
views in Explore"; TRIPOD+AI item 7).

**Which rows.** Under prediction the training rows only: the held-out rows are never read, so a
holdout covers whatever Explore leads to. Under inference every analyzed row (BLUEPRINT §12 ruling
3: a holdout is a prediction concept), the rows the estimates come from.

**The stack.** Each finding is a one-line claim tied to a lever (BLUEPRINT §11 rule 7):

* *the outcome's relationship with each continuous predictor* — an outcome view (binned means, or the
  event's share, over the predictor's tenths); its lever is the functional form;
* *the outcome's distribution* — an outcome view; for a rare class, its lever is the imbalance
  question, where no correction with a threshold chosen in-fold ranks first (van den Goorbergh et al.,
  JAMIA 2022;29:1525: "similar results were obtained by shifting the probability threshold");
* *near-zero-variance predictors*, *more candidate predictors than rows* and *nearly collinear
  pairs* — outcome-free; their levers are the in-fold variance filter and the selection menu;
* *data quality across sociodemographic groups* (TRIPOD+AI 7: data-quality checks compared across
  groups) — each candidate predictor's missing share in each group of the columns named like sex,
  age, race or ethnicity, and income, quoted as their headers read; the groups are the column's
  levels when its numbers are codes and its thirds when they are an amount, as the readings ledger
  holds it (BLUEPRINT §14.3), and asked while that is unsettled; its lever names the groups for
  subgroup performance.

**Outcome views are recorded as looked at, under both purposes** (``view_outcome``), and never
blocked. Gelman & Loken (2013) on forking paths: choices made after seeing the data are an analysis
the reader cannot see unless it is said. The record's sentence says what was viewed, on which rows;
its standing clause (restated in the methods text on the answers as they stand) says which lever on
that column was set after the view:

* under prediction a lever set by hand after an outcome view is "chosen after viewing outcome
  relationships on the rows that validate the model; its optimism is not in the corrected score"
  (the review's wording), and when rows are held out, they "cover it": they were never viewed;
* under inference the choice is disclosed as made after the view.

**Each lever is offered first as an in-fold rule** under prediction: splines by Harrell's rule or the
form by inner cross-validation (``set_levers``), the in-fold variance filter, the selection menu run
in-fold (``set_selection``); the by-hand answer is offered after them with its cost stated.

Nothing here changes a number: the stage reads the settled roles (``readings``), and the columns it
names for subgroups are proposals the user confirms by naming them (``set_intended_use``).
"""
from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from turbotab.core.graph import Bundle, StageContext
from turbotab.core.stages.data import open_store

EXPLORE_READS: tuple[str, ...] = ("target", "purpose", "task", "event", "outcome_order", "lens",
                                  "outcome_views", "levers",
                                  "selection", "exposure_forms", "intended_use", "missing", "split",
                                  "outcome_scale",
                                  # each group column's code-or-amount reading (BLUEPRINT §14.3)
                                  "categorical", "aggregation", "codebooks")
MAX_SHOWN = 12  # BLUEPRINT §11 rule 3: wide data shows what the choice touched (≤ 12)
BINS = 10
RARE_CLASS = 0.20  # a convention: the rarer class below a fifth of the rows is called rare here
COLLINEAR = 0.90  # |r| at or above this, a nearly collinear pair (a convention)
MAX_PAIRS_P = 500  # beyond this many numeric predictors the pair scan is skipped, and says so
GROUP_GAP = 0.05  # a missing-share gap across groups at or above 5 points is called out
FORKING = "Gelman & Loken 2013"
OUTSIDE = ("chosen after viewing outcome relationships on the rows that validate the model; its "
           "optimism is not in the corrected score")
COVERED = "the held-out rows, never viewed, cover it"
# Sociodemographic columns by name (TRIPOD+AI 7 and 23a): a proposal the user confirms, never a reading
# that changes a number.
GROUP_WORDS = {
    "sex": ("sex", "gender", "riagendr"),
    "age": ("age", "ridageyr", "age_years", "ageyrs"),
    "race or ethnicity": ("race", "ethnic", "ethnicity", "ridreth", "ridreth1", "ridreth3"),
    "income": ("income", "poverty", "pir", "indfmpir", "ses"),
}


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)


class LeverOption(_Model):
    key: str
    label: str
    customary: str
    sound: str
    rung: str
    in_fold: bool = False  # an in-fold rule the resampling repeats
    decision: dict[str, Any] | None = None


class Lever(_Model):
    question: str
    options: list[LeverOption]


class BinPoint(_Model):
    x: float
    y: float | None
    n: int


class GroupQuality(_Model):
    group: str
    n: int
    missing_share: float  # rows with any candidate predictor blank


class ExploreFinding(_Model):
    id: str
    kind: str
    summary: str  # ≤ 20 words
    columns: list[str]
    outcome_view: bool = False  # it shows the outcome (recorded as looked at when opened)
    viewed: bool = False  # recorded as looked at (``view_outcome``)
    view: str  # the closed vocabulary: relationship · distribution · table_focus
    points: list[BinPoint] = []
    groups: list[GroupQuality] = []
    detail: str | None = None
    lever: Lever | None = None
    record: dict[str, Any] | None = None  # the ``view_outcome`` a client records on opening it


class HandLever(_Model):
    column: str
    view: str
    what: str  # the lever, in words
    then: str
    now: str
    sentence: str


class ExploreArtifact(_Model):
    """The ``explore`` artifact (module docstring)."""

    purpose: str | None
    rows: str  # "training" or "analyzed"
    n_rows: int
    n_holdout: int
    findings: list[ExploreFinding]
    more: int = 0  # findings of a kind beyond the shown
    proposed_subgroups: list[str] = []
    viewed: list[str] = []  # the outcome views recorded as looked at (their keys)
    hand_levers: list[HandLever] = []
    sentence: str


# ── lever answers, then and now (the forking-paths record) ───────────────────


def _form_words(spec: Any) -> str:
    if spec is None:
        return "linear"
    form = getattr(spec, "form", None) or (spec.get("form") if isinstance(spec, Mapping) else None)
    knots = getattr(spec, "knots", None) if not isinstance(spec, Mapping) else spec.get("knots")
    if form == "spline":
        return f"spline:{knots or 4}"
    return str(form or "linear")


def lever_answers(state: Any, column: str, view: str) -> dict[str, str]:
    """The answers a lever on ``column`` writes, as words (``view_outcome`` keeps them at the first
    look): its form and its role (a predictor dropped after the view is a selection by hand); for
    the outcome's distribution, its scale and the imbalance correction."""
    from turbotab.core.decisions import left_out

    if view == "distribution":
        scale = getattr(state, "outcome_scale", None)
        levers = getattr(state, "levers", None)
        return {"scale": str(getattr(scale, "scale", None) or "as recorded"),
                "imbalance": str(getattr(levers, "imbalance", "none") or "none")}
    roles = getattr(state, "roles", None) or {}
    forms = getattr(state, "exposure_forms", None) or {}
    out = {"form": _form_words(forms.get(column)), "role": str(roles.get(column) or "none")}
    out["kept"] = "no" if column in set(left_out(state)) else "yes"
    return out


_LEVER_WORDS = {"form": "form", "role": "role", "kept": "place among the predictors",
                "scale": "scale", "imbalance": "imbalance correction"}


def _value_words(key: str, value: str) -> str:
    if key == "form" and value.startswith("spline:"):
        return f"a restricted cubic spline with {value.split(':', 1)[1]} knots"
    if key == "kept":
        return "kept" if value == "yes" else "left out"
    return value


def hand_levers(state: Any) -> list[HandLever]:
    """Each lever on a viewed column whose answer differs from the one it had at the first look:
    set after the outcome view (module docstring)."""
    out: list[HandLever] = []
    target = getattr(state, "target", None)
    held = float(getattr(getattr(state, "split", None), "holdout", 0) or 0) > 0
    inference = getattr(state, "purpose", None) == "inference"
    for key, spec in sorted((getattr(state, "outcome_views", None) or {}).items()):
        if spec.target is not None and spec.target != target:
            continue
        now = lever_answers(state, spec.column, spec.view)
        for lever, then in sorted((spec.levers or {}).items()):
            current = now.get(lever)
            if current is None or current == then:
                continue
            subject = ("the outcome's" if spec.view == "distribution" else f"`{spec.column}`'s")
            what = f"{subject} {_LEVER_WORDS.get(lever, lever)}"
            change = f"{_value_words(lever, then)} to {_value_words(lever, current)}"
            seen = ("the outcome's distribution was viewed" if spec.view == "distribution"
                    else f"its relationship with the outcome was viewed")
            if inference:
                said = f"{what} was changed from {change} after {seen} (forking paths)"
            else:
                said = f"{what} was set by hand from {change}, {OUTSIDE}" + (
                    f"; {COVERED}" if held else "")
            out.append(HandLever(column=spec.column, view=spec.view, what=what, then=then,
                                 now=current, sentence=said[0].upper() + said[1:] + "."))
    return out


def _viewed(state: Any) -> dict[str, Any]:
    target = getattr(state, "target", None)
    return {k: v for k, v in (getattr(state, "outcome_views", None) or {}).items()
            if v.target is None or v.target == target}


def explore_sentence(state: Any, n_rows: int, n_holdout: int,
                     hands: Sequence[HandLever]) -> str:
    """The Explore methods sentence (forking paths; module docstring)."""
    from turbotab.core.voice import listing

    inference = getattr(state, "purpose", None) == "inference"
    rows = f"the {n_rows:,} {'analyzed' if inference else 'training'} rows"
    views = _viewed(state)
    related = sorted({v.column for v in views.values() if v.view == "relationship"})
    distribution = any(v.view == "distribution" for v in views.values())
    looked = []
    if related:
        looked.append(f"the outcome's relationship with {listing(related)}")
    if distribution:
        looked.append("the outcome's distribution")
    if inference:
        head = f"Explore read {rows}"
        if looked:
            head += (f"; {' and '.join(looked)} {'was' if len(looked) == 1 else 'were'} viewed "
                     f"before the estimates and recorded as looked at ({FORKING}); nothing was "
                     f"blocked, and each choice made after a view is disclosed")
        else:
            head += "; no outcome view was opened"
    else:
        head = f"Explore read {rows} only"
        if n_holdout:
            head += f" (the {n_holdout:,} held-out rows were never read)"
        head += ", and every lever was offered first as an in-fold rule the resampling repeats"
        if looked:
            head += (f"; {' and '.join(looked)} {'was' if len(looked) == 1 else 'were'} viewed and "
                     f"recorded as looked at ({FORKING})")
        else:
            head += "; no outcome view was opened"
    tail = " ".join(h.sentence for h in hands)
    return f"{head}." + (f" {tail}" if tail else "")


# ── the findings ─────────────────────────────────────────────────────────────


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def relationship_points(x: np.ndarray, y: np.ndarray) -> list[BinPoint]:
    """The outcome's mean (the event's share for 0/1) over the predictor's tenths, by its 10th, 20th
    … quantiles (R type 7); each bin's median predictor value and its rows."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < BINS:
        return []
    cuts = np.unique(np.quantile(x, np.linspace(0.1, 0.9, BINS - 1)))
    group = np.searchsorted(cuts, x, side="left")
    out = []
    for g in range(len(cuts) + 1):
        rows = group == g
        if rows.any():
            out.append(BinPoint(x=float(np.median(x[rows])), y=float(np.mean(y[rows])),
                                n=int(rows.sum())))
    return out


def subgroup_candidates(columns: Sequence[str], skip: Sequence[str] = ()) -> dict[str, str]:
    """Columns named like a sociodemographic group (``GROUP_WORDS``): column → the group kind. A
    name only: a proposal, confirmed by naming it as a subgroup."""
    out: dict[str, str] = {}
    for c in columns:
        if c in skip:
            continue
        tokens = set(re.split(r"[^a-z0-9]+", str(c).lower())) | {str(c).lower()}
        for kind, words in GROUP_WORDS.items():
            if tokens & set(words) and c not in out:
                out[c] = kind
    return out


def _lever(question: str, options: Sequence[LeverOption]) -> Lever:
    return Lever(question=question, options=list(options))


def _levers_now(state: Any) -> dict[str, Any]:
    spec = getattr(state, "levers", None)
    base = {"forms": "none", "variance_filter": "none", "keep": None, "imbalance": "none"}
    if spec is not None:
        base.update(spec.model_dump())
    return base


def _set_levers(state: Any, **change: Any) -> dict[str, Any]:
    return {"kind": "set_levers", **{**_levers_now(state), **change}}


def form_lever(state: Any, column: str, n_rows: int, task: str) -> Lever:
    """The functional form, offered first as an in-fold rule under prediction (module docstring);
    declared under inference."""
    from turbotab.core.methods.levers import knots_by_rule

    k = knots_by_rule(n_rows)
    if getattr(state, "purpose", None) == "inference":
        return _lever("Declare this predictor's form before the estimates?", [
            LeverOption(key="spline", label=f"A restricted cubic spline, {k} knots by Harrell's rule",
                        customary="Quintiles are the field's customary primary form",
                        sound="Declared before estimates; the overall test is the test of "
                              "association; the view is disclosed",
                        rung="recommended",
                        decision={"kind": "set_exposure_form", "column": column, "form": "spline",
                                  "knots": k}),
            LeverOption(key="linear", label="A straight line",
                        customary="A straight line is the default in most papers",
                        sound="Declared; assumes the effect is the same per unit everywhere",
                        rung="available",
                        decision={"kind": "set_exposure_form", "column": column, "form": "linear"}),
        ])
    options = [
        LeverOption(key="rule", label="Every continuous predictor as a spline, knots by a stated rule",
                    customary="Splines are usually chosen per predictor after looking at plots",
                    sound="Sound: an in-fold rule the resampling repeats; it never reads this view",
                    rung="recommended", in_fold=True, decision=_set_levers(state, forms="rule")),
    ]
    if task in ("regression", "binary"):
        options.append(LeverOption(
            key="inner_cv", label="Linear or spline for each predictor, by inner cross-validation",
            customary="Nonlinearity is usually tested on all rows, then kept or dropped",
            sound="Sound: chosen inside each training fold, so its optimism is in the score",
            rung="recommended", in_fold=True, decision=_set_levers(state, forms="inner_cv")))
    held = float(getattr(getattr(state, "split", None), "holdout", 0) or 0) > 0
    options.append(LeverOption(
        key="by_hand", label=f"A spline for `{column}` alone, by hand",
        customary="Bending one predictor because its plot bends is common",
        sound=("Outside the corrected score: chosen after viewing this relationship"
               + ("; the held-out rows cover it" if held else "")),
        rung="available",
        decision={"kind": "set_exposure_form", "column": column, "form": "spline", "knots": k}))
    return _lever("Let a rule bend the continuous predictors, inside the resampling?", options)


def explore_stage(ctx: StageContext) -> Bundle:
    from turbotab.core.models.pipeline import model_predictors, modeling_frame
    from turbotab.core.stages.modeling import _task, coded_outcome, read_assignment, row_ids_of

    state = ctx.state
    task = _task(ctx)
    inference = state.purpose == "inference"
    cohort, split = ctx.inputs["cohort"], ctx.inputs["split"]
    rows = row_ids_of(cohort.frames["rows"]) if isinstance(cohort, Bundle) else None
    assignment = read_assignment(split)
    train_ids = assignment.index[assignment["train"]].to_numpy()
    held_ids = assignment.index[~assignment["train"].to_numpy()].to_numpy()
    ids = assignment.index.to_numpy() if inference else train_ids
    if rows is not None:
        ids = np.intersect1d(ids, rows)
    n_holdout = 0 if inference else int(len(held_ids))
    predictors = [c for c in model_predictors(state) if c != state.target]
    target = state.target
    ctx.progress(0.1, "Reading the rows Explore may read")
    with open_store(ctx) as store:
        candidates = subgroup_candidates([c for c in store.columns if c != target],
                                         skip=[c for c, r in (state.roles or {}).items()
                                               if r in ("identifier", "design")])
        wanted = list(dict.fromkeys([*predictors, *candidates, target]))
        frame = modeling_frame(store, [c for c in wanted if c in set(store.columns)], ids,
                               outcome=target)
        from turbotab.core.readings import whole_facts

        # Each group column's code-or-amount question, read on the whole column (BLUEPRINT §14.3).
        group_facts = whole_facts(list(candidates), None, store)
    y_raw = frame[target].to_numpy()
    coded = coded_outcome(task, y_raw, state.event, order=state.outcome_order)
    y = (pd.to_numeric(pd.Series(np.asarray(coded)), errors="coerce").to_numpy(dtype=float)
         if task in ("regression", "binary") else None)
    n = int(len(frame))
    findings: list[ExploreFinding] = []
    more = 0
    views = _viewed(state)
    word = "analyzed" if inference else "training"

    # ── the outcome's relationship with each continuous predictor (outcome views) ──
    from turbotab.core.methods.levers import MIN_DISTINCT

    numeric = [c for c in predictors if c in frame.columns
               and pd.api.types.is_numeric_dtype(frame[c]) and not pd.api.types.is_bool_dtype(frame[c])]
    curves = [c for c in numeric if frame[c].nunique(dropna=True) >= MIN_DISTINCT]
    if y is not None:
        shown = curves[:MAX_SHOWN]
        more += max(0, len(curves) - MAX_SHOWN)
        for c in shown:
            key = f"relationship:{c}"
            findings.append(ExploreFinding(
                id=f"explore::relationship::{c}", kind="outcome_relationship",
                summary=f"How the outcome moves with `{c}` on the {word} rows; opening it is recorded",
                columns=[c], outcome_view=True, viewed=key in views, view="relationship",
                points=relationship_points(_numeric(frame, c), y),
                lever=form_lever(state, c, n, task),
                record={"kind": "view_outcome", "view": "relationship", "columns": [c]}))

    # ── the outcome's distribution (an outcome view) ──
    finding = _distribution_finding(state, task, y_raw, coded, views, word)
    if finding is not None:
        findings.append(finding)

    # ── outcome-free: near-zero variance, p ≫ n, collinear pairs ──
    from turbotab.core.methods.levers import near_zero_variance

    flat = [c for c in predictors if c in frame.columns and near_zero_variance(frame[c].to_numpy())]
    if flat:
        findings.append(ExploreFinding(
            id="explore::low_variance", kind="low_variance",
            summary=(f"{len(flat):,} predictor{'s' if len(flat) != 1 else ''} barely vary on the "
                     f"{word} rows"),
            columns=flat[:MAX_SHOWN], view="table_focus",
            detail=("caret's nearZeroVar rule: one value, or the most common value at least 19 "
                    "times the next with at most 10% distinct values."),
            lever=_variance_lever(state)))
    if len(predictors) >= n and n > 0:
        findings.append(ExploreFinding(
            id="explore::wide", kind="wide",
            summary=f"{len(predictors):,} candidate predictors for {n:,} rows: selection or a "
                    f"filter comes first",
            columns=predictors[:MAX_SHOWN], view="table_focus", lever=_wide_lever(state)))
    pairs = _collinear(frame, numeric)
    if pairs:
        findings.append(ExploreFinding(
            id="explore::collinear", kind="collinear",
            summary=f"{len(pairs):,} pair{'s' if len(pairs) != 1 else ''} of predictors move almost "
                    f"together (|r| ≥ {COLLINEAR:g})",
            columns=sorted({c for a, b, _ in pairs[:MAX_SHOWN] for c in (a, b)}),
            view="relationship",
            detail="; ".join(f"`{a}` and `{b}`: r = {r:.2f}" for a, b, r in pairs[:MAX_SHOWN]),
            lever=_collinear_lever(state)))

    # ── TRIPOD+AI 7: data quality across sociodemographic groups ──
    for column, kind in candidates.items():
        if column not in frame.columns:
            continue
        finding = _quality_finding(state, frame, predictors, column, kind, word,
                                   group_facts.get(column) if column in group_facts else None)
        if finding is not None:
            findings.append(finding)

    # ── ruling 13: under prediction, whose performance the scores estimate ──
    if not inference:
        finding = _survey_finding(state)
        if finding is not None:
            findings.append(finding)

    hands = hand_levers(state)
    artifact = ExploreArtifact(
        purpose=state.purpose, rows=word, n_rows=n, n_holdout=n_holdout, findings=findings,
        more=more, proposed_subgroups=list(candidates), viewed=sorted(views),
        hand_levers=hands, sentence=explore_sentence(state, n, n_holdout, hands))
    ctx.progress(1.0, "Done")
    return Bundle(data=artifact.model_dump(mode="json"))


def _distribution_finding(state: Any, task: str, y_raw: Any, coded: Any, views: Mapping[str, Any],
                          word: str) -> ExploreFinding | None:
    target = state.target
    key = f"distribution:{target}"
    record = {"kind": "view_outcome", "view": "distribution", "columns": []}
    if task == "regression":
        values = pd.to_numeric(pd.Series(np.asarray(y_raw)), errors="coerce").dropna().to_numpy()
        if not len(values):
            return None
        counts, edges = np.histogram(values, bins=20)
        points = [BinPoint(x=float((edges[i] + edges[i + 1]) / 2), y=None, n=int(c))
                  for i, c in enumerate(counts)]
        return ExploreFinding(
            id="explore::outcome_distribution", kind="outcome_distribution",
            summary=f"The outcome's distribution on the {word} rows; opening it is recorded",
            columns=[target], outcome_view=True, viewed=key in views, view="distribution",
            points=points, record=record, lever=_scale_lever(state))
    if task not in ("binary", "multiclass", "ordinal"):
        return None
    counts = pd.Series(np.asarray(coded, dtype=object)).value_counts()
    points = [BinPoint(x=float(i), y=float(c) / float(counts.sum()), n=int(c))
              for i, c in enumerate(counts.to_numpy())]
    share = float(counts.min()) / float(counts.sum()) if len(counts) else 0.0
    rare = task == "binary" and share < RARE_CLASS
    if state.purpose == "inference":
        lever = _measure_lever(state) if task == "binary" else _scale_lever(state)
    elif task == "binary":
        lever = _imbalance_lever(state) if rare else _use_lever(state)
    else:
        lever = _scale_lever(state)
    summary = (f"The rarer class is {share:.0%} of the {word} rows; opening it is recorded"
               if rare else f"The outcome's classes on the {word} rows; opening it is recorded")
    return ExploreFinding(
        id="explore::outcome_distribution", kind="outcome_distribution", summary=summary,
        columns=[target], outcome_view=True, viewed=key in views, view="distribution",
        points=points, lever=lever, record=record)


def _survey_finding(state: Any) -> ExploreFinding | None:
    """MODELING_SEQUENCE ruling 13: a survey design read under prediction asks whose performance the
    scores estimate; the population's is design-based cross-validation."""
    from turbotab.core.survey import offered, reading_of

    reading = reading_of(state)
    if not reading.present:
        return None
    options = []
    for o in offered(state):
        population = (o.get("decision") or {}).get("estimand") == "population"
        options.append(LeverOption(
            key=str(o["key"]), label=(f"{o['label']}: design-based cross-validation" if population
                                      else f"{o['label']}: unweighted scores"),
            customary="Unweighted cross-validation is the habit even on survey data",
            sound=("Sound for performance in the surveyed population: whole PSUs within strata, "
                   "every score weighted (Wieczorek et al. 2022)" if population else
                   "Sound for the procedure's performance on these rows, labeled so"),
            rung="recommended" if population else "available", decision=o.get("decision")))
    answered = getattr(state, "survey", None)
    said = ("Whose performance do the scores estimate: the surveyed population's or these rows'?"
            if answered is None else
            f"The scores estimate {'the surveyed population' if answered.estimand == 'population' else 'these rows'}'s "
            f"performance, as answered")
    return ExploreFinding(
        id="explore::survey_design", kind="survey_design",
        summary="Survey weights are present: the population's performance needs design-based "
                "cross-validation",
        columns=[c for c in (*reading.weights[:1], *reading.strata[:1], *reading.psu[:1]) if c],
        view="table_focus", detail=said, lever=_lever("Whose performance do the scores estimate?",
                                                      options))


def _scale_lever(state: Any) -> Lever:
    """The outcome's scale and levels are answered with the outcome, before the seal."""
    return _lever("Change how the outcome is read?", [
        LeverOption(key="scale", label="The outcome's scale and levels: answered with the outcome; "
                                       "a re-seal to change them",
                    customary="A skewed outcome is often logged after a look at its histogram",
                    sound=("Set before the seal; a change after this view is disclosed, and "
                           "outside the corrected score" if state.purpose != "inference" else
                           "Set before the seal; a change after this view is disclosed"),
                    rung="available")])


def _use_lever(state: Any) -> Lever:
    use = getattr(state, "intended_use", None)
    keep = use.model_dump() if use is not None else {}
    return _lever("Will a decision rest on a threshold of this risk?", [
        LeverOption(key="decision_support", label="Decision support: the decision curve",
                    customary="Decision curves are recommended but rare in nutrition papers",
                    sound="Net benefit over the declared threshold range (Vickers & Elkin 2006)",
                    rung="recommended",
                    decision={"kind": "set_intended_use", **{**keep, "use": "decision_support"}}),
        LeverOption(key="risk_estimation", label="Risk estimation only",
                    customary="Most papers report discrimination and calibration only",
                    sound="Sound when no decision rests on a cut-off", rung="available",
                    decision={"kind": "set_intended_use", **{**keep, "use": "risk_estimation"}})])


def _measure_lever(state: Any) -> Lever:
    return _lever("Which effect measure does the estimand declare?", [
        LeverOption(key="measure", label="The effect measure (the estimand question)",
                    customary="Odds ratios are the field's habit for a yes/no outcome",
                    sound="For a common outcome a marginal risk difference or ratio ranks first "
                          "(ruling 9); declared before the estimates, the view disclosed",
                    rung="recommended")])


def _imbalance_lever(state: Any) -> Lever:
    use = getattr(state, "intended_use", None)
    keep = use.model_dump() if use is not None else {}
    return _lever("Correct the class imbalance, or move the threshold?", [
        LeverOption(key="none", label="No correction; a threshold chosen in-fold if a decision rests "
                                      "on it",
                    customary="Corrections are common for rare outcomes in ML",
                    sound="Sound: imbalance is not a problem in itself; moving the threshold gives "
                          "the same balance (van den Goorbergh et al. 2022)",
                    rung="recommended", in_fold=True,
                    decision={"kind": "set_intended_use",
                              **{**keep, "use": "decision_support"}}),
        *[LeverOption(key=m, label=f"{label}, in each fold, then recalibration",
                      customary="Class weights and resampling are customary in ML",
                      sound="Miscalibrates until recalibrated; no gain in AUC (van den Goorbergh "
                            "et al. 2022)", rung="rank_lower", in_fold=True,
                      decision=_set_levers(state, imbalance=m))
          for m, label in (("weights", "Balanced class weights"), ("undersample", "Undersampling"),
                           ("oversample", "Oversampling"))],
    ])


def _variance_lever(state: Any) -> Lever:
    if state.purpose == "inference":
        return _lever("Keep these predictors in the declared model?", [
            LeverOption(key="adjustment", label="Answer it in the adjustment set",
                        customary="Near-constant covariates are often dropped silently",
                        sound="Under inference the adjustment set is declared from subject "
                              "knowledge", rung="recommended")])
    return _lever("Drop near-constant predictors inside each training fold?", [
        LeverOption(key="near_zero", label="Drop near-zero-variance predictors in each fold",
                    customary="Filters are usually run once on every row",
                    sound="Sound: outcome-free and learned on each fold (Moscovich & Rosset 2022)",
                    rung="recommended", in_fold=True,
                    decision=_set_levers(state, variance_filter="near_zero")),
        LeverOption(key="by_hand", label="Leave them out by hand",
                    customary="Dropping them before modeling is common",
                    sound="Outside the resampling: learned on every training row",
                    rung="rank_lower")])


def _wide_lever(state: Any) -> Lever:
    if state.purpose == "inference":
        return _lever("Model each feature on its own, with a multiplicity method?", [
            LeverOption(key="featurewise", label="Feature-wise models with FDR",
                        customary="Metabolome- and genome-wide studies test feature by feature",
                        sound="Every member shown; this is not selection", rung="recommended",
                        decision={"kind": "select_models", "models": ["featurewise"]})])
    return _lever("Screen or penalize inside the resampling?", [
        LeverOption(key="screening", label="In-fold screening by correlation with the outcome",
                    customary="Screens are usually run once on every row",
                    sound="Sound in-fold at p ≫ n (Ambroise & McLachlan 2002)", rung="recommended",
                    in_fold=True, decision={"kind": "set_selection", "method": "screening"}),
        LeverOption(key="elastic_net", label="Elastic net in each fold",
                    customary="Penalized selection is common in omics prediction",
                    sound="Sound: tuned and selected inside each fold", rung="recommended",
                    in_fold=True, decision={"kind": "set_selection", "method": "elastic_net"}),
        LeverOption(key="top", label="Keep the 1,000 most variable predictors in each fold",
                    customary="Variance and IQR filters are customary in omics",
                    sound="Outcome-free and in-fold", rung="available", in_fold=True,
                    decision=_set_levers(state, variance_filter="top", keep=1000))])


def _collinear_lever(state: Any) -> Lever:
    if state.purpose == "inference":
        return _lever("Do both belong in the declared model?", [
            LeverOption(key="adjustment", label="Answer it in the adjustment set",
                        customary="One of a collinear pair is often dropped by its p-value",
                        sound="Under inference the set is declared from subject knowledge; the "
                              "intervals show the cost", rung="recommended")])
    return _lever("Let a penalty share the weight between them, inside each fold?", [
        LeverOption(key="elastic_net", label="Elastic net in each fold",
                    customary="One of a pair is often dropped by hand",
                    sound="Sound: the penalty spreads the weight; the resampling repeats it",
                    rung="recommended", in_fold=True,
                    decision={"kind": "set_selection", "method": "elastic_net"}),
        LeverOption(key="by_hand", label="Leave one out by hand",
                    customary="Common practice",
                    sound="Outside the resampling when chosen on these rows", rung="rank_lower")])


def _collinear(frame: pd.DataFrame, numeric: Sequence[str]) -> list[tuple[str, str, float]]:
    """Pairs of numeric predictors with |r| ≥ :data:`COLLINEAR` (none scanned beyond
    :data:`MAX_PAIRS_P` predictors, where the selection lever of p ≫ n speaks instead)."""
    if len(numeric) < 2 or len(numeric) > MAX_PAIRS_P:
        return []
    values = frame[list(numeric)].apply(pd.to_numeric, errors="coerce")
    corr = values.corr().to_numpy()
    out = []
    for i in range(len(numeric)):
        for j in range(i + 1, len(numeric)):
            r = corr[i, j]
            if np.isfinite(r) and abs(r) >= COLLINEAR:
                out.append((numeric[i], numeric[j], float(r)))
    out.sort(key=lambda t: -abs(t[2]))
    return out


def _quality_finding(state: Any, frame: pd.DataFrame, predictors: Sequence[str], column: str,
                     kind: str, word: str, facts: Mapping[str, Any] | None = None
                     ) -> ExploreFinding | None:
    """TRIPOD+AI 7: the share of rows with any candidate predictor blank, in each group of a
    column named like a sociodemographic group. The groups are the column's levels when its numbers
    are codes and its thirds when they are an amount, as the readings ledger holds it
    (``decision_curve.grouping_of``); while that reading is unsettled the finding asks it and shows
    no groups."""
    from turbotab.core.models.decision_curve import grouping_of, subgroup_labels

    others = [c for c in predictors if c in frame.columns and c != column]
    if not others:
        return None
    how, waiting = grouping_of(state, column, facts)
    if how is None:
        from turbotab.core.readings import ask_exits, guess_words

        exits = ask_exits([waiting], state)
        return ExploreFinding(
            id=f"explore::quality::{column}", kind="quality_by_group",
            summary=f"`{column}`'s groups wait on whether its numbers are codes or amounts",
            columns=[column], view="table_focus",
            detail=(f"Named like {kind}: a proposal from the header `{column}`. Its groups are its "
                    f"levels if its numbers are codes and its thirds if they are an amount, and "
                    f"that is not settled (best guess: {guess_words(waiting)}; "
                    f"{waiting.evidence}). Checked on the {word} rows once answered (TRIPOD+AI 7)."),
            lever=_lever(f"Are `{column}`'s numbers codes or amounts?", [
                LeverOption(key=str(e["decision"]["value"]), label=e["label"],
                            customary="Groups are often read off a column's count of values",
                            sound="Its levels as codes, its thirds as an amount, as you confirm "
                                  "(BLUEPRINT §14.3)",
                            rung="recommended", decision=e["decision"])
                for e in exits if (e.get("decision") or {}).get("kind") == "confirm_reading"]))
    blank = frame[others].isna().any(axis=1).to_numpy()
    labels = subgroup_labels(frame[column].to_numpy(), how)
    groups = []
    for level in sorted(set(labels.tolist()), key=lambda v: (v == "(blank)", v)):
        rows = labels == level
        groups.append(GroupQuality(group=level, n=int(rows.sum()),
                                   missing_share=float(blank[rows].mean()) if rows.any() else 0.0))
    shares = [g.missing_share for g in groups if g.n]
    gap = (max(shares) - min(shares)) if shares else 0.0
    summary = (f"Missing predictors differ by {gap:.0%} across `{column}`'s groups"
               if gap >= GROUP_GAP else
               f"Missing predictors are similar across `{column}`'s groups (within {GROUP_GAP:.0%})")
    use = getattr(state, "intended_use", None)
    named = list(use.subgroups) if use is not None else []
    lever = _lever("Do these differences change how missing values are handled?", [
        LeverOption(key="missing", label="The missing-values question, with these differences in "
                                         "view",
                    customary="Missingness is rarely compared across groups",
                    sound="Multiple imputation with the group in its model when missingness "
                          "differs by group; complete cases states its assumption",
                    rung="recommended")])
    if state.purpose != "inference":
        lever = _lever("Report performance within these groups?", [
            LeverOption(key="subgroups", label=f"Score performance within `{column}`'s groups",
                        customary="Subgroup performance is rarely reported",
                        sound="TRIPOD+AI 23a: performance with intervals in key subgroups",
                        rung="recommended",
                        decision={"kind": "set_intended_use",
                                  **(use.model_dump() if use is not None
                                     else {"use": "risk_estimation"}),
                                  "subgroups": list(dict.fromkeys([*named, column]))})])
    return ExploreFinding(
        id=f"explore::quality::{column}", kind="quality_by_group", summary=summary,
        columns=[column], view="table_focus", groups=groups,
        detail=(f"Named like {kind}: a proposal from the header `{column}`, its groups "
                f"{'its own levels (codes)' if how == 'levels' else 'its thirds (an amount)'}; "
                f"checked on the {word} rows (TRIPOD+AI 7)."),
        lever=lever)


# ── the view_outcome record: its server-filled fields, its leash, its sentence ──


def _view_columns_exist(decision: Any, ctx: Any) -> None:
    from turbotab.core.decisions import Refusal, _columns_of, _state

    state = _state(ctx)
    if state is not None and not getattr(state, "target", None):
        raise Refusal("no_outcome", "An outcome view needs the outcome chosen first.",
                      exits=[{"label": "Choose the outcome", "decision": None}])
    if decision.view == "relationship" and not decision.columns:
        raise Refusal("no_column", "A relationship view names the predictor it sets against the "
                                   "outcome.", exits=[{"label": "Name a predictor", "decision": None}])
    columns = _columns_of(ctx)
    if columns is not None:
        unknown = [c for c in decision.columns if c not in columns]
        if unknown:
            raise Refusal("unknown_column", f"This dataset has no column named `{unknown[0]}`.",
                          exits=[{"label": "Choose one of the dataset's columns", "decision": None}])


def _view_filled(decision: Any, ctx: Any) -> Any:
    """The server's fields: the outcome, the rows, and each column's lever answers at its first look
    (a later look at one column keeps the first look's answers)."""
    from turbotab.core.decisions import _state
    from turbotab.core.sequence import artifact

    state = _state(ctx)
    if state is None:
        return decision
    target = state.target
    inference = state.purpose == "inference"
    split = artifact(ctx, "split") or {}
    n_rows = None
    if split:
        n_rows = int(split.get("n_train") or 0) + (int(split.get("n_holdout") or 0) if inference
                                                   else 0)
    earlier = getattr(state, "outcome_views", None) or {}
    columns = list(decision.columns) or ([target] if target else [])
    levers: dict[str, dict[str, str]] = {}
    for c in columns:
        first = earlier.get(f"{decision.view}:{c}")
        if first is not None and first.target == target and first.levers:
            levers[c] = dict(first.levers)
        else:
            levers[c] = lever_answers(state, c, decision.view)
    return decision.model_copy(update={
        "columns": columns if decision.view == "relationship" else [],
        "target": target, "rows": "analyzed" if inference else "training",
        "n_rows": n_rows or decision.n_rows, "levers": levers})


def _register() -> None:
    from turbotab.core.decisions import register_completion, register_validator

    register_validator("view_outcome", _view_columns_exist)
    register_completion("view_outcome", _view_filled)


_register()


def view_sentence(d: Any, state: Any) -> str:
    """The ``view_outcome`` record's sentence: what was viewed, on which rows (forking paths)."""
    from turbotab.core.voice import count, listing

    rows = f" on the {count(d.n_rows)} {d.rows} rows" if d.n_rows else (
        f" on the {d.rows} rows" if d.rows else "")
    what = (f"The outcome's relationship with {listing(d.columns)}" if d.view == "relationship"
            else "The outcome's distribution")
    return (f"{what} was viewed in Explore{rows} and recorded as looked at ({FORKING}: a choice made "
            f"after it is disclosed)")


def view_standing(d: Any, state: Any, ctx: Any) -> str | None:
    """The standing clause: each lever on this view's columns set after it (module docstring)."""
    if state is None:
        return None
    columns = set(d.columns or ([d.target] if d.target else []))
    hands = [h for h in hand_levers(state) if h.column in columns and h.view == d.view]
    if not hands:
        return None
    return " ".join(h.sentence for h in hands)


def _register_contract() -> None:
    from turbotab.core.contracts import (CONTRACTS, ContractOption, MethodContract, Relation,
                                        register_contract)

    if "explore" in CONTRACTS:
        return
    here = "turbotab.core.stages.explore"
    register_contract(MethodContract(
        key="explore", label="Explore after the seal", slot="in_fold", scope="descriptive",
        package="EXPLORE", decision="view_outcome", stage="explore", place="1 · Explore",
        run_order=0.5,
        scope_note=("It reads the training rows under prediction and every analyzed row under "
                    "inference, informs no choice by itself, and offers each lever as an in-fold "
                    "rule; an outcome view it shows is recorded as looked at."),
        needs=("the drawn seal", "the settled roles"),
        question="(stated: the findings on the rows Explore may read, each tied to its lever)",
        options=(
            ContractOption("in_fold_rule", "Pull a lever as an in-fold rule",
                           "Levers are usually pulled by hand from plots of every row",
                           {"prediction": "Sound: the resampling repeats the rule, so its optimism "
                                          "is in the corrected score",
                            "inference": "Not offered: under inference each form is declared"},
                           {"prediction": "recommended", "inference": "not_offered"}),
            ContractOption("by_hand", "Pull a lever by hand after an outcome view",
                           "Common practice in both purposes",
                           {"prediction": "Recorded as outside the corrected score; a holdout "
                                          "covers it",
                            "inference": "Recorded and disclosed as made after the view (forking "
                                         "paths); never blocked"},
                           {"prediction": "available", "inference": "available"}),
        ),
        storyboard=("read the rows Explore may read", "state each finding with its lever",
                    "record each outcome view opened", "offer the in-fold rule first"),
        relations=(
            Relation("implies", "outcome_view_recorded",
                     "an outcome view opened is recorded as looked at, and the methods text "
                     "discloses it (forking paths)", condition="an outcome view is opened",
                     enforced_by=f"{here}:view_sentence", id="outcome_views_recorded"),
            Relation("implies", "in_fold_rule_first",
                     "each lever is offered first as an in-fold rule the resampling repeats",
                     purposes=("prediction",), enforced_by=f"{here}:form_lever",
                     id="levers_in_fold_first"),
            Relation("implies", "outside_corrected_score",
                     "a lever set by hand after an outcome view is recorded as outside the "
                     "corrected score", purposes=("prediction",),
                     condition="a lever on a viewed column changed after the view",
                     enforced_by=f"{here}:hand_levers", id="by_hand_outside_corrected"),
            Relation("implies", "holdout_covers",
                     "the held-out rows, never viewed, cover a lever set by hand",
                     purposes=("prediction",), condition="rows held out",
                     enforced_by=f"{here}:hand_levers", id="holdout_covers_hand_levers"),
            Relation("implies", "quality_by_group",
                     "data quality is compared across sociodemographic groups (TRIPOD+AI 7)",
                     enforced_by=f"{here}:_quality_finding", id="quality_across_groups"),
        ),
        sources=("Gelman & Loken 2013, The garden of forking paths",
                 "Harrell, Regression Modeling Strategies, Validation",
                 "Moscovich & Rosset, JRSS B 2022;84:1474", "TRIPOD+AI item 7"),
        sentence=f"{here}:view_sentence"))


_register_contract()

__all__ = ["EXPLORE_READS", "ExploreArtifact", "ExploreFinding", "HandLever", "explore_sentence",
           "explore_stage", "form_lever", "hand_levers", "lever_answers", "relationship_points",
           "subgroup_candidates", "view_sentence", "view_standing"]

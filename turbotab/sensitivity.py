"""turbotab.sensitivity — run it the other way, and say whether it mattered.

`MISC-014`. `DOMAIN_SCIENCE.md` §03 listed the sensitivity fork as primitive 4
and called it *"currently absent from the app entirely."* That clause was
false: `ml/sensitivity.py` has 132 lines, `pages/08_Sensitivity_Analysis.py`
has 568, `FEATURE_PARITY.md` lists it as shared, and `ml/publication.py` writes
it into the manuscript. The clause is corrected in that file; this module is
what makes the corrected sentence true of **Guided**.

Routed rather than rebuilt. Classic's version works and this one reuses its
vocabulary; what it does not reuse is Classic's two landmines, and both
exclusions are structural rather than careful.

## `STATE-013` is not inherited, and it cannot be

> *Seed sensitivity pools train+val+test and re-splits, dissolving the lockbox
> for numbers that appear beside lockbox-derived metrics.*

Classic re-splits per variation. **This module never splits.** It takes the
recorded lockbox, and both arms fit on `training_rows` and score on the sealed
labels — the same rows, in the same roles, in both arms. That is not a rule
this module follows; it is the only thing it can do, because it re-enters
`training.train` against a project whose lockbox it copies unchanged and never
touches a splitter at all. The seal exists precisely so that a number computed
twice is comparable, and a fork that re-split would be comparing two studies.

## `STATE-034` is not inherited, and this is where the discipline is

> *Two independent robustness verdict systems are shown side by side and can
> contradict each other.*

Classic renders coefficient-of-variation bands (<2% highly robust, <5%
moderately robust, …) beside absolute-range bands (<0.03 "publication-ready
without caveat", …). Two ladders, both invented, and a run can land on
different rungs.

**This module has no ladder.** It reports one thing, and it is a fact rather
than a grade: **did the substantive conclusion change?** For a model
comparison the substantive conclusion is *which model came first*, which is
checkable without choosing a threshold, and the metric under each arm is
reported beside it as a number the reader compares themselves. No band, no
adjective, no ✅, no "publication-ready" — the app does not know what would be
publishable, and `CLINICAL_SURVEY_PACK.md` §A5.4's own framing is the one
adopted here:

> *"If the substantive conclusion is the same under both, the dispute is moot
> for your paper and you can say that in one sentence — which is a much
> stronger position than picking a side."*

That is what this reports: whether the dispute is moot **for this study**. Not
whether the study is good.

## Why the axis is imputation and not the seed

One axis, and it is justified from the research files rather than picked for
convenience. Two packs name it:

- `METABOLOMICS_PACK.md` §260, marked ★ in the pack's own text: *"Always run
  the primary analysis under two imputation schemes and report whether
  conclusions change. This sensitivity analysis is the single highest-value
  thing a tool can add here — cheap, almost never done, and it directly
  answers the reviewer's objection."*
- `CLINICAL_SURVEY_PACK.md` §227, on MNAR: *"Both require untestable
  assumptions; the recommended practice is sensitivity analysis across
  plausible assumptions, not selecting one 'correct' method."*

And `DOMAIN_SCIENCE.md`'s own evidence-badge table already commits the app to
it: a **DISPUTED** item is *"never defaulted silently. Both sides stated. A
sensitivity analysis offered."* The missingness route is the one recorded
decision on this journey where the app has both a user's answer and a named,
eligible alternative sitting in `missingness.STRATEGIES_BY_BRANCH` — so it is
the axis the app can fork honestly today.

The seed axis is Classic's and is deliberately **not** ported. It is the one
where re-splitting is the tempting implementation, it answers a narrower
question, and no pack names it.

## What this does NOT do

It does not decide, recommend, or rank the two arms. It does not run when
nothing was recorded to fork on — silence, not a fork over a default nobody
chose. It does not write to the project: the alternative arm is fitted against
a **copy**, so the recorded plan is the one the user chose and the fork cannot
become the analysis by accident.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from turbotab import missingness as _miss

#: The alternative arm, per recorded strategy. Chosen so the two arms differ in
#: the ASSUMPTION rather than in the arithmetic — median and mean are both
#: single imputation under MCAR and forking between them answers nothing, so
#: each fill forks against a genuinely different assumption about why the value
#: is absent.
#:
#: Read against `missingness.STRATEGIES_BY_BRANCH` at call time, never trusted
#: from here: a pairing whose partner is not eligible on that column's branch is
#: dropped rather than declared.
#: **ONLY STRATEGIES WHOSE WHOLE EFFECT IS IN THE PIPELINE**, and the first
#: version of this table got that wrong in a way worth recording, because it is
#: the defect this project keeps finding in itself.
#:
#: It paired every fill against `INDICATOR_AND_IMPUTE`. Driven, both arms
#: produced a matrix of exactly 458 columns and every model's metric differed
#: by exactly 0.0000 — and the reason is `GUIDED-089` again: the indicator half
#: of that strategy is ROW-LOCAL and is written into `project.df` by
#: `route_missingness` at Preprocess time. Swapping the record on a copy leaves
#: it out. The alternative arm's record said *add a was-it-missing column and
#: fill the value*, its pipeline only filled, and the module would have reported
#: *this choice changes nothing* about a choice it had not made. A confident
#: "robust" produced by not having varied anything is worse than silence.
#:
#: So the rule is structural: a fork is eligible only when both arms defer
#: entirely to `pipeline_plan`, which is what makes a record swap a complete
#: swap. `_arms_differ` checks it again at run time on the actual matrices,
#: because a table is a claim and this one was wrong once.
COUNTERPART = {
    # Single vs multiple imputation, which is what CLINICAL_SURVEY_PACK's
    # missing-data section is actually about: the median is univariate and
    # assumes the value is missing at random given nothing, MICE models it
    # conditional on the other columns. Different assumptions, both untestable
    # from the data, and both fitted inside the fold.
    _miss.IMPUTE_MEDIAN: _miss.IMPUTE_MICE,
    _miss.IMPUTE_MEAN: _miss.IMPUTE_MICE,
    _miss.IMPUTE_MICE: _miss.IMPUTE_MEDIAN,
    # NOT FORKED, each for a stated reason rather than by omission:
    #
    # `INDICATOR_AND_IMPUTE` and `INDICATOR` and `EXPLICIT_CATEGORY` are
    # row-local in whole or in part — see above. Forking them needs the
    # alternative WORKING TABLE, not just the alternative record, and that is a
    # larger piece than this loop: it means re-running Preprocess against a
    # counterfactual plan and keeping the two frames apart. Filed rather than
    # half-built.
    #
    # `LEAVE` against a fill is a real fork and is also not available here: it
    # is honored by some estimators and not others, so the two arms would
    # differ per model for a reason that is not the axis (`GUIDED-095`'s
    # divergence machinery exists for exactly that) and the comparison would
    # not be about missingness.
    #
    # `IMPUTE_MODE` has no wholly-stateful counterpart on the categorical
    # branch: its alternatives there are `EXPLICIT_CATEGORY` and the indicator
    # pair, all row-local. A categorical-only project therefore gets no fork,
    # and gets silence rather than a fork it cannot honestly run.
}

#: The evidence status of the FORK ITSELF, not of either arm. Both packs
#: describe running it as recommended practice; neither settles which arm is
#: right, which is the whole reason a fork is the answer.
SOURCES = {
    "why_fork": {
        "source": "research/METABOLOMICS_PACK.md#03 · Missing data",
        "evidence_status": "CONVENTION",
        "claim": ("Always run the primary analysis under two imputation "
                  "schemes and report whether conclusions change. This "
                  "sensitivity analysis is the single highest-value thing a "
                  "tool can add here — cheap, almost never done, and it "
                  "directly answers the reviewer's objection."),
    },
    "why_not_one_answer": {
        "source": "research/CLINICAL_SURVEY_PACK.md#A2 · ★ Missing data — where TurboTab differentiates itself",
        "evidence_status": "DISPUTED",
        "claim": ("Selection models and pattern-mixture models both require "
                  "untestable assumptions; the recommended practice is "
                  "sensitivity analysis across plausible assumptions, not "
                  "selecting one 'correct' method."),
    },
    "what_to_report": {
        "source": "research/CLINICAL_SURVEY_PACK.md#B4 · ★ Ordinal vs interval — the long-running dispute",
        "evidence_status": "CONVENTION",
        "claim": ("Whichever you choose, run the other as a sensitivity "
                  "analysis and say so. If the substantive conclusion is the "
                  "same under both, the dispute is moot for your paper and "
                  "you can say that in one sentence — which is a much "
                  "stronger position than picking a side."),
    },
}


def fork(project: Any) -> Optional[Dict[str, Any]]:
    """The one axis this project can be forked on, or `None`.

    `None` means there is nothing recorded to fork — no missingness
    declaration, or none whose alternative is eligible on its own branch.
    Silence rather than a fork over a default nobody chose: `LOOP.md`'s rule
    that the app may be silent, and may refuse, and must never assert
    something false.
    """
    recorded = getattr(project, "missingness", None) or []
    if not recorded:
        return None

    swaps: List[Dict[str, Any]] = []
    for record in recorded:
        strategy = record.get("strategy")
        other = COUNTERPART.get(strategy)
        if other is None:
            continue
        # An alternative that is not offered on this column's branch is not an
        # alternative. Dropped rather than declared — a fork whose other arm
        # the app would refuse to record is a fork it cannot honestly run, and
        # the eligibility question is answered by `missingness`'s own table
        # rather than by a second list here.
        allowed = _miss.STRATEGIES_BY_BRANCH.get(record.get("branch") or "")
        if allowed is None or other not in allowed:
            continue
        swaps.append({"column": str(record["column"]), "recorded": strategy,
                      "alternative": other,
                      "recorded_label": _miss.strategy(strategy)["label"],
                      "alternative_label": _miss.strategy(other)["label"]})

    if not swaps:
        return None
    return {
        "axis": "missingness",
        "title": "Missing values, handled the other way",
        "swaps": swaps,
        "n_columns": len(swaps),
        "because": (
            f"{len(swaps)} column(s) carry a recorded answer about missing "
            f"values, and each has an eligible alternative that assumes "
            f"something different about why the value is absent. Neither "
            f"assumption is testable from this data."),
        **SOURCES["why_fork"],
    }


def _counterfactual(project: Any, swaps: List[Dict[str, Any]]) -> Any:
    """The same project with the other arm recorded, as a COPY.

    Shallow, so the frame and the lockbox are shared rather than re-derived —
    which is exactly the point: both arms score the same held-out rows, and
    there is no code path here that could produce a different split.
    `missingness` is the only attribute replaced, so a divergence between the
    arms can only have come from the axis.
    """
    other = copy.copy(project)
    swapped = {s["column"]: s["alternative"] for s in swaps}
    plan = []
    for record in (getattr(project, "missingness", None) or []):
        entry = dict(record)
        key = swapped.get(str(entry.get("column")))
        if key is not None:
            # The WHOLE record is re-declared from the alternative, not just
            # its `strategy` field: `defers`, `fit_on`, `label` and `sentence`
            # all describe the strategy, and swapping one of five would produce
            # a record whose sentence disagrees with what it fits — which is
            # `GUIDED-089` reintroduced inside the module that exists to
            # compare two honest plans.
            spec = _miss.strategy(key)
            entry.update(
                strategy=key, label=spec["label"], because=spec["because"],
                defers=spec["defers"],
                fit_on=("training folds only" if spec["defers"]
                        else "row-local, applied now"),
                sentence=_miss.sentence_for(entry["column"],
                                            entry.get("branch") or "numeric",
                                            key))
        plan.append(entry)
    other.missingness = plan
    return other


def _arms_differ(project: Any, other: Any, model_key: str):
    """Did the swap change what the model sees? `(bool, why_not)`.

    Transforms the training rows through each arm's preprocessing — the
    pipeline minus the estimator — and compares the resulting matrices. Two
    ways to differ, and both count: a different shape (a strategy that adds a
    column) or different values in the same shape (a different fill).

    Returns `(True, "")` when it cannot tell. **Deliberately**: an inability to
    check is not evidence that the arms are identical, and refusing on it would
    turn a diagnostic failure into a claim about the study. The comparison
    below is the one that catches the real case.
    """
    import numpy as np
    from turbotab import pipeline_plan as _plan_mod
    from turbotab import training as _training

    try:
        rows = project.training_rows
        target = str(project.target)
        group_col = (project.grain or {}).get("group_col")
        X = _training._feature_frame(rows, target, group_col)
        y = rows[target]
        seen = []
        for arm in (project, other):
            frame = _training._feature_frame(arm.working_table, target, group_col)
            pipe = _plan_mod.compose(arm, model_key, frame).build(
                _NullEstimator())
            pipe.fit(X, y)
            seen.append(pipe[:-1].transform(X))
    except Exception:                                       # pragma: no cover
        return True, ""

    a, b = seen
    if a.shape != b.shape:
        return True, ""
    if list(getattr(a, "columns", [])) != list(getattr(b, "columns", [])):
        return True, ""
    if not np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                       equal_nan=True):
        return True, ""
    return False, (
        "Both ways of handling the missing values produced the same feature "
        "matrix, so there is nothing to compare and no result is reported. "
        "Reporting 'the conclusion does not change' here would be a claim "
        "about a choice that was never actually varied.")


class _NullEstimator:
    """Stands in for the model so the preprocessing can be fitted without one.

    Not `DummyClassifier`: this is used for both task types and must not impose
    a task, and it never predicts anything — only the steps before it are read.
    """

    def fit(self, X, y=None):
        return self

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


def run(project: Any, model_keys: List[str], *, seed: int = 42
        ) -> Optional[Dict[str, Any]]:
    """Fit both arms over the sealed split and report whether it mattered.

    Returns `None` when there is nothing to fork. Otherwise the two arms, the
    per-model metric under each, and **one** statement about the conclusion.
    """
    from turbotab import training as _training

    spec = fork(project)
    if spec is None:
        return None
    if not model_keys:
        return None

    counterfactual = _counterfactual(project, spec["swaps"])

    # THE ARMS MUST ACTUALLY DIFFER, checked before anything is fitted.
    #
    # This is the guard the first version of this module needed and did not
    # have. `no change under either handling` is a strong claim, and produced
    # by an alternative arm that never reached the fit it is a false one — the
    # most expensive kind, because it reads as reassurance. Checked on the
    # matrices rather than on the record, since the record is precisely what
    # was already known to differ.
    differ, why = _arms_differ(project, counterfactual, model_keys[0])
    if not differ:
        return {"axis": spec["axis"], "unavailable": why}

    primary = _training.train(project, model_keys, seed=seed)
    other = _training.train(counterfactual, model_keys, seed=seed)

    # THE SAME SPLIT, ASSERTED RATHER THAN ASSUMED. `STATE-013` is a landmine
    # because nothing in Classic checks it; here the check is in the path that
    # produces the number, so the failure would be a refusal rather than a
    # quietly incomparable table.
    if (primary.n_train, primary.n_test) != (other.n_train, other.n_test):
        return {"axis": spec["axis"], "unavailable": (
            "The two arms did not score the same rows, so their numbers are "
            "not comparable and none is reported.")}

    metric = _headline_metric(primary)
    if metric is None:
        return {"axis": spec["axis"], "unavailable": (
            "No model produced a finite score under both arms, so there is "
            "nothing to compare.")}

    rows, ranked = [], {}
    for arm_name, run_obj in (("recorded", primary), ("alternative", other)):
        for result in run_obj.results:
            value = (result.metrics or {}).get(metric)
            if value is None:
                continue
            ranked.setdefault(arm_name, []).append((result.key, float(value)))
    for arm in ranked:
        ranked[arm].sort(key=lambda kv: -kv[1] if _higher_is_better(metric)
                         else kv[1])

    for result in primary.results:
        a = (result.metrics or {}).get(metric)
        b = next((r.metrics.get(metric) for r in other.results
                  if r.key == result.key), None)
        if a is None or b is None:
            continue
        rows.append({"key": result.key, "name": result.name,
                     "recorded": round(float(a), 4),
                     "alternative": round(float(b), 4),
                     "difference": round(float(b) - float(a), 4)})

    return {
        **spec,
        "metric": metric,
        "n_train": primary.n_train, "n_test": primary.n_test,
        "rows": rows,
        "conclusion": _conclusion(ranked, rows, metric, spec),
        "sources": SOURCES,
    }


#: Which metric the comparison is about, in the order the run reports them.
#: ONE metric, because two would be two verdict systems by another name and
#: `STATE-034` is what that looks like when it ships. The names are `ml.eval`'s
#: own keys, taken from that module rather than guessed — a second spelling
#: here would silently report *nothing to compare* on every project.
_HEADLINE = {"classification": ("ROC-AUC", "PR-AUC", "F1", "Accuracy"),
             "regression": ("R2", "RMSE", "MAE")}
_LOWER_IS_BETTER = frozenset({"RMSE", "MAE", "MedianAE", "LogLoss", "MSE"})


def _higher_is_better(metric: str) -> bool:
    return metric not in _LOWER_IS_BETTER


def _headline_metric(run_obj: Any) -> Optional[str]:
    for name in _HEADLINE.get(run_obj.task_type, ()):
        if any((r.metrics or {}).get(name) is not None for r in run_obj.results):
            return name
    return None


def _conclusion(ranked: Dict[str, List], rows: List[Dict[str, Any]],
                metric: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    """ONE statement, and it is a fact rather than a grade.

    *Did the substantive conclusion change?* For a model comparison the
    substantive conclusion is which model came first, which is checkable
    without inventing a band. The largest metric difference is reported beside
    it as a NUMBER the reader compares against what would matter in their
    field — the app does not know that, and `STATE-034` is what happens when a
    tool pretends it does.
    """
    a = (ranked.get("recorded") or [None])[0]
    b = (ranked.get("alternative") or [None])[0]
    if a is None or b is None:
        return {"changed": None, "sentence": (
            "Neither arm produced a ranking, so whether the conclusion is "
            "stable under this choice is not established.")}

    same = a[0] == b[0]
    biggest = max((abs(r["difference"]) for r in rows), default=0.0)
    columns = ", ".join(f"`{s['column']}`" for s in spec["swaps"][:3])
    more = "" if len(spec["swaps"]) <= 3 else f" and {len(spec['swaps']) - 3} more"

    if same:
        sentence = (
            f"Handling missing values the other way on {columns}{more} does "
            f"not change which model ranks first: {a[0]} leads under both. "
            f"The largest change in {metric} for any model is "
            f"{biggest:.4f}.")
    else:
        sentence = (
            f"Handling missing values the other way on {columns}{more} "
            f"changes which model ranks first: {a[0]} under the recorded "
            f"plan, {b[0]} under the alternative. The largest change in "
            f"{metric} for any model is {biggest:.4f}.")
    return {"changed": not same, "leader_recorded": a[0],
            "leader_alternative": b[0], "largest_difference": round(biggest, 4),
            "sentence": sentence,
            **SOURCES["what_to_report"]}


def methods_sentence(result: Optional[Dict[str, Any]]) -> Optional[str]:
    """The line for the manuscript. States what was varied and what happened,
    and stops — the interpretation is the author's, per `draft.py`'s rule that
    the app never speaks in the user's name."""
    if not result or result.get("unavailable") or not result.get("conclusion"):
        return None
    swaps = "; ".join(
        f"{s['column']}: {s['recorded']} → {s['alternative']}"
        for s in result["swaps"])
    return (f"A sensitivity analysis re-fitted every model on the same "
            f"training rows and scored it on the same held-out rows with the "
            f"missing-value handling varied ({swaps}). "
            f"{result['conclusion']['sentence']}")

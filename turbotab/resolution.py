"""turbotab.resolution — what a holdout this size can resolve.

`GUIDED-102`, and it is the case where a **shipped step contradicts its own
specification** rather than a future step being unbuilt. `engine.draw_holdout`
takes a constant `0.15` with no *n* term. Driven on
`metabolomics_untargeted.csv` — n=80, `PRODUCT_VISION.md`'s own worked case —
the seal draws **11 held-out rows** and says *"11 rows (15%) are held out and
will not be looked at again until the models are scored."* True, and the only
thing said.

`PRODUCT_VISION.md` §04 specifies the missing half:

> **State the instrument's resolution and let the researcher judge their claim
> against it.**

## The four rules, because the difference between the correct form and the
## anti-pattern is the entire content of this module

**1 · It never says "don't."** No refusal, no blocked step, no gate, no
severity. A researcher who wants a holdout at n=80 gets one. The app states
what that study can see and records the sentence.

**2 · It states the INSTRUMENT's resolution, never a verdict on the claim.**
*"A metric estimated on 11 rows has an interval spanning most of the unit
interval"* is arithmetic over quantities the app already holds. *"This study is
underpowered"* is **post-hoc power in a nicer suit** — listed flatly as an
anti-pattern in `research/METABOLOMICS_PACK.md` §10 — and writing it would be
committing a named error while presenting as the tool that catches them. We do
not hold the claim: at the seal we know the target, the task, the grain and the
eligibility — `project.seal_lockbox` refuses without the last two and cannot be
reached without the first two — and we do not know the expected effect size,
which predictor is the exposure of interest, or what magnitude would be
meaningful.

**The purpose was in that list and is not any more (`AUDIT-011`).**

    before: "at the seal we know the target, the task, the grain, the
             eligibility and the purpose"
    after:  the four the seal actually holds, and the purpose said separately
             for what it is

Question 2.5 — *what is this model for* — is asked and never required. Nothing
gates the seal on it, so `project.purpose` is `None` on any project whose user
walked past it, and this module is handed a frame at a seal that was drawn
without it. Listing it beside four preconditions asserted a fifth.

**And the two front doors diverge here, which is the finding underneath.** The
Guided door records the answer on `AnalysisProject.purpose`, and `purpose.py`'s
`CONSUMER` names the decisions that read it. The Streamlit workflow has no
purpose field: `model_purpose` is READ at three sites —
`pages/06_Train_and_Compare.py`, `ml/publication.py`, `ml/narrative_engine.py`
— and written at none, `_build_manuscript_context` included. So the one
production caller of `ml.imbalance_advice.advice` is
`pages/06_Train_and_Compare.py` passing `session_state["model_purpose"]`, and
it can only ever reach the `UNANSWERED` branch — one of `DOMAIN_SCIENCE.md`
§01.3's five inversions, unreachable on that door by construction rather than
by the user's silence.

**`AnalysisProject.purpose` is the authoritative record**, and it is the only
one. A claim about knowing the purpose is a claim about the Guided door alone,
and only after the user answered. Recording it where the Streamlit workflow
records its other answers is `AUDIT-011`'s other half; it lives in
`utils/session_state.py`, which this module does not own.

**3 · It fires only when stark, and the trigger is DERIVED.** A card on every
dataset is wallpaper (`PRODUCT_VISION.md` §04, *push the notable*). The trigger
below contains no number anybody picked — see `_push_because`.

**4 · It is recorded, and it reaches the manuscript.** Its natural home is the
seal extended: the sealed *n* is the input and the seal is the moment the
cohort stops changing. A statement *beside* the basis, never a fifth basis
value.

## What is deliberately NOT here

The full resolution statement — the metabolomics detectable-fold-change curve
(`METABOLOMICS_PACK.md` §751), nutrition's λ and its 1/λ² penalty
(`NUTRITION_PACK.md` §254), Riley's criteria-based minimum with anticipated R²
and prevalence (`CLINICAL_SURVEY_PACK.md` §A5.4) — is specified-unbuilt by
design and is a larger piece. Each needs an input the app does not hold at the
seal: an observed per-feature CV, a within-to-between variance ratio, an
anticipated R². This module computes only what the seal already knows.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# `AUDIT-020`. §A5.4's parameter count has exactly one implementation, and both
# doors read it. See `candidate_parameters` below for what it replaced.
from ml import sample_size as _sample_size

#: The two-sided normal multiplier for a 95% interval. Arithmetic rather than a
#: domain constant — it is not a threshold anybody in the research files chose,
#: and it is stated here so a reader can see which interval is meant.
Z95 = 1.96

#: The widest a proportion's standard error can be, at p = 0.5. Used because
#: the seal has no metric yet: the honest statement before a model is fitted is
#: the WORST case this holdout could produce, not a guess at the actual one.
WORST_CASE_SD = 0.5

#: Where each cited claim comes from. Read at build, never recollected — a
#: number marked `[verify-at-build]` may not ship as a constant, and none of
#: these is one: they are the pack's own words about method, not thresholds.
SOURCES = {
    "single_split": {
        "source": "research/CLINICAL_SURVEY_PACK.md#A5.5 Modeling practice",
        "evidence_status": "CONVENTION",
        "claim": (
            "Internal validation must resample the entire modeling pipeline. "
            "Bootstrap optimism correction is the recommended default because "
            "it uses all the data and has smaller variance than a single "
            "split; repeated k-fold is acceptable. A single train/test split "
            "is the weakest option and is discouraged at typical clinical "
            "sample sizes."),
    },
    "candidate_parameters": {
        "source": "research/CLINICAL_SURVEY_PACK.md#A5.4 Sample size",
        "evidence_status": "SETTLED",
        "claim": (
            "Candidate predictors count toward sample size even if they are "
            "later dropped: screening 40 variables and keeping 8 means sizing "
            "for 40, because data-driven selection consumes degrees of "
            "freedom whether or not it appears in the final model. Count "
            "PARAMETERS, not variables — a 5-level factor is 4."),
    },
}


def candidate_parameters(frame: pd.DataFrame, target: str,
                         group_col: Optional[str] = None,
                         excluded: Sequence[Any] = ()) -> Dict[str, Any]:
    """How many PARAMETERS the model may spend, not how many columns there are.

    `CLINICAL_SURVEY_PACK.md` §A5.4 is explicit and it is the count people get
    wrong: *count parameters, not variables — a 4-knot spline is 3, a 5-level
    factor is 4.* A numeric column is one; a categorical column with k observed
    levels is k−1, which is exactly what the one-hot encoder in
    `pipeline_plan` will produce.

    **Counted over the columns this app will actually hand the model** — the
    same frame `training.feature_frame` builds — rather than over the file, so
    the number describes the fit rather than the spreadsheet.

    ## `excluded`, and why a structural exclusion is not "later dropped"
    (`AUDIT-019`)

    Three sets never reach a model: the target, the grain's group column, and
    the per-row identifiers `identifiers.excluded` names. The first two were
    dropped here from the beginning; the third was added to
    `training.feature_frame` when `GUIDED-108` was fixed and **was not added
    here**, so the seal's methods sentence reported 344 candidate parameters
    on `survey_instrument.csv` where the models are handed 45. 299 of them
    were `respondent_id` — a column the app structurally refuses to encode,
    which spends no degrees of freedom, so counting it is not conservatism but
    a number about a fit nobody performed.

    The caller passes the set rather than this module deriving it, because
    `identifiers.excluded` answers about a PROJECT and this module answers
    about a frame. `project.seal_lockbox` is the one caller that has both.

    **This does not contradict §A5.4's ⚠ clause** — *candidate predictors
    count toward sample size even if they are later dropped; if you screen 40
    and keep 8 you must size for 40.* That clause is about **data-driven
    selection**, which is PROBAST's signal because it reads the OUTCOME and so
    borrows precision the reported model does not pay for. Identifier
    exclusion reads no outcome at all: it is `nunique == nrow` on the training
    rows, a structural fact about the column. A column the app never offers
    the model was never a candidate, the way a constant column is not one. A
    user who disagrees puts it back with `project.keep_identifier`, and then
    it is in `excluded` no longer and counts again — the count follows what
    the app will hand the model, which is the only definition that stays true.

    **AND THE ARITHMETIC IS `ml.sample_size`'s** (`AUDIT-020`). The two
    fixes are independent and they compose: that one made a k-level factor cost
    k−1 in ONE place, because the Classic door's `ml/dataset_profile.py` was
    charging one parameter per column and the two doors reported different
    events-per-parameter for the same file; this one decides WHICH columns are
    counted at all. Merged from two worktrees that could not see each other.

    The excluded columns are **reported, not silently removed**
    (`PRODUCT_VISION.md`, *the shelf is never shortened*): `excluded_columns`
    names them and `excluded_parameters` is what they would have added, so a
    reader who counts columns in the data dictionary can reconcile the two
    numbers instead of finding them merely different.
    """
    set_aside = {str(c) for c in excluded}
    drop = ({str(target)} | ({str(group_col)} if group_col else set())
            | set_aside)
    numeric, categorical, per_column = 0, 0, []
    spent_on_excluded, excluded_seen = 0, []
    for column in frame.columns:
        name = str(column)
        if name in drop and name not in set_aside:
            continue
        series = frame[column]
        if isinstance(series, pd.DataFrame):                # pragma: no cover
            continue
        if pd.api.types.is_numeric_dtype(series):
            spent, kind = 1, "numeric"
        else:
            levels = int(series.dropna().nunique())
            spent, kind = max(0, levels - 1), f"{levels} observed levels"
        if name in set_aside:
            # COUNTED AND SUBTRACTED, so the number that left is stated.
            spent_on_excluded += spent
            excluded_seen.append(name)
            continue
        if kind == "numeric":
            numeric += 1
        else:
            categorical += spent
        per_column.append({"column": name, "parameters": spent, "kind": kind})
    per_column.sort(key=lambda d: -d["parameters"])
    return {
        "total": int(numeric + categorical),
        "numeric": int(numeric),
        "from_categorical": int(categorical),
        "excluded_columns": sorted(excluded_seen),
        "excluded_parameters": int(spent_on_excluded),
        "largest": per_column[:5],
        **SOURCES["candidate_parameters"],
    }


def _widest_interval(n_test: int) -> Optional[float]:
    """The WIDTH of the widest 95% interval a proportion on `n_test` rows can
    have. `None` below one row, because a width computed from nothing is a
    number that asserts precision it does not have."""
    if n_test < 1:
        return None
    return float(2 * Z95 * WORST_CASE_SD / math.sqrt(n_test))


def informative_range(n_classes: Optional[int]) -> float:
    """The width of the range a discrimination statistic can say anything in.

    **`GUIDED-125`, and this is the generalization rather than a special case
    for k = 3.** A classifier that guesses is right `1/k` of the time, so the
    distance between chance and perfect is `1 − 1/k`. For two classes that is
    0.5, which is what the original derivation used and why it was correct for
    every fixture that existed when it was written; for three it is 0.667, and
    the boundary moves DOWN with k rather than up.

    `1.0` where the arity is unknown or the task is not classification — the
    widest possible range, so an unknown k can only make the card fire LATER
    and never earlier. An unknown that made a warning more likely would be the
    app asserting something about a study it could not see.
    """
    if not n_classes or n_classes < 2:
        return 1.0
    return 1.0 - (1.0 / float(n_classes))


def _push_because(task_type: str, n_test: int, events_test: Optional[int],
                  non_events_test: Optional[int],
                  n_classes: Optional[int] = None,
                  thin_classes: int = 0) -> Optional[str]:
    """Whether this is stark, and **the trigger contains no picked number.**

    Three conditions, each of which is a fact rather than a threshold:

    **A · The interval is wider than the whole distance from chance to
    perfect.** A discrimination statistic runs from 0.5 (a coin) to 1.0
    (perfect), so the range that carries any information is 0.5 wide. When the
    widest 95% interval this holdout can produce exceeds that, the holdout
    cannot distinguish a useless model from a perfect one. The 0.5 comes from
    the interval named, and the range from `informative_range` — nothing here
    was chosen for taste. **It resolves to n_test ≤ 15 for two classes and
    n_test ≤ 9 for three**, and the boundary moves DOWN with k because a
    three-class model has more room between chance and perfect, so a given
    interval width says relatively less.

    `GUIDED-125` corrected this: the first version compared against a constant
    0.5, which is `1 − 1/k` at k = 2 and was right for every fixture that
    existed when it was written.

    **B · A class is missing or all-but-missing from the holdout.** With fewer
    than two of either class, sensitivity or specificity is undefined rather
    than imprecise. Not a threshold: two is the smallest number from which a
    proportion has any spread at all.

    **Both are about the HOLDOUT**, which is this card's subject.

    A third condition was measured and **deliberately rejected**: *more
    candidate parameters than training rows*. It is true arithmetic — a model
    cannot estimate p parameters from fewer than p observations — but driven
    across the fixtures it fired on five of six, which is wallpaper, and it
    fired for a reason that is not about resolution at all. The counts were
    dominated by per-row identifier columns the app was handing the model as
    candidate predictors: `respondent_id` was 299 of `survey_instrument.csv`'s
    344 parameters, `admission_id` 159 of `leaky_sepsis.csv`'s 164. That is a
    real defect and it was filed as one (`GUIDED-108`) rather than being
    laundered through this card.

    **Both halves of that are now closed and the trigger stays rejected.**
    `GUIDED-108` stopped the identifiers reaching the model and `AUDIT-019`
    stopped them reaching this count, so the same two fixtures now report 45
    and 5. The trigger is still not reinstated: what was measured was that it
    is not about resolution, and a smaller number does not make *p > n* a
    statement about what a holdout can see. The parameter count is still
    **reported** in every statement, because `PRODUCT_VISION.md` §04 names it
    as an input; it just does not decide whether the card is raised.

    `None` means do not push. The statement is still computed and still
    recorded — this decides only whether the app raises it unprompted, which is
    `PRODUCT_VISION.md` §04's *push the notable, pull the rest*.
    """
    widest = _widest_interval(n_test)
    span = informative_range(n_classes)
    if widest is not None and task_type == "classification" and widest > span:
        chance = 1.0 / float(n_classes) if n_classes and n_classes >= 2 else 0.5
        return (
            f"the widest 95% interval a metric on {n_test} held-out rows can "
            f"have is {widest:.2f} wide, which is more than the whole distance "
            f"from chance to perfect — with {n_classes or 2} classes a model "
            f"that guesses is right {chance:.0%} of the time, so that distance "
            f"is {span:.2f}")
    if task_type == "classification":
        # CONDITION B, ALSO GENERALIZED (`GUIDED-125`). The first version
        # checked the minority class and its complement, which is every class
        # when k = 2 and two of three when k = 3 — so a three-class holdout
        # missing its MIDDLE class passed silently. `thin_classes` is the count
        # of classes with fewer than two held-out rows, whatever k is.
        if thin_classes:
            return (f"{thin_classes} of the {n_classes or 2} outcome classes "
                    f"have fewer than two held-out rows, so a per-class rate "
                    f"is undefined rather than imprecise")
        if events_test is not None and events_test < 2:
            return (f"the held-out set contains {events_test} of the outcome, "
                    f"so sensitivity is undefined rather than imprecise")
        if non_events_test is not None and non_events_test < 2:
            return (f"the held-out set contains {non_events_test} rows without "
                    f"the outcome, so specificity is undefined rather than "
                    f"imprecise")
    return None


def statement(frame: pd.DataFrame, target: str, task_type: str,
              labels: List[Any], group_col: Optional[str] = None,
              excluded: Sequence[Any] = ()) -> Dict[str, Any]:
    """What this holdout can resolve, computed at the seal.

    `labels` is the lockbox's own row labels, so the counts describe the split
    that was actually drawn rather than the fraction that was requested — the
    same reason `draw_holdout` reports the achieved row fraction
    (`IMPORT-255`).

    `excluded` is the identifier columns this project keeps out of the models
    — `identifiers.excluded(project)`. It is an argument rather than something
    this module derives because the exclusion is a property of the project and
    this function is given a frame. `AUDIT-019`, and `candidate_parameters`
    carries the reasoning.
    """
    has_y = frame[target].notna()
    sealed = set(labels)
    is_test = pd.Series([i in sealed for i in frame.index], index=frame.index)
    n = int(has_y.sum())
    n_test = int((has_y & is_test).sum())
    n_train = int((has_y & ~is_test).sum())

    events_test = non_events_test = events_train = None
    minority = None
    n_classes = None
    thin_classes = 0
    if task_type == "classification":
        counts = frame.loc[has_y, target].value_counts()
        # THE ARITY, and it is a count rather than a label — `GUIDED-125` needs
        # k and the archive guard refuses class VALUES in a serialized project
        # (`GUIDED-102`'s own correction), so what travels is how many.
        n_classes = int(len(counts)) or None
        if len(counts) >= 1:
            minority = counts.index[-1]
            in_test = frame.loc[has_y & is_test, target]
            events_test = int((in_test == minority).sum())
            non_events_test = int(n_test - events_test)
            events_train = int(
                (frame.loc[has_y & ~is_test, target] == minority).sum())
            # EVERY CLASS, not just the minority and its complement. A class
            # absent from the holdout is absent whatever its rank.
            held = frame.loc[has_y & is_test, target].value_counts()
            thin_classes = int(sum(
                1 for level in counts.index if int(held.get(level, 0)) < 2))

    parameters = candidate_parameters(frame, target, group_col, excluded)
    widest = _widest_interval(n_test)
    because = _push_because(task_type, n_test, events_test, non_events_test,
                            n_classes, thin_classes)

    return {
        "n": n, "n_train": n_train, "n_test": n_test,
        "task_type": task_type,
        # THE CLASS LABEL IS NOT STORED, and the guard is what found it.
        # `archive.assert_no_participant_data` rejected a serialized project
        # carrying `responder` — a cell value from the table — and its
        # instruction is the right one: *persist decisions and inputs;
        # regenerate derivatives.* This statement is a derivative. Exempting
        # the value instead would have meant exempting every project's class
        # labels, which is the guard switched off rather than satisfied.
        #
        # Nothing is lost that a reader needs here: the COUNT is what the
        # holdout's resolution depends on, and which class is the positive one
        # is already carried, per model, on `ModelResult.positive_label` where
        # it is about a fitted thing rather than about the record.
        "events_held_out": events_test,
        "non_events_held_out": non_events_test,
        "events_in_training": events_train,
        # STATED, because a threshold that changes with the outcome's arity and
        # does not say so is a threshold nobody can check.
        "n_classes": n_classes,
        "chance": (None if not n_classes or n_classes < 2
                   else round(1.0 / n_classes, 4)),
        "classes_with_fewer_than_two_held_out": (
            None if task_type != "classification" else thin_classes),
        "informative_range": (None if task_type != "classification"
                              else round(informative_range(n_classes), 4)),
        # Counts only. `candidate_parameters` returns its badge and its
        # per-column breakdown for a caller that wants them; the RECORD keeps
        # the arithmetic, and the citation is served from `SOURCES` at read
        # time so one edit to the pack reference reaches every project rather
        # than only the ones sealed afterwards.
        # `excluded_columns` and `excluded_parameters` travel with the three
        # counts because without them `total` is a number a reader cannot
        # reconcile against their own column list. `AUDIT-019`.
        "parameters": {k: parameters[k] for k in
                       ("total", "numeric", "from_categorical",
                        "excluded_columns", "excluded_parameters")},
        # The step a single held-out row moves a proportion by. Exact counting,
        # and the most legible form of "what this instrument can resolve".
        "step_per_row": None if n_test < 1 else round(1.0 / n_test, 4),
        "widest_interval": None if widest is None else round(widest, 3),
        "push": because is not None,
        "because": because,
        "headline": _headline(task_type, n_test, widest, events_test,
                              n_classes),
        "sentence": _sentence(n, n_train, n_test, task_type, widest,
                              parameters["total"], events_test, n_classes,
                              parameters["excluded_columns"],
                              parameters["excluded_parameters"]),
        # SAID OUT LOUD, because the absence of this line is what would turn
        # the module into the anti-pattern it exists to avoid.
        "not_a_verdict": (
            "This is a statement about the instrument, not about your study. "
            "The app does not know your expected effect size, which predictor "
            "is your exposure of interest, or what difference would matter — "
            "so it cannot say whether this design is adequate for your "
            "question, and it does not try. What it can do is arithmetic over "
            "what it holds, and leave the judgment where it belongs."),
    }


def _headline(task_type: str, n_test: int, widest: Optional[float],
              events_test: Optional[int],
              n_classes: Optional[int] = None) -> str:
    where = f"{n_test:,} held-out row" + ("" if n_test == 1 else "s")
    if task_type == "classification" and events_test is not None:
        where += f" ({events_test:,} with the outcome)"
    if widest is None:
        return f"This seal holds out {where}."
    line = (f"A metric estimated on {where} carries a 95% interval up to "
            f"{widest:.2f} wide, on a scale of 0 to 1.")
    if task_type == "classification" and n_classes and n_classes >= 2:
        # `k` ON THE CARD. `GUIDED-125`.
        line += (f" With {n_classes} classes a model that guesses is right "
                 f"{1.0 / n_classes:.0%} of the time, so the range that "
                 f"carries information is {informative_range(n_classes):.2f} "
                 f"wide.")
    return line


def _sentence(n: int, n_train: int, n_test: int, task_type: str,
              widest: Optional[float], parameters: int,
              events_test: Optional[int],
              n_classes: Optional[int] = None,
              excluded_columns: Sequence[str] = (),
              excluded_parameters: int = 0) -> str:
    """The methods line. Reports what was done and what it can resolve, and
    stops there — no adequacy, no adjective, no recommendation.

    `AUDIT-019`: `parameters` is now the count over the frame the models are
    handed, and the clause that follows says which columns are not in it and
    what they would have added. Reporting the count without the exclusion
    would be a true number a reader cannot check; reporting the exclusion
    without the count was the false sentence this row was filed against.
    """
    line = (f"Of {n:,} rows with a value for the outcome, {n_test:,} were "
            f"sealed as a held-out set before exploration and {n_train:,} were "
            f"available for fitting")
    if task_type == "classification" and events_test is not None:
        line += f", with {events_test:,} of the held-out rows carrying the outcome"
    line += (f". {parameters:,} candidate predictor parameters were available "
             f"to the models, counted including any later dropped by feature "
             f"selection")
    if excluded_columns:
        named = ", ".join(str(c) for c in excluded_columns)
        line += (f", and excluding {len(excluded_columns):,} column"
                 f"{'' if len(excluded_columns) == 1 else 's'} "
                 f"({named}) whose every value is different, which the app "
                 f"does not hand the model and which would otherwise have "
                 f"added {excluded_parameters:,}")
    if widest is not None:
        line += (f"; a performance metric estimated on {n_test:,} rows carries "
                 f"a 95% interval up to {widest:.2f} wide")
        if task_type == "classification" and n_classes and n_classes >= 2:
            line += (f", against a range of {informative_range(n_classes):.2f} "
                     f"between chance ({1.0 / n_classes:.0%}, with "
                     f"{n_classes} classes) and perfect")
    return line + "."

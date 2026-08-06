"""turbotab.missingness — clause §07: routing by dtype **and** mechanism.

> Prediction is not inference, and the distinction is load-bearing: the
> missing-indicator method discouraged for causal estimation is defensible and
> often helpful for prediction under informative missingness.

That sentence is why this module cannot import a causal-inference default and
call the clause satisfied. The two branches fail in different ways and are
therefore separate objects here:

**Binary / categorical** — ask FIRST whether the missingness is informative
(*"could a blank here mean something?"*); in EHR data it usually is. Default to
an explicit `Missing` category or an indicator, which preserve the signal.
Imputing an informatively-missing field is a **blocker with typed
acknowledgment**, and the **stability assumption** — that missingness means the
same thing at deployment — is recorded as a methods assumption, because it may
not hold across sites.

**Numeric** — single vs multiple imputation and the strategy; fit **inside the
fold**; and **never place the outcome in the imputation model**, which is a
blocker in any configuration.

**Almost nothing here executes**, and that is clause §06 rather than a
limitation. Every routine below is stateful by the litmus — a median, a mode, a
MICE model and a frequency table are all statements about the column's
distribution — so this module produces **declarations** that per-model pipelines
fit inside training folds. The one exception is the explicit-`Missing` category
for a categorical column, which is row-local (a blank becomes a literal token,
using nothing but that row's own cell) and is marked so.

The consequence for the interface is the awkward part and is dealt with in
`plan_receipt`: the user answers several questions, the table does not change,
and the step still has to be believable.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from turbotab import exits as _exits

# The mechanism answers. `NOT_SURE` is first-class for the same reason it is on
# the grain question: the app cannot know, and a forced guess is worse than a
# recorded uncertainty.
INFORMATIVE = "informative"
NOT_INFORMATIVE = "not_informative"
NOT_SURE = "not_sure"
MECHANISMS = (INFORMATIVE, NOT_INFORMATIVE, NOT_SURE)

# What may be done about it, per branch.
EXPLICIT_CATEGORY = "explicit_category"      # row-local
INDICATOR = "indicator"                      # row-local
INDICATOR_AND_IMPUTE = "indicator_and_impute"  # BOTH — see below
IMPUTE_MODE = "impute_mode"                  # stateful
IMPUTE_MEDIAN = "impute_median"              # stateful
IMPUTE_MEAN = "impute_mean"                  # stateful
IMPUTE_MICE = "impute_mice"                  # stateful
LEAVE = "leave"                              # nothing, recorded

ROW_LOCAL_STRATEGIES = frozenset({EXPLICIT_CATEGORY, INDICATOR, LEAVE})

# ─────────────────────────────────────────────────────────────────────────────
# WHERE a deferred statistic is fitted, which is not the same as WHETHER it is
# ─────────────────────────────────────────────────────────────────────────────
#
# `AUDIT-028`. `defers` above answers clause §06's question — does this
# transform's output for row *i* depend on any other row? — and it answers it
# correctly. What it does NOT answer is the question the methods sentence was
# nevertheless using it to answer: over WHICH rows the statistic gets fitted.
#
# Every deferring strategy's sentence read *"within each training fold"*, and
# the Guided door has no folds. `turbotab/training.py` calls `pipe.fit(X_train,
# y_train)` once per model on the training partition; nothing in `turbotab/`
# imports `KFold`, `cross_val_score` or `cross_validate` at all. So a decision
# a user made in the Preprocess step was recorded — and exported into the
# manuscript's Missing Data section — claiming a resampled design that §A5.5
# calls acceptable, over a door that performs the single split §A5.5 calls "the
# weakest option ... discouraged at typical clinical sample sizes".
#
# The vocabulary and the values are `turbotab/selection.py`'s deliberately, and
# `test_the_two_scope_vocabularies_are_one` pins them together. `GUIDED-104`'s
# adjudication set the standard this follows: *"selection.declare exists
# precisely so a door that fits once can SAY train_rows instead of implying the
# stronger claim."*
#
# THE DEFAULT IS THE WEAKER CLAIM, and that is the whole design. `selection.py`
# defaults to `TRAIN_FOLDS` and the Guided door silently took it — which is
# `AUDIT-027`, still open. A caller who forgets to say must understate what
# happened, never overstate it: the governing rule permits silence and forbids a
# false assertion, so the direction of the default is not a style choice.
TRAIN_ROWS = "train_rows"
TRAIN_FOLDS = "train_folds"
FIT_SCOPES = (TRAIN_ROWS, TRAIN_FOLDS)

#: The clause each scope contributes to a deferred strategy's sentence.
_SCOPE_PHRASE = {
    TRAIN_ROWS: " once over the training rows (held-out rows excluded)",
    TRAIN_FOLDS: " within each training fold",
}

#: The machine-readable half of the same fact. Trap 7: the structured payload is
#: what everything downstream reads, so it must not be lossier — or falser —
#: than the prose beside it. This used to read "training folds only" for every
#: deferring strategy, from the same wrong premise as the sentence.
_SCOPE_FIT_ON = {
    TRAIN_ROWS: "training rows only",
    TRAIN_FOLDS: "training folds only",
}


def _checked_scope(scope: str) -> str:
    if scope not in FIT_SCOPES:
        raise MissingnessRefusal(
            f"scope must be {TRAIN_ROWS!r} or {TRAIN_FOLDS!r}; got {scope!r}. "
            f"There is no third option, and in particular there is no option "
            f"that fits over the whole table.")
    return scope

# THE COMPOUND ONE, and it is here because the card was already offering it and
# the record had nowhere to put it (`GUIDED-098`).
#
# `ml/missingness_plan.py` has always offered *"Impute, and record that it was
# missing"* on the numeric branch and *"Keep the absence as information"* on the
# binary one, and both said in their own decision sentence that the value is
# imputed within each training fold. `CARD_STRATEGY` mapped both to `INDICATOR`,
# whose sentence is *"the underlying value is left blank."* One click, two
# methods sentences, opposite claims — and after `GUIDED-095` the pipeline
# honors the record, so the fill the card promised does not happen.
#
# It is genuinely both halves of clause §06: the indicator is row-local and
# lands now, the fill is stateful and is fitted in the fold. Modeling it as
# either one alone is what produced the contradiction.
MIXED_STRATEGIES = frozenset({INDICATOR_AND_IMPUTE})


class MissingnessRefusal(Exception):
    """The step was asked for something clause §07 forbids."""


# ── the mechanism question, asked before the strategy ────────────────────────

MECHANISM_QUESTION = "Could a blank in `{column}` mean something?"

MECHANISM_WHY = (
    "In records collected during care, a missing value is often a decision "
    "rather than an accident — a test not ordered because the patient looked "
    "well is different from a test that was ordered and lost. If a blank "
    "carries information, filling it in throws that information away.")

MECHANISM_OPTIONS = (
    "Yes — a blank here means something",
    "No — these are accidents of collection",
    "I'm not sure",
)

MECHANISM_CONSUMER = (
    "The answer decides how this column is handled and what the methods section "
    "has to say. 'Yes' routes to an explicit Missing category or an indicator, "
    "which keep the signal, and makes plain imputation a blocked choice that "
    "needs a typed acknowledgment. It also records the stability assumption — "
    "that a blank will still mean the same thing wherever the model is "
    "deployed — as a stated limitation, because it may not hold across sites.")

# §07's own words, recorded when the mechanism is stated informative. Not a
# warning: an assumption, written where a methods section can quote it.
STABILITY_ASSUMPTION = (
    "This analysis assumes that a blank in `{column}` will mean the same thing "
    "wherever the model is used as it means here. That assumption is not "
    "checkable from this dataset — missingness patterns are a property of how "
    "a site collects data, and a model that reads a blank as a signal will read "
    "a differently-collected blank as the same signal.")

# The CONSEQUENCE. Plain imputation over a column the user has just said carries
# information is not a mistake the app can correct on their behalf — they may
# know something it does not — so it resolves or is attested, never a dead end.
INFORMATIVE_IMPUTATION_BLOCKER = (
    "You said a blank in `{column}` means something, and {strategy} would "
    "replace every one of those {n_missing:,} blanks with {filler}. The fact "
    "that the value was missing would no longer be in the data at all, and no "
    "model can recover it afterward.\n\n"
    "If that is what you want, say so and it is recorded as a stated "
    "limitation. If it is not, an explicit `Missing` category keeps the blank "
    "as its own answer and costs nothing.")

# THE WAY THROUGH DEPENDS ON THE BRANCH, and it used to not.
#
# One hand-written resolve exit offered `explicit_category` for every column,
# including numeric ones, where it is not an available strategy at all — so the
# blocker's own way out was a route the record refuses. It also carried no
# `retry`, unlike the attest exit beside it, so a client holding only the
# payload could read the offer and not take it: `GUIDED-072`'s unifying test,
# failed by the one exit that is supposed to be the easy answer.
#
# Both exits now carry a retry payload, and the resolve exit names a strategy
# that KEEPS THE SIGNAL on the branch it is offered for — which is the thing
# the blocker exists to protect.
_RESOLVE_BY_BRANCH = {
    "categorical": {
        "id": EXPLICIT_CATEGORY,
        "label": "Keep the blanks as their own category",
        "detail": "A blank becomes a literal `Missing` value, so the model can "
                  "use it the way it uses any other level."},
    "numeric": {
        "id": INDICATOR,
        "label": "Add a was-it-missing column and leave the value blank",
        "detail": "The fact that the value was absent becomes a column of its "
                  "own, so the model can use it — and the number itself is not "
                  "invented. Writing a category into a numeric column would "
                  "stop it being numeric."},
}


def blocker_exits(branch: str) -> tuple:
    resolve = _RESOLVE_BY_BRANCH.get(branch)
    exits = []
    if resolve is not None:
        # BOTH SPELLINGS, because the handler prefers `card_option` and the
        # Explore door sends one. A payload naming only `strategy` is merged
        # into a request that already carries the refused `card_option`, and
        # the refused one wins — see `card_option_for_strategy`.
        payload = {"strategy": resolve["id"]}
        card = card_option_for_strategy(resolve["id"])
        if card:
            payload["card_option"] = card
        exits.append({
            "id": resolve["id"], "kind": "resolve", "label": resolve["label"],
            "detail": resolve["detail"],
            "retry": {"payload": payload,
                      "how": "Sent again with this strategy in place of the "
                             "one that would erase the blanks.",
                      "typed": None}})
    exits.append(_exits.attest(
        "Fill them anyway — I know what these blanks are",
        "Recorded as a stated limitation: the missingness signal is "
        "removed deliberately, and the methods section says so.",
        _exits.ACKNOWLEDGE_SIGNAL_LOSS))
    return tuple(exits)


# `BLOCKER_EXITS` is bound at the FOOT of this module, not here.
# `blocker_exits` now reads `card_option_for_strategy`, which is defined beside
# its inverse further down — and a module-level call at this point runs before
# either that function or `CARD_STRATEGY` exists. Moving the binding is a
# two-line change; moving the join table up past six hundred lines of prose is
# not, and the join belongs with its inverse.

# ── the outcome inside an imputation scope · one question, two answers ───────
#
# **`AUDIT-005`.** This used to be one sentence ending *"there is no
# configuration in which this is acceptable, so it is not offered as a choice"*
# — and `research/CLINICAL_SURVEY_PACK.md` §A2 marks the opposite **[SETTLED]**:
#
# > *Imputing with the outcome EXCLUDED from the imputation model. Biases
# > associations toward the null. The outcome MUST be in the imputation model.*
#
# **Both are right, about different purposes**, which is
# `DOMAIN_SCIENCE.md` §01.3's whole subject: *the advice inverts.* Under
# prediction, an imputer fitted with the outcome in scope writes the outcome
# into the features and every number scored afterwards is scored against
# features that already encode the answer. Under inference, the imputation model
# is part of the estimation and omitting the outcome makes the imputed
# covariates conditionally independent of it, which shrinks the association
# toward the null — the classic Little–Rubin result the pack cites.
#
# **The defect was the ABSOLUTE, not the refusal.** The app records the purpose
# and this module never read it, so it asserted a universal it had the
# information to qualify. That is the governing rule failing in a refusal, and
# it is also an `AUDIT-008` instance: a capability the core exposes that the
# path needing it does not consult.
#
# The prediction branch is unchanged and is still a blocker. The inference
# branch does not become a blocker with a softer message — it stops being one,
# because under inference the thing being refused is the correct thing to do.

OUTCOME_IN_IMPUTATION_REFUSAL = (
    "The outcome `{target}` cannot be one of the columns the imputation model "
    "reads. An imputer fitted with the outcome in scope writes the outcome's "
    "own information into the feature columns, so every number scored "
    "afterwards is scored against features that already encode the answer. "
    "You recorded that this model is for PREDICTING an outcome for a new "
    "person, and at deployment that leak has nowhere to come from — the "
    "features would carry information the app will not have. So it is not "
    "offered as a choice here. "
    "(If you were estimating how strongly something is associated with the "
    "outcome, the answer would be the opposite one: the outcome belongs in the "
    "imputation model, and leaving it out biases the association toward the "
    "null. `research/CLINICAL_SURVEY_PACK.md` §A2.)")

# The same configuration, under the other purpose. Not a refusal — a note, and
# it is affirmative rather than permissive: the pack marks the inclusion
# REQUIRED, so an inference analysis that leaves the outcome out is the one
# making the error.
OUTCOME_IN_IMPUTATION_UNDER_INFERENCE = (
    "The outcome `{target}` is inside the imputation scope, and under an "
    "association objective that is correct rather than a leak. Excluding it "
    "makes the imputed covariates conditionally independent of the outcome and "
    "biases the association toward the null, which is why the clinical "
    "literature treats including it as required rather than as permitted. "
    "(Were this model for prediction, the same configuration would be refused: "
    "the imputer would write the outcome into features the app will not have "
    "at deployment.)")

OUTCOME_IN_IMPUTATION_EVIDENCE = {
    "prediction": "research/CLINICAL_SURVEY_PACK.md#A2 · ★ Missing data — where TurboTab differentiates itself",
    "inference": "research/CLINICAL_SURVEY_PACK.md#A2 · ★ Missing data — where TurboTab differentiates itself",
}


def outcome_in_scope(target: Optional[str],
                     purpose: Optional[str]) -> Dict[str, Any]:
    """What to say about the outcome sitting inside an imputation scope.

    `refuse` is the only field a caller must honor; the rest is what it says.
    `None` purpose refuses, and that is deliberate rather than conservative: the
    purpose question is a CHOICE the constitution says is always asked and never
    inferred, so an unanswered purpose is not evidence for the inference branch.
    Refusing there keeps the safe answer AND names the question that would
    change it, which is the recorded-absence rule rather than a shrug.
    """
    from turbotab import purpose as _purpose

    if purpose == _purpose.INFERENCE:
        return {"refuse": False, "purpose": purpose,
                "message": OUTCOME_IN_IMPUTATION_UNDER_INFERENCE.format(
                    target=target),
                "source": OUTCOME_IN_IMPUTATION_EVIDENCE["inference"],
                "evidence_status": "SETTLED"}
    message = OUTCOME_IN_IMPUTATION_REFUSAL.format(target=target)
    if purpose is None:
        message += (
            " The purpose question has not been answered on this project. It "
            "is the one question that would change this answer, and nothing in "
            "your data reveals it — so the safe branch stands until you say.")
    return {"refuse": True, "purpose": purpose, "message": message,
            "source": OUTCOME_IN_IMPUTATION_EVIDENCE["prediction"],
            "evidence_status": "SETTLED"}


# ── where a blank came from ──────────────────────────────────────────────────
#
# `GUIDED-166`. Explore asks the user what a blank MEANS — §07's mechanism
# question, the fork this whole module is built on. The impossibility card two
# cards later **makes blanks**: `project.set_impossible_missing` is real since
# L47, and on `clinical_labs.csv` it turns 4 entries of `sbp` outside 40–300
# mmHg into 4 NaNs. The column had **zero** blanks before that press and has 4
# after, and the Preprocess survey then asked *"could a blank in `sbp` mean
# something?"* about four blanks the app itself had written.
#
# Every honest answer to that question is wrong for those cells. They do not
# mean *not asked*, they do not mean *not applicable*, and they are not an
# accident of collection — they mean *this app judged the recorded value
# impossible and removed it*, which is a fact the record holds and the question
# never saw. Asking it anyway is the app forgetting what it did.
#
# **The fix is provenance, not a fourth mechanism answer.** A blank the app made
# is not a new kind of missingness; it is a blank whose cause is already
# written down. So this reads the decisions and says how many, and which, and
# quotes the decision's own sentence for why.

#: The payload flag a decision sets when it turns values into blanks. Keyed on
#: rather than on the decision KIND, so a second kind that also creates blanks
#: joins this reader instead of needing its own branch here.
MADE_BLANKS = "made_blanks"

#: The per-column blocks beside the flag: `[{column, n, rows, rows_known}, …]`.
#: A LIST rather than the single `column`/`rows` pair L48 shipped, because the
#: writers enumerated at L49 are not one-column-at-a-time — `apply_bulk`
#: installs nine frames under one decision and `melt_repeated` blanks a column
#: that did not exist a moment earlier.
BLANKS_MADE = "blanks_made"

#: Columns where a pass MAY have blanked cells and the app cannot say whether
#: it did. Recorded rather than omitted: an omission and a refusal must not look
#: the same, and this one is load-bearing — it is what stops `n_blank_in_the_file`
#: from being asserted over a frame whose rows were reshaped underneath it.
BLANKS_UNATTRIBUTABLE = "blanks_unattributable"

#: Why a pass could not attribute. Both are properties of the frame swap, not of
#: the pass, so they are detected rather than declared per writer.
ROWS_RESHAPED = "the pass rebuilt the row index, so no cell in this column can "\
                "be compared with the cell that preceded it"
ROWS_NOT_UNIQUE = "the row labels are not unique on both sides of the pass, so "\
                  "a cell cannot be matched to the one it replaced"


def blanks_made(before: pd.DataFrame, after: pd.DataFrame, *,
                pass_name: str) -> Dict[str, Any]:
    """**THE ONE RECORDER.** What a frame swap did to the blanks, as a payload.

    Every pass that installs a working table calls this and merges the result
    into its own decision's payload; nothing else composes a provenance record.
    `GUIDED-191` is the class where a writer changes the table and files no
    receipt, and one recorder is the only shape that closes it — a second copy
    per writer is the same defect with more places to forget it.

    **This observes; it does not guess.** The distinction matters because
    `blanks_the_app_made` refuses to read the table at all. That refusal is
    about the READER: reconstructing months later which of a column's blanks
    came from where is an invention. Comparing the frame a pass was handed with
    the frame it produced, at the instant it produced them, is the same
    measurement `set_impossible_missing` has always made against its own gate
    output — it is how the record gets written, not a substitute for having one.

    Returns **`{}`** when the pass blanked nothing and could see that it blanked
    nothing. Three states, never two:

    * **attributed** — the cells are named. Rows are labels.
    * **counted, rows unknown** — a column the pass CREATED. Every blank in it
      is one this pass wrote whatever happened to the index, so the count is
      exact and the labels are not available.
    * **unattributable** — a column that existed on both sides while the row
      index was rebuilt. `melt_repeated`, `promote_header`, `drop_empty_rows`,
      `drop_rows` and aggregation all do this. A net difference of blank counts
      would be a number here, and it would be the wrong number the moment a
      pass both fills and blanks; trap #9 says return nothing instead.
    """
    made: List[Dict[str, Any]] = []
    unattributable: List[Dict[str, Any]] = []

    before_cols = {str(c) for c in before.columns}
    # Duplicated labels make `.loc` return a frame, so the comparison below
    # would be comparing something other than a cell.
    unique = (not before.index.has_duplicates
              and not after.index.has_duplicates)
    identity = bool(unique and after.index.equals(before.index))
    kept = None                       # the surviving labels, where rows dropped
    if unique and not identity:
        # A PASS THAT ONLY DROPPED ROWS IS STILL ATTRIBUTABLE, and treating it
        # as opaque would have made an eligibility criterion — the single most
        # ordinary thing this app does — erase the provenance of every column
        # in the table. The surviving labels are a subset of the old ones and
        # they still name the same rows.
        #
        # A subset of labels is NOT enough on its own to conclude that, because
        # `.reset_index(drop=True)` also produces a subset of a `RangeIndex`
        # while every label now names a different row. So it is confirmed by
        # content, the same way `engine.preview_fix` confirms
        # `row_identity_preserved`: if the surviving rows are byte-for-byte the
        # rows that carried those labels, no renumbering happened. A pass that
        # dropped rows AND edited cells fails this and lands as unattributable,
        # which is the right answer — the two are indistinguishable from here.
        try:
            if len(after) < len(before) and after.index.isin(before.index).all():
                shared = [c for c in after.columns if c in set(before.columns)]
                aligned = before.loc[after.index, shared]
                same = aligned.astype(str).eq(after[shared].astype(str))
                both_na = aligned.isna() & after[shared].isna()
                if bool((same | both_na).to_numpy().all()):
                    kept = after.index
        except (KeyError, ValueError, IndexError, TypeError):
            kept = None
    why_not = (ROWS_NOT_UNIQUE if not unique else ROWS_RESHAPED)

    for col in after.columns:
        name = str(col)
        series = after[col]
        if isinstance(series, pd.DataFrame):          # duplicated column label
            continue
        if name not in before_cols:
            # A COLUMN THAT DID NOT EXIST. Every blank in it was written here,
            # by definition — there was no earlier value for one to have come
            # from. `add_feature`'s `log`/`sqrt`/`ratio` and `melt_repeated`'s
            # value column are both this case, and the count is exact even
            # where the index was rebuilt.
            n = int(series.isna().sum())
            if n:
                # THE LABELS COME OFF THE FRAME THE USER NOW HAS, so they are
                # nameable whenever that frame's labels are unique — a reshape
                # that rebuilt the index does not make them unnameable, it just
                # means they are new names for new rows, which is what the
                # table downstream is keyed by anyway. What genuinely defeats
                # this is a duplicated label, where a name picks out more than
                # one row and so names none of them.
                nameable = not after.index.has_duplicates
                made.append({
                    "column": name, "n": n, "pass": str(pass_name),
                    "rows": ([_row_label(i) for i in series.index[series.isna()]]
                             if nameable else []),
                    "rows_known": nameable,
                    "new_column": True})
            continue
        prior = before[col]
        if isinstance(prior, pd.DataFrame):
            continue
        if not identity and kept is None:
            # ONLY WHERE THERE IS SOMETHING TO BE UNCERTAIN ABOUT. A column
            # holding no blanks after the pass raises no provenance question,
            # so recording a refusal over it would put a *"cannot say"* on
            # every column of every reshaped table — the same noise a
            # zero-filled block would be, wearing a more alarming word.
            if int(series.isna().sum()):
                unattributable.append({"column": name, "pass": str(pass_name),
                                       "because": why_not})
            continue
        prior = prior if identity else prior.loc[kept]
        turned = prior.notna() & series.isna()
        n = int(turned.sum())
        if n:
            made.append({"column": name, "n": n, "pass": str(pass_name),
                         "rows": [_row_label(i) for i in series.index[turned]],
                         "rows_known": True, "new_column": False})

    frag: Dict[str, Any] = {}
    if made:
        frag[MADE_BLANKS] = True
        frag[BLANKS_MADE] = made
    if unattributable:
        frag[BLANKS_UNATTRIBUTABLE] = unattributable
    return frag


def _row_label(value: Any) -> Any:
    """Coerce one index label to something JSON can carry.

    Labels are scalars — ints, strings, occasionally timestamps. Anything with
    an `item()` is a numpy scalar hiding as a Python one.

    Lives here rather than in `project.py` because the row list is composed
    here now, and `project._label` delegates to it. Two coercions would
    eventually disagree about a timestamp label, and the row lists they build
    are compared to each other by every consumer of `provenance`.
    """
    if isinstance(value, (str, bool)) or value is None:
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return str(value)
    if isinstance(value, (int, float)):
        return value
    return str(value)


def blanks_the_app_made(decisions: Sequence[Any]) -> Dict[str, Dict[str, Any]]:
    """Per column: which blanks this app wrote, and the decision that wrote them.

    Takes the project's decision list — objects with `kind`, `subject`,
    `payload` and `text`, or the dicts they serialize to. Reads nothing from
    the table: this is the RECORD's answer, and a table-derived guess about
    which blank came from where is exactly the invention it exists to replace.

    Reads only what `blanks_made` writes. There is no second accepted payload
    shape — a legacy branch beside it would be the two mechanisms this loop
    exists to collapse into one, and it would go stale silently because the
    survey would still render.
    """
    out: Dict[str, Dict[str, Any]] = {}

    def _get(decision, field, default=None):
        return (decision.get(field, default) if isinstance(decision, dict)
                else getattr(decision, field, default))

    def _entry(column: str) -> Dict[str, Any]:
        return out.setdefault(str(column), {
            "column": str(column), "n": 0, "rows": [], "rows_known": True,
            "by": [], "unattributable": []})

    for decision in decisions or ():
        payload = _get(decision, "payload") or {}
        text = _get(decision, "text", "") or ""
        kind = _get(decision, "kind", "") or ""
        for block in (payload.get(BLANKS_UNATTRIBUTABLE) or []):
            entry = _entry(block.get("column", ""))
            entry["unattributable"].append(
                {"kind": kind, "pass": block.get("pass", ""),
                 "because": block.get("because", ""), "sentence": text})
        if not payload.get(MADE_BLANKS):
            continue
        for block in (payload.get(BLANKS_MADE) or []):
            entry = _entry(block.get("column", ""))
            n = int(block.get("n") or 0)
            entry["n"] += n
            entry["rows"].extend(block.get("rows") or [])
            if not block.get("rows_known", True):
                entry["rows_known"] = False
            # The decision's OWN sentence, quoted rather than paraphrased. One
            # composer, and this is not it — a second sentence about the same
            # event is the drift `GUIDED-098` cost a loop. `pass` is beside it
            # because a decision `kind` of `apply` covers nine operations and
            # WHICH ONE blanked the cell is the thing the survey has to say.
            entry["by"].append({"kind": kind, "pass": block.get("pass", ""),
                                "sentence": text, "n": n})
    return out


#: What the survey row says when some of a column's blanks are the app's own.
#: It states the split and then STOPS — it does not answer §07's question on
#: the user's behalf, which is `GUIDED-091`'s rule and the reason this is a
#: provenance note rather than a fourth mechanism.
#
#: **THE REMAINDER IS NOW NAMED, AND THE RENAME IS THE FIX.** L48 shipped this
#: saying *"the other 6 are not recorded as made here"*, and said in its own
#: comment why: *"would be false the moment any other path creates a blank
#: without recording that it did — `coerce_numeric` turns unparseable text into
#: `NaN` and files no `made_blanks`, so it is exactly that path."* That is a
#: hedge with a named cause, which makes it a to-do rather than a limit.
#: `coerce_numeric` now files, through `blanks_made`, along with every other
#: pass that installs a working table — so the stronger sentence is the true
#: one and the weaker one would now be the app under-claiming what it knows.
#: The hedge is kept exactly where it is still earned: `PROVENANCE_OPAQUE`.
PROVENANCE_MIXED = (
    "{n_created:,} of the {n_missing:,} blanks in `{column}` are ones this app "
    "wrote — {why} The other {n_original:,} came with the file. The question "
    "below is about both, and they did not get there the same way.")

#: Where the record accounts for EVERY blank in the column, the stronger
#: sentence is available and is used: nothing is left over to be wrong about.
PROVENANCE_ALL = (
    "Every one of the {n_missing:,} blanks in `{column}` is one this app "
    "wrote — {why} None of them came with the file.")

#: The record and the table disagree, and the app says so instead of doing
#: arithmetic that would come out negative. Trap #9 at the sentence layer.
PROVENANCE_DISAGREES = (
    "The record says this app set {n_created:,} entries of `{column}` to "
    "missing and the column now holds {n_missing:,} blanks, so the two cannot "
    "be reconciled here — something filled blanks after they were made. The "
    "count this app is responsible for is the recorded one; how many of "
    "today's blanks are those is not something this can answer.")

#: **WHERE THE HEDGE IS STILL EARNED.** A pass that rebuilds the row index —
#: reshaping wide to long, promoting a header, combining a person's rows —
#: leaves no cell that can be compared with the cell it replaced, so how many
#: of this column's blanks it made is not answerable. The count is withheld
#: rather than estimated, and the sentence says which pass took the answer
#: away. This is the branch `n_blank_in_the_file: None` belongs to.
PROVENANCE_OPAQUE = (
    "`{column}` holds {n_missing:,} blanks and how many of them this app made "
    "is not something the record can say: {why} So the question below is "
    "about blanks of at least two kinds, and this app cannot tell you the "
    "split rather than guess at it.")

#: Both at once: some blanks are attributed and a later pass then reshaped the
#: rows, so the attributed count stands and the remainder cannot be named.
PROVENANCE_PARTLY_OPAQUE = (
    "At least {n_created:,} of the {n_missing:,} blanks in `{column}` are ones "
    "this app wrote — {why} How many of the rest came with the file is not "
    "something the record can say, because {because}")


def provenance(column: str, n_missing: int,
               made: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """How many of `column`'s blanks this app made, and the sentence for it.

    `None` — not a zero-filled block — where the app made none and could see
    that it made none. A labeled region saying *"0 of these blanks are ours"*
    on every column in the table is noise on the 99% of columns no pass ever
    touched, and the survey row's own `n_missing` already says the rest.

    **`n_blank_in_the_file` is the field this loop renamed**, and the rename is
    the finding. It was `n_not_recorded_as_made_here` because exactly one
    writer filed provenance, so the remainder could only be described by what
    the record had failed to see. Every writer files now, so the remainder is
    the file's own blanks and the field says so.

    It is **`None`**, never a number, in the two cases where that claim is not
    the record's to make:

    * the record and the table cannot be reconciled — something filled blanks
      after a pass made them, and `n_missing - n_created` is negative. A
      clamped zero would assert the app made every blank in the column.
    * a pass rebuilt the row index over this column, so what it did to the
      blanks is unattributable. This is the honest heir to the old field name:
      the hedge did not disappear, it moved to the case that earns it.
    """
    made = made or {}
    n_created = int(made.get("n") or 0)
    opaque = list(made.get("unattributable") or [])
    if not n_created and not opaque:
        return None
    n_missing = int(n_missing)
    # `the \`coerce_numeric\` pass: <the decision's own sentence>`. A colon
    # rather than a dash because the surrounding sentence already spends one,
    # and the quoted half has to survive verbatim — the note quotes the
    # decision rather than paraphrasing it, which is the rule `GUIDED-098`
    # cost a loop to learn.
    why = " ".join(f"the `{b['pass']}` pass: {b['sentence']}".strip()
                   if b.get("pass") else b.get("sentence", "")
                   for b in made.get("by") or () if b.get("sentence")).strip()
    because = " ".join(
        f"{o.get('because', '')} (`{o.get('pass', '')}`)."
        for o in opaque).strip()

    if opaque and not n_created:
        sentence = PROVENANCE_OPAQUE.format(
            column=column, n_missing=n_missing, why=because)
        n_in_file = None
    elif opaque:
        sentence = PROVENANCE_PARTLY_OPAQUE.format(
            column=column, n_created=n_created, n_missing=n_missing, why=why,
            because=because)
        n_in_file = None
    elif n_created > n_missing:
        sentence = PROVENANCE_DISAGREES.format(
            column=column, n_created=n_created, n_missing=n_missing)
        n_in_file = None
    elif n_created == n_missing:
        sentence = PROVENANCE_ALL.format(
            column=column, n_missing=n_missing, why=why)
        n_in_file = 0
    else:
        n_in_file = n_missing - n_created
        sentence = PROVENANCE_MIXED.format(
            column=column, n_created=n_created, n_missing=n_missing,
            n_original=n_in_file, why=why)
    return {
        "column": str(column),
        "n_created_by_the_app": n_created,
        # The rename. See the docstring: `None` is the refusal, and it is the
        # only thing this field is allowed to say when it cannot say a number.
        "n_blank_in_the_file": n_in_file,
        "rows": list(made.get("rows") or []),
        # A pass that created a column while rebuilding the index knows HOW
        # MANY it blanked and not WHICH — so `rows` is short and must not be
        # read as the whole set. Said in the payload rather than left for a
        # consumer to infer from a length that happens to match.
        "rows_known": bool(made.get("rows_known", True)),
        "by": [dict(b) for b in made.get("by") or []],
        "unattributable": [dict(o) for o in opaque],
        "sentence": sentence,
    }


def survey(df: pd.DataFrame, target: Optional[str] = None) -> List[Dict[str, Any]]:
    """Which columns have missing values, and which branch each one is on.

    Reports; decides nothing. The dtype half of clause §07 is mechanical and
    lives here; the mechanism half cannot be — it is the user's answer, and
    inferring it is the same error the grain question exists to prevent.
    """
    out: List[Dict[str, Any]] = []
    for col in df.columns:
        if target is not None and str(col) == str(target):
            continue
        s = df[col]
        if isinstance(s, pd.DataFrame):        # duplicated label
            continue
        n_missing = int(s.isna().sum())
        if not n_missing:
            continue
        numeric = bool(pd.api.types.is_numeric_dtype(s))
        out.append({
            "column": str(col),
            "branch": "numeric" if numeric else "categorical",
            "n_missing": n_missing,
            "n_rows": int(len(s)),
            "fraction": round(n_missing / max(1, len(s)), 4),
            # Asked, never inferred — so this is None until the user answers.
            "mechanism": None,
            "strategies": list(NUMERIC_STRATEGIES if numeric
                               else CATEGORICAL_STRATEGIES),
        })
    out.sort(key=lambda d: (-d["fraction"], d["column"]))
    return out


CATEGORICAL_STRATEGIES = (EXPLICIT_CATEGORY, INDICATOR, INDICATOR_AND_IMPUTE,
                          IMPUTE_MODE, LEAVE)
NUMERIC_STRATEGIES = (INDICATOR, INDICATOR_AND_IMPUTE, IMPUTE_MEDIAN,
                      IMPUTE_MEAN, IMPUTE_MICE, LEAVE)
# One table, so the offer and the check read the same thing. Two lists that
# happen to agree are two lists.
STRATEGIES_BY_BRANCH = {"numeric": NUMERIC_STRATEGIES,
                        "categorical": CATEGORICAL_STRATEGIES}
# Every strategy this module can declare, from the two branches rather than from
# a third list — a third list is the one that goes stale.
STRATEGIES_ALL = frozenset(CATEGORICAL_STRATEGIES) | frozenset(NUMERIC_STRATEGIES)

_LABELS = {
    EXPLICIT_CATEGORY: "Keep blanks as an explicit `Missing` category",
    INDICATOR: "Add a was-it-missing column and leave the value blank",
    INDICATOR_AND_IMPUTE: "Add a was-it-missing column and fill the value",
    IMPUTE_MODE: "Fill with the most common value",
    IMPUTE_MEDIAN: "Fill with the median",
    IMPUTE_MEAN: "Fill with the mean",
    IMPUTE_MICE: "Fill by modeling it from the other columns (MICE)",
    LEAVE: "Leave it alone for now",
}

_FILLERS = {
    IMPUTE_MODE: "the column's most common value",
    IMPUTE_MEDIAN: "the column's median",
    IMPUTE_MEAN: "the column's mean",
    IMPUTE_MICE: "a value modeled from the other columns",
}

# Why each strategy is where it is on clause §06's litmus. Held as data so the
# interface can show the reasoning rather than assert the classification, the
# same way the transform catalogue does.
_BECAUSE = {
    EXPLICIT_CATEGORY:
        "Row-local: a blank becomes a literal `Missing` token using nothing but "
        "that row's own cell, so it can execute now.",
    INDICATOR:
        "Row-local: the new column is 1 where this row's value is blank and 0 "
        "where it is not. Nothing about any other row is consulted.",
    INDICATOR_AND_IMPUTE:
        "Both halves of clause §06 at once. The was-it-missing column is "
        "row-local and is added now; the value under it is a fact about the "
        "whole column and is fitted inside each training fold. The fact of the "
        "absence is kept AND the model gets a number to work with.",
    IMPUTE_MODE:
        "Stateful: the most common value is a fact about the whole column, so "
        "computing it over the full table would compute it over the held-out "
        "rows too.",
    IMPUTE_MEDIAN:
        "Stateful: the median is a fact about the whole column. Fitted inside "
        "each training fold, never over the sealed rows.",
    IMPUTE_MEAN:
        "Stateful: the mean is a fact about the whole column, and a more "
        "fragile one than the median — one extreme value moves it.",
    IMPUTE_MICE:
        "Stateful, and the most so: MICE fits a model per column against the "
        "others, so it learns the joint distribution of the training rows.",
    LEAVE:
        "Nothing is computed and nothing is deferred. Recorded so that "
        "'decided to leave it' and 'never looked at it' are different states.",
}


def strategy(key: str) -> Dict[str, Any]:
    if key not in _LABELS:
        raise MissingnessRefusal(
            f"'{key}' is not a missingness strategy. Known: "
            f"{', '.join(sorted(_LABELS))}.")
    return {"key": key, "label": _LABELS[key], "because": _BECAUSE[key],
            "defers": key not in ROW_LOCAL_STRATEGIES,
            # A compound strategy does both, and saying so is the whole point
            # of the third timing (`DRIVE-008`): understating what already
            # happened to the table and overstating it are both wrong.
            "executes_now": bool(key in (ROW_LOCAL_STRATEGIES - {LEAVE})
                                 or key in MIXED_STRATEGIES)}


def blocks(mechanism: Optional[str], strategy_key: str) -> bool:
    """Is this pairing the CONSEQUENCE clause §07 names?

    Only the informative branch blocks, and only for a strategy that destroys
    the signal. `NOT_SURE` deliberately does not block: the user has said they
    do not know, and turning an admission of uncertainty into a wall teaches
    people to stop admitting it.
    """
    return mechanism == INFORMATIVE and strategy_key in _FILLERS


# ── an option the constitution can refuse is not an equal peer ───────────────
#
# `GUIDED-163`. `blocks('informative', 'impute_median')` has always returned
# `True` and the 409 has always been correct on the wire. What was wrong is
# what a person READ: on the NHANES drive `meds_hbp` is observed
# `{True: 5527, False: 770}` with 15,552 blanks, the median is 1, and *"fill
# with the median"* was offered third on a flat list of peers — so the route
# that assigns every person of unknown medication status to being ON blood
# pressure medication sat beside the routes that keep the signal, with the same
# weight, under a heading that reads as a list of things you may do.
#
# **The shelf is never shortened.** The option is not removed and is not made
# unclickable: the user may know something the app does not, and §09's
# resolve-or-attest exit is how they say so. What changes is that the app
# **orders** it and **states its own concern beside it** — the two moves
# `PRODUCT_VISION.md` names as the alternatives to deletion.
#
# The three functions below are the only place that decision is made, and each
# one is DERIVED from `blocks` rather than restating it. A second hand-written
# list of which fills are refused is a second thing to drift, and this module
# has already paid for that twice (`GUIDED-090`, `GUIDED-098`).

#: Stated when the mechanism is ANSWERED informative. Definite, and it names
#: the exit rather than reading as a wall: the blocker resolves or is attested.
BLOCKED_CONCERN = (
    "You said a blank in `{column}` means something, so clause §07 refuses "
    "this one: filling those {n_missing:,} blanks with {filler} would take the "
    "fact that they were blank out of the data, and no model can recover it "
    "afterward. It is offered after the choices that keep the signal, and "
    "taking it needs a typed acknowledgment that the loss was deliberate.")

#: Stated when the mechanism is NOT YET ANSWERED. The app cannot say this is
#: refused, because it does not know — so it says what would refuse it, which
#: is the recorded-absence rule rather than either a warning or a silence.
BLOCKABLE_CONCERN = (
    "This one is refused if you answer that a blank in `{column}` means "
    "something: filling those {n_missing:,} blanks with {filler} would take "
    "the fact that they were blank out of the data, and no model can recover "
    "it afterward. That is why it is offered after the choices that keep the "
    "signal.")


def blocked_under(strategy_key: str) -> tuple:
    """Every mechanism answer under which clause §07 refuses this strategy.

    Asked of `blocks` rather than written out, so there is exactly one
    statement of which pairings are refused and this is a reading of it. An
    empty tuple means no answer to the mechanism question refuses this option —
    which is a stronger claim than "not refused right now" and is the one the
    shelf order is built on.
    """
    return tuple(m for m in MECHANISMS if blocks(m, strategy_key))


def shelf_rank(strategy_key: str) -> int:
    """Where this option sits: `0` above, `1` below.

    `1` for any option there is an answer to the mechanism question under
    which the constitution refuses it; `0` for the rest.

    **Keyed on the STRATEGY and not on the answer, deliberately.** An order
    that re-sorted itself when the user answered would move a control out from
    under the cursor, and — worse — would make *"the app put this last"* mean
    something different on two readings of the same card. The claim the order
    makes is the durable one: this fill CAN be refused here and those cannot,
    so it is not their peer. Which of them is refused *right now* is `blocked`,
    and what to say about it is `concern`.
    """
    return 1 if blocked_under(strategy_key) else 0


def concern(column: str, strategy_key: str, mechanism: Optional[str],
            n_missing: int) -> Optional[str]:
    """The server's own sentence about this option, or **nothing**.

    `None` in three cases and each is deliberate silence rather than an
    omission: an option no answer refuses, an answer of `not_informative`
    (the user has said the blanks are accidents, and a second uncalibrated
    layer of caution over a settled answer makes a real concern and a
    reflexive one read the same), and `not_sure` — which `blocks` already
    declines to block on, because turning an admission of uncertainty into a
    wall teaches people to stop admitting it.
    """
    if not blocked_under(strategy_key):
        return None
    template = (BLOCKED_CONCERN if blocks(mechanism, strategy_key)
                else BLOCKABLE_CONCERN if mechanism not in MECHANISMS
                else None)
    if template is None:
        return None
    return template.format(column=column, n_missing=int(n_missing),
                           filler=_FILLERS[strategy_key])


def reading(column: str, strategy_key: str, mechanism: Optional[str],
            n_missing: int) -> Dict[str, Any]:
    """What one offered option is, constitutionally — the whole of it.

    Trap #7 is the reason this is one object rather than a sentence plus a
    boolean: the machine-readable form beside a true sentence has twice been
    the lossier of the two, and the structured payload is what everything
    downstream reads.

    `blocked` is **three-valued on purpose**. `None` is *the mechanism question
    has not been answered, so this cannot be answered yet* — not `False`.
    Returning `False` there would assert the constitution permits a fill it may
    well refuse, which is trap #9 at the field level: return nothing rather
    than a wrong value.
    """
    under = blocked_under(strategy_key)
    return {
        "blocked": (blocks(mechanism, strategy_key)
                    if mechanism in MECHANISMS else None),
        "blocked_under": list(under),
        "concern": concern(column, strategy_key, mechanism, n_missing),
        "shelf_rank": shelf_rank(strategy_key),
    }


def blocker(column: str, mechanism: Optional[str], strategy_key: str,
            n_missing: int, branch: str = "categorical") -> Optional[Dict[str, Any]]:
    """The interruption, with both terminal exits attached.

    `DESIGN_LANGUAGE.md` §09: a CONSEQUENCE resolves or is attested, never a
    dead end. Both exits travel with the refusal so an interface cannot render
    the interruption without also rendering its way out.
    """
    if not blocks(mechanism, strategy_key):
        return None
    return {
        "column": str(column),
        "kind": "blocker",
        "strategy": strategy_key,
        "message": INFORMATIVE_IMPUTATION_BLOCKER.format(
            column=column, strategy=_LABELS[strategy_key].lower(),
            n_missing=int(n_missing), filler=_FILLERS[strategy_key]),
        "exits": [dict(e) for e in blocker_exits(branch)],
        "acknowledgment_kind": "typed",
    }


def declare(column: str, branch: str, mechanism: str, strategy_key: str,
            target: Optional[str] = None,
            uses_columns: Optional[Sequence[str]] = None,
            acknowledged: bool = False,
            n_missing: int = 0,
            purpose: Optional[str] = None,
            scope: str = TRAIN_ROWS) -> Dict[str, Any]:
    """Record how one column's missingness will be handled. Executes nothing.

    `scope` is where a deferred statistic gets fitted, and it defaults to
    `TRAIN_ROWS` because that is what this door does (`AUDIT-028`).

    Refuses three things, each for a different reason:

    * an unknown mechanism or strategy — a typo would otherwise become a silent
      default;
    * the outcome inside a MICE scope — clause §07's hard blocker, and the one
      that is not a judgment call, so it is refused rather than offered;
    * an informative-missingness imputation with no acknowledgment — the
      CONSEQUENCE, which resolves or is attested.
    """
    if mechanism not in MECHANISMS:
        raise MissingnessRefusal(
            f"{mechanism!r} is not one of {list(MECHANISMS)}. The mechanism is "
            f"asked, never inferred — `not_sure` is a real answer.")
    # THE STRATEGY HAS TO BELONG TO THE BRANCH, and this was not checked.
    # `explicit_category` on a numeric column was accepted, wrote the literal
    # string `Missing` into it, and turned a column of numbers into a column of
    # text — silently, while the recorded sentence said only that the blanks
    # were kept as their own category. Everything downstream then read the
    # column differently: the profile, the numeric candidate lists, the recipe
    # lattice. A strategy list per branch that nothing enforces is a comment.
    allowed = STRATEGIES_BY_BRANCH.get(branch)
    if allowed is not None and strategy_key not in allowed:
        raise MissingnessRefusal(
            f"{strategy_key!r} is not offered for a {branch} column. "
            f"{_LABELS.get(strategy_key, strategy_key)} would change what the "
            f"column IS — a {branch} column filled this way stops being one, "
            f"and nothing downstream would say so. Available here: "
            f"{', '.join(allowed)}.")
    spec = strategy(strategy_key)

    fit_scope = _checked_scope(scope)
    uses = [str(c) for c in (uses_columns or [])]
    outcome_note = None
    if strategy_key == IMPUTE_MICE and target and str(target) in uses:
        # `AUDIT-005`. The recorded purpose decides which of the two true
        # sentences applies. Under prediction it is still a hard blocker; under
        # inference the configuration is the correct one and the note travels
        # with the record so the methods section can say why.
        reading = outcome_in_scope(target, purpose)
        if reading["refuse"]:
            raise MissingnessRefusal(reading["message"])
        outcome_note = reading

    if blocks(mechanism, strategy_key) and not acknowledged:
        raise MissingnessRefusal(
            blocker(column, mechanism, strategy_key, n_missing)["message"])

    record: Dict[str, Any] = {
        "column": str(column),
        "branch": branch,
        "mechanism": mechanism,
        "strategy": strategy_key,
        "label": spec["label"],
        "because": spec["because"],
        "defers": spec["defers"],
        # `AUDIT-028`. `fit_on` said "training folds only" for every deferring
        # strategy over a door with no folds, and `fit_scope` is the same fact
        # in the vocabulary `selection.py` already records it in — so a parity
        # check can compare the two doors instead of parsing prose.
        "fit_scope": fit_scope if spec["defers"] else None,
        "fit_on": (_SCOPE_FIT_ON[fit_scope] if spec["defers"]
                   else "row-local, applied now"),
        "outcome_in_scope": outcome_note,
        "uses_columns": uses or None,
        "acknowledged_signal_loss": bool(blocks(mechanism, strategy_key) and acknowledged),
        "sentence": sentence_for(column, branch, strategy_key, scope=fit_scope),
    }
    if mechanism == INFORMATIVE:
        # §07: recorded as a methods ASSUMPTION rather than a warning, because a
        # warning is something a user dismisses and an assumption is something a
        # manuscript carries.
        record["assumption"] = STABILITY_ASSUMPTION.format(column=column)
    return record


#: What a compound strategy fills with, per branch. Named here so the card and
#: the record cannot describe the same click differently.
_MIXED_FILL = {"numeric": "the median", "categorical": "the most common value"}


def sentence_for(column: str, branch: str, key: str,
                 scope: str = TRAIN_ROWS) -> str:
    """The methods-prose line for one (column, branch, strategy, scope).

    `scope` says over which rows a DEFERRED statistic is fitted, and it changes
    nothing for a row-local strategy — `leave`, `explicit_category` and
    `indicator` compute no statistic, so there is no scope for one to have
    (`AUDIT-028`). It defaults to `TRAIN_ROWS` because a caller who does not say
    must understate rather than overstate; see the note beside `TRAIN_ROWS`.

    **One composer, read by both doors** (`GUIDED-098`). The Explore card used
    to write its own `decision_sentence` per option and the record wrote
    another when the option was taken, and on two options they said opposite
    things: the card promised a training-fold median and the record said the
    value was left blank. Two strings that happen to agree are two strings, and
    these two did not even agree.

    So `ml/missingness_plan.py` asks this, and `declare` asks this, and there is
    nothing left to drift.
    """
    if key == LEAVE:
        return (f"Missing values in `{column}` are left as they are; no "
                f"imputation is applied and none is scheduled.")
    if key == EXPLICIT_CATEGORY:
        return (f"Missing values in `{column}` are kept as an explicit "
                f"`Missing` category rather than filled.")
    if key == INDICATOR:
        return (f"A was-it-missing indicator is added for `{column}`; the "
                f"underlying value is left blank.")
    if key == INDICATOR_AND_IMPUTE:
        # The compound one: the indicator is written now, the fill is deferred,
        # and only the fill has a scope. `_MIXED_FILL` names the statistic, so
        # the scope clause attaches to it rather than to the whole sentence.
        return (f"A was-it-missing indicator is added for `{column}`, and the "
                f"underlying value is filled with "
                f"{_MIXED_FILL.get(branch, 'the deferred value')} computed"
                f"{_SCOPE_PHRASE[_checked_scope(scope)]}.")
    where = (_SCOPE_PHRASE[_checked_scope(scope)]
             if strategy(key)["defers"] else "")
    return (f"Missing values in `{column}` will be filled using "
            f"{_LABELS[key].lower().replace('fill with ', '').replace('fill by ', '')}"
            f"{where}.")


def executes_now(strategy_key: str) -> bool:
    """Clause §06's litmus, as a question about one strategy.

    > **Does this transform's output for row *i* depend on any other row?**

    No for `explicit_category` and `indicator` — a blank becomes the level
    `Missing`, or a `1` in a was-it-missing column, and neither reading consults
    another row. Yes for every imputation, which is why they are declared and
    fired inside training folds.

    `leave` is row-local and executes nothing, which is not the same as being
    stateful: it is a recorded decision with no operation behind it. It is
    excluded here because *"applied now"* would be a receipt for work that did
    not happen.
    """
    return strategy_key in (EXPLICIT_CATEGORY, INDICATOR)


MISSING_LEVEL = "Missing"

# ─────────────────────────────────────────────────────────────────────────────
# The card's vocabulary and the record's, joined
#
# `DRIVE-008`. `ml/missingness_plan.py` builds the card the user reads and names
# its options `explicit_missing`, `impute_mode`, `indicator_and_impute`. This
# module names the DECLARATIONS `explicit_category`, `impute_mode`, `indicator`.
# Two vocabularies for one concept, and the panel bridged them by recording a
# free-text `note` — so pressing *"Record this"* wrote a sentence and routed
# nothing, which is why the panel showed and could not execute.
#
# The join lives here rather than in `ml/`, because the engine builds a card for
# both doors and must not learn the Guided door's record shape. A card option
# with no entry is refused rather than guessed at: a missing key would otherwise
# become a silent default, which is the failure `declare` already refuses for
# strategies.
# SINCE `GUIDED-090` THE CARD EMITS THE RECORD'S OWN KEYS, because one table
# decides what both doors offer. What remains here is the join for the card's
# OLD spellings, which a client written against the previous payload may still
# send — and `indicator_and_impute`, which is the entry that was wrong.
#
# It mapped to `INDICATOR`, whose sentence is *"the underlying value is left
# blank"*, while the card option it came from promised a training-fold median.
# One click, two methods sentences, opposite claims (`GUIDED-098`).
CARD_STRATEGY: Dict[str, str] = {
    "explicit_missing": EXPLICIT_CATEGORY,          # the card's old spelling
    "explicit_category": EXPLICIT_CATEGORY,
    "indicator": INDICATOR,
    "indicator_and_impute": INDICATOR_AND_IMPUTE,
    "impute_mode": IMPUTE_MODE,
    "impute_median": IMPUTE_MEDIAN,
    "impute_mean": IMPUTE_MEAN,
    "impute_iterative": IMPUTE_MICE,                # the card's old spelling
    "impute_mice": IMPUTE_MICE,
    "leave": LEAVE,
}

# The one card option that is NOT a missingness strategy, with the reason.
# Clause §04: dropping the rows with no value for a column changes who the study
# is about, which makes it an eligibility criterion reported in participant
# flow — not a way of handling a blank. Routing it through `declare` would file
# an exclusion as a preprocessing decision and lose it from the flow diagram.
NOT_A_STRATEGY: Dict[str, str] = {
    "drop_rows": (
        "Dropping every row with no value for this column is a complete-case "
        "analysis: it changes who the study is about, so it is an eligibility "
        "criterion reported in participant flow rather than a way of handling "
        "a blank. It is asked as one, before the seal."),
}


def card_option_for_strategy(strategy: str) -> Optional[str]:
    """The card spelling that records this declaration, or `None`.

    **The inverse of `CARD_STRATEGY`, and it exists because an exit's retry was
    being shadowed.** `api.py`'s `route_missingness` reads `card_option` in
    PREFERENCE to `strategy` — the join `DRIVE-008` added — so a retry payload
    carrying only `strategy` is silently overridden when it is merged into a
    request that came from the Explore door, which posts `card_option`. Driven
    at L48: the blocker's own resolve exit, merged the way `showRefusal` merges
    it, re-posted the refused strategy and produced a **second 409**. That is
    `GUIDED-072`'s defect — an exit that renders as a way through and opens
    nothing — alive inside the fix built for it.

    `None`, never a guess (trap #9). Several card spellings map to one
    declaration and two of them are the card's OLD spellings; a caller that got
    an old spelling back would be handed a key the current page does not emit.
    The canonical spelling is the one identical to the declaration, which is
    true of every strategy a resolve exit names today and is asserted in
    `test_a_resolve_exit_can_actually_be_taken.py` rather than assumed.
    """
    return strategy if CARD_STRATEGY.get(strategy) == strategy else None


def strategy_for_card_option(key: str) -> str:
    """The declaration a card option records, or a refusal saying why not."""
    if key in NOT_A_STRATEGY:
        raise MissingnessRefusal(NOT_A_STRATEGY[key])
    if key not in CARD_STRATEGY:
        raise MissingnessRefusal(
            f"{key!r} is not an option this record knows how to keep. A card "
            f"option with no declaration behind it would be recorded as a "
            f"sentence and executed as nothing.")
    return CARD_STRATEGY[key]


def indicator_column(column: str) -> str:
    """The name a was-it-missing indicator takes.

    Stated once and read three times — by the executor in
    `project.route_missingness`, by the collision check beside it, and by
    anything asking whether a column is one of these. The name was a literal in
    one place and a claim in another before `DRIVE-008`.
    """
    return f"{column}_was_missing"


def plan_receipt(declared: Sequence[Dict[str, Any]],
                 n_columns_with_missing: int) -> Dict[str, Any]:
    """What the user reads at the end of a step in which nothing visibly changed.

    **The hard part of this step.** Almost every preprocessing transform is
    stateful by clause §06's litmus, so the output is a set of recorded
    decisions that fire inside training folds — the user answers questions, the
    table does not change, and the step still has to be believable.

    The Features step's settle-sentence is the pattern: name the counts in each
    category, and name where the deferred ones will happen. What this adds is
    that the counts can legitimately be `0 applied now`, so the sentence has to
    be honest about a zero rather than embarrassed by it.
    """
    now = [d for d in declared if not d["defers"] and d["strategy"] != LEAVE]
    later = [d for d in declared if d["defers"]]
    left = [d for d in declared if d["strategy"] == LEAVE]
    # A COMPOUND STRATEGY IS IN BOTH COLUMNS AND THE RECEIPT SAYS SO
    # (`GUIDED-098`). `indicator_and_impute` puts the was-it-missing column on
    # the table now and defers the fill, so counting it only as deferred
    # understates what already happened — which is the same understatement
    # `DRIVE-008` fixed on the card's timing.
    mixed = [d for d in declared if d["strategy"] in MIXED_STRATEGIES]
    attested = [d for d in declared if d.get("acknowledged_signal_loss")]
    assumptions = [d["assumption"] for d in declared if d.get("assumption")]

    parts: List[str] = []
    if now:
        parts.append(f"{len(now)} column(s) changed now")
    if mixed:
        parts.append(f"{len(mixed)} given an indicator now and a fill in the "
                     f"folds")
    parts.append(f"{len(later)} recorded to be fitted inside the training folds")
    if left:
        parts.append(f"{len(left)} deliberately left alone")

    unanswered = max(0, n_columns_with_missing - len(declared))
    return {
        "n_applied_now": len(now),
        "n_deferred": len(later),
        "n_mixed": len(mixed),
        "n_left": len(left),
        "n_unanswered": unanswered,
        "assumptions": assumptions,
        "n_attested": len(attested),
        "headline": "Missingness settled: " + ", ".join(parts) + ".",
        # THE SENTENCE THAT MAKES A ZERO BELIEVABLE. Stated plainly rather than
        # hidden, because a step that says nothing after a user answered six
        # questions reads as a step that did nothing.
        "why_nothing_changed": (
            "Your table looks the same because it is the same. Filling a blank "
            "with a median means computing that median, and computing it over "
            "every row would compute it over the held-out rows too — so the "
            "decision is recorded now and the arithmetic happens inside each "
            "training fold, where it can only see training data. What you just "
            "did is the part that cannot be automated; what is left is "
            "bookkeeping the pipeline does on its own."
            if later else
            "Nothing was deferred, so nothing is waiting: every answer here "
            "either changed the table or deliberately left it alone."),
        "outstanding": (
            f"{unanswered} column(s) with missing values have not been "
            f"answered yet." if unanswered else ""),
    }


#: The categorical blocker's exits, bound after `card_option_for_strategy`.
#: See the note where this used to sit.
BLOCKER_EXITS = blocker_exits("categorical")

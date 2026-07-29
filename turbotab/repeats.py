"""turbotab.repeats — questions 4 to 7 of the opening sequence.

`OPENING_SEQUENCE.md` §03, questions 4–7; `STATE-108`, `STATE-109`, `STATE-110`.
Four questions, all of which fire only when the grain answer is *people repeat*,
and each of which gates the next:

| # | Question | Fires when | Kind |
|---|---|---|---|
| 4 | Are these repeats or different time points? | grain = repeat | usually **stated** |
| 5 | When you analyze this, what is one row? | grain = repeat | asked, **no default** |
| 6 | How should each person's rows be combined? | unit = person | asked |
| 7 | Are you predicting something later from earlier? | 4 = time points **and** 5 = row | asked |

## Why question 4 exists at all

Neither the grain nor the unit of analysis asks the thing that decides whether
averaging is correct: **what varies between one person's rows?** The same
structural fact — a person appearing three times — means two different things,
and only this question separates them:

* **replicate measurements of one quantity** — two dietary recalls, technical
  replicates. Averaging them *reduces measurement error*. It is not information
  loss; the variation being averaged away is noise that attenuates associations
  toward the null.
* **different time points** — clinical visits, a time course. Averaging
  **destroys the signal**, because the signal is the change.

## Why it is stated and not asked

It is largely inferable, and the evidence is spacing. A schedule is a schedule:
`clinical_longitudinal.csv`'s visits sit 80 to 100 days apart with a coefficient
of variation of 0.05. Two dietary recalls sit 3 to 14 days apart with a CV of
0.42, which is not a schedule — it is whenever the interviewer got through.

So the app **states its reading and cites the measurement**, rendered as a skip
the user can open (`DESIGN_LANGUAGE.md` §09), never as a decision taken on their
behalf. Where the evidence is thin it is asked rather than guessed, and "thin" is
a measured condition rather than a feeling: no date column and no replicate
index means there is nothing to read.

## Why aggregation cannot move

Decision A's identity barrier forces it. Combining three visits into one
person-row **changes what a row is**, and a seal drawn beforehand names rows that
no longer exist. This is the same rule that makes `melt_repeated` a pre-barrier
repair, and :meth:`AnalysisProject.set_aggregation` refuses after the seal for
exactly that reason rather than as a policy.

And **target precedes aggregation**, which is easy to miss: if the outcome is
measured at every visit, combining rows requires deciding *which* outcome.
`clinical_longitudinal.csv` is the concrete case — 127 of its 200 people change
`progressed` across their visits — and question 6 cannot be asked coherently
there without question 2 answered.

## What is filed rather than built

Slope, area under the curve and usual-intake modeling. All three are real
practice and materially more work; the v1 menu is **mean · first · last · change
from baseline**. Usual-intake modeling in particular is close enough to nutrition
practice that omitting it may make the app feel like a toy to that audience,
which is recorded in `OPENING_SEQUENCE.md` §05 rather than smoothed over here.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ── question 4 ───────────────────────────────────────────────────────────────

REPEATS = "repeats"
TIME_POINTS = "time_points"
REPEAT_KINDS = (REPEATS, TIME_POINTS)

# ── question 5 ───────────────────────────────────────────────────────────────

UNIT_PERSON = "person"
UNIT_RECORD = "record"
# The same escape hatch as the grain question's, and it belongs here for the same
# reason: matched pairs are not one row per person and not one row per record,
# they are one row per *set*, and the app has no aggregation that means that.
# Routes to the conservative treatment — rows survive, nothing is combined — and
# leaves the manuscript gap where the design would be described.
UNIT_NOT_DESCRIBED = "not_described"
UNITS = (UNIT_PERSON, UNIT_RECORD, UNIT_NOT_DESCRIBED)

# ── question 6 ───────────────────────────────────────────────────────────────

MEAN = "mean"
FIRST = "first"
LAST = "last"
CHANGE = "change_from_baseline"
AGGREGATIONS = (MEAN, FIRST, LAST, CHANGE)


class RepeatsError(Exception):
    """The chain was asked something its own answers do not support."""


# A gap this short is not a schedule, it is a convenience. Fourteen days is the
# shortest interval at which a clinical follow-up is normally BOOKED; below it,
# spacing is more likely to reflect when somebody was available.
_SCHEDULE_MIN_DAYS = 14.0
# Regularity, as a coefficient of variation of the within-person gaps. A
# schedule is regular by construction; the fixtures measure 0.05 against 0.42,
# so the threshold sits far from both rather than being tuned to either.
_SCHEDULE_MAX_CV = 0.35


def _date_columns(df: pd.DataFrame, min_parse: float = 0.9) -> List[str]:
    """Columns that read as dates. Parsed, never matched on the name.

    A third name list would be the mistake constitution §02 names, spelled
    differently — and `visit_date`, `date_of_visit`, `dov` and `RecallDate` are
    four spellings of one thing that no list gets right.
    """
    out: List[str] = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue
        if pd.api.types.is_datetime64_any_dtype(s):
            out.append(str(c))
            continue
        raw = s.dropna().astype(str)
        if len(raw) < 3:
            continue
        parsed = pd.to_datetime(raw, errors="coerce", format="ISO8601")
        if parsed.isna().all():
            # The fallback exists for the formats ISO8601 does not cover —
            # `3/14/2024`, `14-Mar-2024`. It is noisy by design: pandas warns
            # that it is parsing element by element, and it does that on every
            # non-date text column in the table, which is most of them. The
            # warning is correct and is not news, so it is silenced here rather
            # than printed once per column per render.
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed = pd.to_datetime(raw, errors="coerce")
        if float(parsed.notna().mean()) >= min_parse and parsed.nunique() >= 3:
            out.append(str(c))
    return out


def _replicate_index(df: pd.DataFrame, group_col: str) -> Optional[str]:
    """An integer column that runs 1..k within every person.

    Shape only: a value set of consecutive small integers starting at 1 (or 0),
    complete and non-repeating inside each group. `recall_number` and `visit`
    both match, which is the point — this evidence says there is an ORDER and
    says nothing about what the order means.
    """
    for c in df.columns:
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue
        values = s.dropna()
        if values.empty or values.nunique() > 20:
            continue
        try:
            if not np.all(np.equal(np.mod(values.to_numpy(dtype=float), 1), 0)):
                continue
        except (TypeError, ValueError):
            continue
        ok = True
        for _, block in df.groupby(group_col, dropna=True)[c]:
            got = sorted(int(v) for v in block.dropna())
            if len(got) != len(set(got)) or not got:
                ok = False
                break
            start = got[0]
            if start not in (0, 1) or got != list(range(start, start + len(got))):
                ok = False
                break
        if ok:
            return str(c)
    return None


def spacing(df: pd.DataFrame, group_col: str) -> Optional[Dict[str, Any]]:
    """Within-person gaps, in days, over the best date column available.

    "Best" is the one that yields a gap for the most people — a table with both
    an enrollment date (constant per person, so no gap) and a visit date should
    read the visit date, and choosing by name would get that wrong on the first
    table that spells it differently.
    """
    best: Optional[Dict[str, Any]] = None
    for col in _date_columns(df):
        parsed = pd.to_datetime(df[col], errors="coerce")
        gaps: List[float] = []
        people = 0
        for _, block in parsed.groupby(df[group_col], dropna=True):
            values = block.dropna().sort_values()
            if len(values) < 2:
                continue
            people += 1
            gaps.extend(np.diff(values.to_numpy()).astype("timedelta64[D]")
                        .astype(float).tolist())
        if not gaps:
            continue
        arr = np.array(gaps, dtype=float)
        mean = float(arr.mean())
        reading = {
            "column": col, "n_people": people, "n_gaps": len(arr),
            "min_days": float(arr.min()), "max_days": float(arr.max()),
            "median_days": float(np.median(arr)),
            "cv": float(arr.std() / mean) if mean else 0.0,
            "all_identical": bool(np.all(arr == 0)),
        }
        if best is None or reading["n_gaps"] > best["n_gaps"]:
            best = reading
    return best


def read(df: pd.DataFrame, group_col: Optional[str]) -> Dict[str, Any]:
    """Question 4's reading: what it believes, why, and how strongly.

    Returns `reading=None` when the evidence is thin, and thin is a MEASURED
    condition — no dates and no replicate index — rather than a judgment call.
    Where it is thin the question is asked, because *"where the evidence is thin,
    it is asked rather than guessed"* is the clause and guessing here decides
    whether averaging is correct.
    """
    out: Dict[str, Any] = {"reading": None, "stated": False, "confidence": None,
                           "evidence": [], "sentence": "", "spacing": None,
                           "replicate_index": None}
    if not group_col or group_col not in df.columns:
        return out

    gaps = spacing(df, group_col)
    out["spacing"] = gaps
    index_col = _replicate_index(df, group_col)
    out["replicate_index"] = index_col

    if gaps is not None:
        if gaps["all_identical"]:
            out.update(reading=REPEATS, stated=True, confidence="high")
            out["evidence"] = [
                f"every one of a person's records carries the same date in "
                f"`{gaps['column']}`"]
        elif (gaps["cv"] <= _SCHEDULE_MAX_CV
                and gaps["median_days"] >= _SCHEDULE_MIN_DAYS):
            out.update(reading=TIME_POINTS, stated=True, confidence="high")
            out["evidence"] = [
                f"a person's records in `{gaps['column']}` are "
                f"{gaps['median_days']:.0f} days apart at the median, ranging "
                f"{gaps['min_days']:.0f} to {gaps['max_days']:.0f} — regular "
                f"enough to be a schedule"]
        elif gaps["median_days"] < _SCHEDULE_MIN_DAYS:
            out.update(reading=REPEATS, stated=True, confidence="medium")
            out["evidence"] = [
                f"the gaps between one person's records in `{gaps['column']}` "
                f"run {gaps['min_days']:.0f} to {gaps['max_days']:.0f} days "
                f"(median {gaps['median_days']:.0f}), too close together and "
                f"too uneven to be a visit schedule"]
        else:
            # Widely spaced AND irregular. Unscheduled encounters are still
            # time points; this is the one date-bearing case where the reading
            # is genuinely uncertain, so it is asked.
            out["evidence"] = [
                f"a person's records in `{gaps['column']}` are "
                f"{gaps['median_days']:.0f} days apart at the median but vary "
                f"widely ({gaps['min_days']:.0f} to {gaps['max_days']:.0f}), "
                f"which fits unscheduled encounters as well as it fits repeats"]
    elif index_col:
        # No dates at all. A replicate index says there is an ORDER and not what
        # the order means, so this is stated at lower confidence and the skip is
        # the same one sentence away from being reopened.
        out.update(reading=REPEATS, stated=True, confidence="medium")
        out["evidence"] = [
            f"`{index_col}` numbers each person's records 1, 2, 3 and there is "
            f"no date column, so nothing here spaces them out in time"]

    if index_col and out["reading"] and gaps is not None:
        out["evidence"].append(
            f"`{index_col}` numbers each person's records in order")

    out["sentence"] = _sentence(out)
    return out


def _sentence(reading: Dict[str, Any]) -> str:
    """The rendered skip, or the question when the evidence is thin.

    A rendered skip is a **muted neutral row** with a mono provenance clause and
    a sans reopen affordance — never green, because green means a human recorded
    it (`DESIGN_LANGUAGE.md` §09).
    """
    evidence = "; ".join(reading["evidence"])
    if reading["reading"] == REPEATS:
        return (f"Not asked: these look like repeated measurements of the same "
                f"quantity rather than different time points — {evidence}.")
    if reading["reading"] == TIME_POINTS:
        return (f"Not asked: these look like different time points rather than "
                f"repeated measurements of the same quantity — {evidence}.")
    if evidence:
        return (f"Asked rather than stated: {evidence}. Which of the two it is "
                f"decides whether averaging a person's rows is correct, so it "
                f"is not something to guess at.")
    return ("Asked rather than stated: there is no date column and nothing "
            "numbering a person's records, so nothing here says what varies "
            "between them.")


REOPEN = "Ask me anyway"


# ── question 6 · the menu is domain-shaped ───────────────────────────────────

AUTHOR_REQUIRED = "[AUTHOR REQUIRED]"

# What the manuscript carries where the app was told its vocabulary does not fit.
# `[AUTHOR REQUIRED]` is `ml/narrative_engine.py`'s own marker, borrowed rather
# than reinvented so the export's existing gap-reporting finds it.
DESIGN_GAP = (
    AUTHOR_REQUIRED + " Describe the study design and how it constrains the "
    "held-out set. The analysis was told that none of the offered shapes — one "
    "row per participant, repeated measures of one participant, or unknown — "
    "describes this data, so the app applied the most conservative treatment it "
    "has and did not attempt a description. Matched sets, crossover periods and "
    "nested sampling all need a sentence the app cannot write.")

UNIT_GAP = (
    AUTHOR_REQUIRED + " State what one row of the analyzed table is, and why. "
    "The app was told that neither one row per participant nor one row per "
    "record describes this design, so the records were left as they are and "
    "nothing was combined — the most conservative treatment available. A "
    "matched set, a crossover period or a cluster is a unit of analysis the app "
    "has no aggregation for, and the sentence describing it is yours.")

_MENU: Dict[str, Dict[str, str]] = {
    MEAN: {
        "label": "Their mean",
        "sentence": "Each person's records were averaged into one row."},
    FIRST: {
        "label": "The first",
        "sentence": "Each person's first record was kept and the rest dropped."},
    LAST: {
        "label": "The last",
        "sentence": "Each person's last record was kept and the rest dropped."},
    CHANGE: {
        "label": "The change from the first",
        "sentence": ("Each person's change from their first record to their "
                     "last was used.")},
}


def menu(kind: str, lens: Sequence[str] = ()) -> Dict[str, Any]:
    """The aggregation options, and which one is recommended, and why.

    **The recommendation is the whole of the domain knowledge here**, and it
    inverts between the two readings:

    * **repeats** → the mean, `derived`, with the measurement-error reason
      stated. This is not a preference: a single measurement is a noisy
      estimate of the underlying quantity and that noise attenuates
      associations toward the null. Averaging reduces it.
    * **time points** → **no default at all**, and that absence is the finding.
      Averaging a trajectory destroys the trajectory, and which summary replaces
      it depends on the research question rather than on the data.

    **The dietary reason is READ FROM THE PACK, never restated here**
    (`GUIDED-026`). This function used to carry its own nearly identical
    sentences, and the pack's copy was the unreachable one — so editing the pack
    changed nothing, and editing this file made the pack's stated prior a false
    description of the app. Two implementations of one rule, with the
    documented one inert. The pack is the implementation; this reads it and
    reports which pack supplied it, so the record can name the source of a
    recommendation rather than asserting it in the app's own voice.
    """
    if kind not in REPEAT_KINDS:
        raise RepeatsError(f"{kind!r} is not one of {list(REPEAT_KINDS)}.")
    options = [{"key": k, **_MENU[k]} for k in AGGREGATIONS]
    if kind == TIME_POINTS:
        return {
            "options": options, "recommended": None, "marker": "offered",
            "reason": ("No default. These are different time points, and "
                       "averaging them destroys the signal — the change over "
                       "time IS the signal. Which summary replaces it comes "
                       "from your research question, not from the data."),
            "from_pack": None, "filed": _FILED}

    from turbotab import packs as _packs
    supplied = [p for p in _packs.priors(lens or [], "repeat_treatment")
                if p.get("treatment") == "mean"]
    if supplied:
        # The pack's own sentence, and the pack's own name beside it. A user
        # reading "averaging reduces measurement error" is entitled to know
        # which field's convention said so.
        return {"options": options, "recommended": MEAN,
                "marker": supplied[0]["marker"],
                "reason": supplied[0]["reason"],
                "from_pack": supplied[0]["pack"],
                "from_pack_label": supplied[0]["label"],
                "filed": _FILED}

    # No lens, or a lens with nothing to say about repeated measurements. The
    # general argument still holds — it is arithmetic about noise, not a domain
    # convention — so the mean is still recommended, without a pack's name on
    # it. This sentence is the GENERAL case and is not a copy of any pack's:
    # the dietary one is about 24-hour recalls and usual intake, and lives
    # there.
    return {"options": options, "recommended": MEAN, "marker": "derived",
            "reason": ("A single measurement is a noisy estimate of the "
                       "underlying quantity, and that noise attenuates "
                       "associations toward the null. Using the mean of a "
                       "person's records reduces it — this is "
                       "measurement-error reduction, not information loss."),
            "from_pack": None, "filed": _FILED}


# Named rather than silently absent, because a menu of four that does not say
# what is missing reads as a menu of everything.
_FILED = ("Slope, area under the curve and usual-intake modeling are real "
          "practice and are not built here. They are recorded as missing rather "
          "than left out quietly.")


def aggregate(df: pd.DataFrame, group_col: str, method: str,
              target: Optional[str] = None,
              order_col: Optional[str] = None) -> Dict[str, Any]:
    """Combine each person's rows into one. **Pre-seal, always.**

    This changes what a row IS, which is why it cannot happen after the lockbox
    names rows — the same reasoning that makes `melt_repeated` a pre-barrier
    repair. The refusal lives on the project, which owns the lockbox; this
    function does the work and reports what it did.

    Non-numeric columns take the person's first value, which is right for a
    fixed attribute (sex, site) and is a compromise for one that varies. The
    compromise is REPORTED rather than hidden: `varying_categoricals` names
    every non-numeric column that was not constant within a person, so the
    receipt can say which columns lost information.
    """
    if method not in AGGREGATIONS:
        raise RepeatsError(f"{method!r} is not one of {list(AGGREGATIONS)}.")
    if group_col not in df.columns:
        raise RepeatsError(f"No column named {group_col!r} in this table.")

    ordered = df
    if order_col and order_col in df.columns:
        key = df[order_col]
        if not pd.api.types.is_numeric_dtype(key):
            key = pd.to_datetime(key, errors="coerce")
        ordered = df.assign(_order=key).sort_values(
            [group_col, "_order"], kind="stable").drop(columns=["_order"])

    numeric = [str(c) for c in df.columns
               if pd.api.types.is_numeric_dtype(df[c]) and str(c) != group_col]
    other = [str(c) for c in df.columns
             if str(c) not in numeric and str(c) != group_col]

    varying = [c for c in other
               if int(ordered.groupby(group_col, dropna=True)[c]
                      .nunique(dropna=True).max() or 0) > 1]

    grouped = ordered.groupby(group_col, dropna=True, sort=False)
    if method == MEAN:
        out = grouped[numeric].mean()
    elif method == FIRST:
        out = grouped[numeric].first()
    elif method == LAST:
        out = grouped[numeric].last()
    else:
        out = grouped[numeric].last() - grouped[numeric].first()
        if target and target in out.columns:
            # The outcome is NOT differenced. A change score on the target is a
            # different research question — "who improved" rather than "who is
            # ill" — and silently substituting one for the other would be the
            # app choosing the paper's question.
            out[target] = grouped[target].last()

    for c in other:
        out[c] = grouped[c].first()
    out = out[[c for c in df.columns if c != group_col]]
    out.insert(0, group_col, out.index)
    out.index = pd.RangeIndex(len(out))

    return {
        "frame": out,
        "n_before": int(len(df)),
        "n_after": int(len(out)),
        "method": method,
        "varying_categoricals": varying,
        "target_not_differenced": bool(method == CHANGE and target
                                       and target in numeric),
        "sentence": (
            f"{_MENU[method]['sentence']} {len(df):,} rows became "
            f"{len(out):,}, one per {group_col}."
            + (f" The outcome `{target}` was taken from each person's last "
               f"record rather than differenced, because a change score on the "
               f"outcome asks a different question."
               if method == CHANGE and target and target in numeric else "")
            + (f" {len(varying)} non-numeric column(s) varied within a person "
               f"and took the first value: "
               + ", ".join(f"`{c}`" for c in varying[:4]) + "."
               if varying else "")),
    }


# ── question 7 ───────────────────────────────────────────────────────────────

TEMPORAL_WHY = (
    "A random split — even one grouped by person — is optimistic when the task "
    "is predicting a later outcome from earlier measurements, because the model "
    "trains on rows from after the rows it is scored on. TRIPOD treats temporal "
    "validation as a distinct thing from internal validation.")

TEMPORAL_CONSUMER = (
    "`ml/splits.py` reads this to choose between its chronological and grouped "
    "strategies — both already exist with sixteen equivalence tests, and what "
    "has been missing is the routing that decides when each applies. Answering "
    "yes selects the chronological split, grouped as well because people "
    "repeat. Answering wrongly does not raise an error: it produces a held-out "
    "score that is optimistic by an amount nothing on screen can show you.")


def split_strategy(temporal: bool, unit: str) -> Dict[str, str]:
    """What question 7's answer selects. Named, so the record can carry it."""
    if temporal and unit == UNIT_RECORD:
        return {"strategy": "chronological_grouped",
                "sentence": ("The held-out rows are the latest ones, and whole "
                             "people are held out rather than individual "
                             "records — so the model is scored on people it "
                             "never saw, at times after the ones it trained "
                             "on.")}
    return {"strategy": "grouped",
            "sentence": ("Whole people are held out rather than individual "
                         "records, so nobody appears on both sides of the "
                         "split.")}

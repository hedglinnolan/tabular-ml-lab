"""Test-set lockbox: freeze held-out test rows the moment the modeling
problem is defined.

Methodological contract (split-first workflow):
- Immediately after the user confirms data + target + task type on Upload &
  Audit, a seeded (stratified where feasible) test fraction is drawn and its
  row labels are frozen in _state()['test_lockbox'].
- Every target-aware step upstream of Train & Compare — feature-engineering
  fits, feature selection, target-association views — operates on training
  rows only, via train_row_mask().
- Train & Compare consumes the frozen labels as THE test set and only divides
  the remaining rows into train/validation. Opening it is COUNTED
  (`opened_count`, `record_lockbox_open`): the contract is one opening, both
  train buttons are re-runnable, and a chip that promised "opened once" beside
  an uncounted button was asserting something it could not know (`SWEEP-008`).
- 'Exploratory mode' disables the quarantine explicitly; downstream metrics
  and the manuscript are watermarked accordingly (never silently).

The lockbox stores index LABELS (not positions) so membership survives
feature engineering and row filtering, which preserve the original index.
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
# The host is reached through _state(), not imported at module scope. Most of
# this file is decision logic — the grouping-candidate ranking, the repeated-
# subject detection, the signature — and a module-level import made all of it
# unreachable without Streamlit. Only render_lockbox_status() is true UI.
#
# ARCHITECTURE.md 02 lists what is actually read: exploratory_mode,
# test_lockbox, test_lockbox_fraction, random_seed, plus the sealed lockbox
# written back. Those reads are the remaining coupling; routing them through one
# accessor makes them countable and gives the parameterised version one place to
# land.


class _NoHost(dict):
    """Stands in for session state when there is no host.

    A dict, so a headless caller can read defaults and get sensible answers
    rather than an AttributeError three frames down. Writes go nowhere, which is
    correct: with no host there is no session to seal a lockbox into.
    """

    def get(self, key, default=None):
        return default


def _state():
    """The host's session state, or an empty stand-in when there is no host.

    Only `ImportError` falls back. A broader `except` here would be the landmine
    `TRANSITION_PLAN.md` §04 catalogues — *exceptions swallowed to a clean
    default* — and it is not hypothetical: the first version of this function
    caught everything, and when a bad edit made it recurse, the `RecursionError`
    came back as an empty session state. Every lockbox read then returned a
    plausible `None` and the quarantine silently stopped existing. A failure to
    reach the host has to be loud.
    """
    try:
        import streamlit as st
    except ImportError:
        return _NoHost()
    return st.session_state

DEFAULT_TEST_FRACTION = 0.15
_MIN_ROWS_FOR_LOCKBOX = 10
# Splitting by subject needs enough subjects for the fraction to be meaningful.
_MIN_GROUPS_FOR_GROUPED_LOCKBOX = 8
# An identifier repeats a handful of times; a category repeats hundreds. Above
# this, a column is not read as an identifier — but the rejection is REPORTED
# rather than silent, because a many-to-many merge product lands here and it
# repeats harder, not less (constitution §03, `IMPORT-020`).
_MAX_ROWS_PER_GROUP = 50

# The three bases a seal may rest on. Never two: `undetermined` is what the
# lockbox records when it could not read the data's grain, and it is not the
# same claim as "there is no repetition".
SEAL_GROUPED = "grouped"
SEAL_ABANDONED = "repetition_found_grouping_abandoned"
SEAL_UNDETERMINED = "undetermined"
SEAL_CROSS_SECTIONAL = "cross_sectional"

# How the basis was arrived at. Today everything is `detected`; constitution §02
# adds `user_stated` when the grain question ships and `inherited_from_assembly`
# when a project arrives through multi-file assembly having already answered it.
# The field exists now so those land without a schema change to a persisted,
# round-tripped artifact.
BASIS_DETECTED = "detected"
BASIS_USER_STATED = "user_stated"
BASIS_INHERITED = "inherited_from_assembly"


def is_exploratory() -> bool:
    """Explicit, user-chosen escape hatch. Never enabled by default."""
    return bool(_state().get("exploratory_mode", False))


def get_lockbox() -> Optional[Dict[str, Any]]:
    lb = _state().get("test_lockbox")
    if lb and lb.get("labels") is not None:
        return lb
    return None


def _index_identity(df: pd.DataFrame) -> str:
    """A hash of the row LABELS, which are what the lockbox actually seals.

    The signature used to cover cell content only, so a frame whose rows were
    renumbered — `reset_index()`, a re-export, a fresh assembly of the same
    tables — hashed IDENTICALLY to the frame the seal was drawn on, and
    `ensure_lockbox` handed back the stale lockbox. Its labels then named
    different people, or nobody at all.

    Order-insensitive on purpose: membership is decided by label, so a permuted
    frame with the same labels seals exactly the same rows and is the same
    identity. A legitimate re-upload of the same file rebuilds the same labels
    (a fresh RangeIndex over the same rows) and still matches, which is what
    keeps a redraw from firing on every reload.
    """
    try:
        return str(int(pd.util.hash_pandas_object(pd.Series(df.index)).sum()))
    except Exception:
        try:
            return f"{len(df.index)}|{df.index.dtype}|{df.index[:1].tolist()}"
        except Exception:
            return str(len(df))


def _lockbox_signature(df: pd.DataFrame, target_col: str, task_type: str,
                       fraction: float, seed: int, group_col: Optional[str] = None) -> str:
    try:
        content = int(pd.util.hash_pandas_object(df, index=False).sum())
    except Exception:
        # Unhashable cells (e.g. a list from nested JSON) must not silently
        # collapse the signature to something stable — that would stop the
        # lockbox from ever being redrawn. Fall back to a coarse but
        # content-sensitive descriptor.
        try:
            content = f"{df.shape}|{tuple(df.columns)}|{df.notna().sum().sum()}"
        except Exception:
            content = str(df.shape)
    return (f"{content}|{_index_identity(df)}|{df.shape}|{target_col}|{task_type}|"
            f"{fraction:.4f}|{seed}|{group_col or ''}")


# Whole-token names for a subject identifier. A bare substring test — "id" in
# name — matches uric_acid, folic_acid, linoleic_acid, lipid, oxidized,
# residual, and NHANES's entire RID* family including RIDAGEYR, which is age in
# years. Grouping the held-out set by one of those splits the study by a
# covariate: every test row then holds a value the model never trained on.
_SUBJECT_ID_TOKENS = frozenset({
    "id", "ids", "seqn", "subject", "subjid", "usubjid", "subjectid",
    "participant", "participantid", "patient", "patientid", "pid", "sid",
    "record", "recordid", "case", "caseid", "person", "personid", "mrn",
    "studyid", "sampleid", "specimenid", "respondent", "respondentid",
    "enrollmentid", "uid", "guid",
})


# An ID column names one of three very different things, and the lockbox must
# not confuse them. A PERSON is what the quarantine is about. A CLUSTER (site,
# plate, batch) CONTAINS people: grouping by one keeps whole people on one side,
# but only if the people really are nested inside it — assay plates are usually
# crossed with participants, so grouping by plate splits a person in half. A
# WITHIN-SUBJECT unit (visit, replicate, aliquot) is finer than a person, so
# grouping by one is exactly the leak we are trying to prevent.
_PERSON_TOKENS = frozenset({
    "subject", "subjid", "usubjid", "subjectid", "participant", "participantid",
    "patient", "patientid", "person", "personid", "respondent", "respondentid",
    "mrn", "seqn", "pid", "sid", "individual", "enrollmentid", "enrolment",
})
_CLUSTER_TOKENS = frozenset({
    "site", "sites", "clinic", "center", "centre", "hospital", "institution",
    "lab", "laboratory", "plate", "batch", "chip", "array", "well", "flowcell",
    "lane", "sequencer", "instrument", "machine", "cohort", "arm", "block",
    "family", "household", "cluster", "school", "region", "ward", "practice",
    "group", "run", "study", "wave", "center_id", "team",
})
_WITHIN_SUBJECT_TOKENS = frozenset({
    "visit", "timepoint", "occasion", "cycle", "session", "followup",
    "round", "measurement", "observation", "obs", "event", "encounter",
    "replicate", "aliquot", "draw", "reading", "trial",
})


def _tokenize(col: Any) -> List[str]:
    """Lowercase tokens, splitting on separators AND camelCase."""
    import re as _re
    spaced = _re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(col))
    return [t for t in _re.split(r"[^A-Za-z0-9]+", spaced.lower()) if t]


def _name_looks_like_a_subject_id(col: Any) -> bool:
    """Whole-token match, so `uric_acid` and `RIDAGEYR` are not subject IDs."""
    tokens = _tokenize(col)
    if any(t in _SUBJECT_ID_TOKENS for t in tokens):
        return True
    # A single run-together token is an ID only if the whole thing is one.
    return "".join(tokens) in _SUBJECT_ID_TOKENS


def _id_kind(col: Any) -> Optional[str]:
    """'subject', 'cluster', 'record' — or None when the name is not an ID."""
    if not _name_looks_like_a_subject_id(col):
        return None
    tokens = set(_tokenize(col))
    if tokens & _WITHIN_SUBJECT_TOKENS:
        return None                       # finer than a person: never group by it
    if tokens & _PERSON_TOKENS:
        return "subject"
    if tokens & _CLUSTER_TOKENS:
        return "cluster"
    return "record"


# ── the name-blind half ──────────────────────────────────────────────────────
# A column is ROSTER-SHAPED when its values repeat like a list of people:
# many distinct values, each carrying a handful of rows, and carrying them
# REGULARLY. The test never sees the column's name, so no spelling can defeat
# it — which is the whole point, because `_SUBJECT_ID_TOKENS` is a list and the
# next unrecognized spelling is always one dataset away (`IMPORT-022`).
#
# Regularity is what separates a roster from a coincidence: three visits per
# person is a study design; an integer covariate that happens to average 2.4
# rows per value is not. Measured as the share of values whose row count equals
# the modal row count.
_ROSTER_MIN_REPEATS = 2.0
_ROSTER_MIN_REGULARITY = 0.5


def _roster_shape(s: pd.Series, k: int, n: int) -> Optional[Dict[str, Any]]:
    """Shape facts about a column that repeats like a roster, or None.

    Deliberately NOT told the column's name. The bounds are the ones the seal
    already uses, so the detector and the split cannot disagree about what
    "repeats" means.

    The floor of two rows per value is the price of being name-blind: at 1.3
    rows per value a partial-follow-up roster and a mostly-unique measurement
    are the same shape, and there is no evidence left to separate them. That
    case is covered by the name list when it recognizes the spelling and by the
    declared subject column when it does not — never by guessing here.
    """
    try:
        if pd.api.types.is_float_dtype(s) or pd.api.types.is_bool_dtype(s):
            return None                   # a measurement or a flag, not a roster
        if pd.api.types.is_datetime64_any_dtype(s):
            return None                   # a visit date repeats; it is not a person
    except Exception:
        return None
    if k < _MIN_GROUPS_FOR_GROUPED_LOCKBOX:
        return None                       # a stratum, not a roster
    non_null = int(s.notna().sum())
    if not non_null:
        return None
    rows_per = non_null / k
    if rows_per < _ROSTER_MIN_REPEATS or rows_per > _MAX_ROWS_PER_GROUP:
        return None
    try:
        sizes = s.dropna().value_counts()
        modal = int(sizes.mode().iloc[0]) if len(sizes) else 0
        regular = float((sizes == modal).mean()) if len(sizes) else 0.0
    except Exception:
        return None
    if regular < _ROSTER_MIN_REGULARITY:
        return None
    return {"n_groups": k, "n_rows": n, "rows_per": rows_per,
            "modal_rows_per": modal, "regular_share": regular}


def _nests_within(df: pd.DataFrame, fine: str, coarse: str) -> bool:
    """True when every value of `fine` sits inside exactly one value of `coarse`."""
    try:
        pair = df[[fine, coarse]].dropna()
        if pair.empty:
            return False
        return int(pair.groupby(fine, observed=True)[coarse].nunique().max()) == 1
    except Exception:
        return False


_NOUNS = {"subject": "subjects", "record": "records"}


def group_noun(kind: str, col: Any = "") -> str:
    """What to call the units on screen. A site is not a subject."""
    return _NOUNS.get(kind) or f"`{col}` groups"


def rank_grouping_candidates(df: pd.DataFrame,
                             candidate_cols: Optional[list] = None
                             ) -> List[Dict[str, Any]]:
    """Columns the held-out split could be grouped by, best first.

    Ranked by what the column IS, not by how many values it has. Ranking by
    count in either direction is wrong: most-distinct lets a per-sample barcode
    outrank the participant, and fewest-distinct lets `plate_id` or `site_id`
    outrank it — both of which the token list admits only because `id` is a
    token in its own right. A coarser column is preferred ONLY where the data
    show the finer one really is nested inside it.
    """
    if df is None or df.empty:
        return []
    n = len(df)
    cols = candidate_cols if candidate_cols is not None else list(df.columns)
    found: List[Dict[str, Any]] = []
    unclear: List[Dict[str, Any]] = []
    _state().pop("_lockbox_repetition_unclear", None)
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col]
        if isinstance(s, pd.DataFrame):
            continue
        try:
            k = int(s.nunique(dropna=True))
        except TypeError:
            continue
        if k < 2 or k >= n:
            continue                     # unique per row => no repetition
        rows_per = n / k
        # THE LOWER BOUND IS GONE. It rejected 1.30 rows per value, and any
        # k < n means repetition exists by definition — partial follow-up,
        # where only some subjects have a second visit, is the commonest
        # longitudinal shape there is and it lands at about 1.3 (`IMPORT-020`).
        # Removing it does not close the hole; constitution §02 is explicit
        # that name lists and ratio bounds cannot, because the engine is
        # guessing at something the user simply knows. It is removed because it
        # is wrong on its own terms, and a heuristic demoted to *suggestion and
        # contradiction detector* is worse at both jobs when its bound is wrong.
        #
        # The upper bound has a real purpose — an identifier repeats a handful
        # of times, a category repeats hundreds — but rejecting on it must not
        # be silent, because the shape that trips it is a many-to-many merge
        # product, which repeats HARDER rather than less. Those columns are
        # reported as `unclear` so the seal can record `undetermined` (§03)
        # instead of rendering a clean lock over them.
        kind = _id_kind(col)
        if kind is None:
            # MEASURED REPETITION IS NEVER DISCARDED. This used to `continue`,
            # and a column holding 60 people across 180 rows vanished because
            # `SUBJ` is not in the token list: the split went by row, the same
            # person landed on both sides, and the seal recorded
            # `cross_sectional` — a positive claim that the study has one row
            # per person, asserted over repetition this loop had already
            # measured (`IMPORT-020`, `IMPORT-022`).
            #
            # A name we do not recognize is not evidence of anything. The
            # SHAPE is, so a roster-shaped column reaches `unclear` and the
            # seal records `undetermined` — a disclosure rather than a guess.
            # It is not promoted to a group column: which column identifies a
            # participant is the user's to state, not ours to infer from shape
            # alone (the picker on Upload & Audit is where they state it).
            shape = _roster_shape(s, k, n)
            if shape:
                unclear.append({"column": str(col), "kind": None,
                                "reason": "unrecognized_name", **shape})
            continue
        if rows_per > _MAX_ROWS_PER_GROUP:
            unclear.append({"column": str(col), "n_groups": k, "n_rows": n,
                            "kind": kind, "rows_per": rows_per,
                            "reason": "repeats_too_often"})
            continue
        found.append({"column": str(col), "n_groups": k, "n_rows": n, "kind": kind})
    if not found:
        # Nothing groupable, but something repeated far too often to be an
        # identifier. The caller needs to know the difference between "no
        # repetition" and "repetition we could not read".
        if unclear:
            # A column whose name we did not recognize but whose shape is a
            # roster is the more actionable of the two, so it is named first:
            # the user can point at it in the subject picker.
            _state()["_lockbox_repetition_unclear"] = sorted(
                unclear,
                key=lambda c: (0 if c.get("reason") == "unrecognized_name" else 1,
                               -(c.get("regular_share") or 0.0),
                               -c["rows_per"]))[:3]
        return []

    # A person beats a bare record id; both beat a cluster. Only if nothing
    # names a person do we consider grouping by the thing people sit inside.
    for kind in ("subject", "record", "cluster"):
        tier = [c for c in found if c["kind"] == kind]
        if tier:
            break

    # Within the tier, climb to the coarsest column that genuinely CONTAINS the
    # finest one (subject_id over subject_visit_id), and stop there.
    tier.sort(key=lambda c: -c["n_groups"])
    best = tier[0]
    for other in tier[1:]:
        if (other["n_groups"] < best["n_groups"]
                and _nests_within(df, best["column"], other["column"])):
            best = other
    ranked = [best] + [c for c in tier if c["column"] != best["column"]]
    for c in ranked:
        c["noun"] = group_noun(c["kind"], c["column"])
    return ranked


def detect_repeated_subjects(df: pd.DataFrame,
                             candidate_cols: Optional[list] = None
                             ) -> Optional[Tuple[str, int, int]]:
    """Find a column that looks like a subject ID appearing on several rows.

    Returns (column, n_subjects, n_rows) or None. Used to catch the case that
    silently defeats the quarantine: a merge with repeated measures puts the
    SAME subject in both the training rows and the sealed test rows, so the
    "held-out" set was already trained on.
    """
    ranked = rank_grouping_candidates(df, candidate_cols)
    if not ranked:
        return None
    top = ranked[0]
    return (top["column"], top["n_groups"], top["n_rows"])


def declared_subject_col() -> Tuple[bool, Optional[str]]:
    """What the USER said identifies a subject: (answered, column).

    `(True, None)` is the answer "one row per participant" and is a different
    fact from `(False, None)`, which is no answer at all. The first stops the
    heuristic; the second is what runs it. Read from the same
    `cohort_structure_detection` record Train & Compare reads, so the seal and
    the train/val split cannot disagree about who the subject is.
    """
    cohort = _state().get("cohort_structure_detection")
    if cohort is None or not getattr(cohort, "entity_id_override_enabled", False):
        return False, None
    return True, (getattr(cohort, "entity_id_override_value", None) or None)


def record_lockbox_open(source: str = "") -> Optional[Dict[str, Any]]:
    """Count one opening of the sealed test set, with a timestamp.

    Called where held-out metrics are actually computed. The count is what
    makes "opened once" a measured fact rather than a promise the interface
    repeats: both train buttons are re-runnable, so a second opening costs two
    clicks, and nothing else in the app would notice (`SWEEP-008`).
    """
    lb = get_lockbox()
    if lb is None:
        return None
    from datetime import datetime
    opens = list(lb.get("opened_at") or [])
    opens.append(datetime.now().isoformat(timespec="seconds")
                 + (f" ({source})" if source else ""))
    lb["opened_at"] = opens
    lb["opened_count"] = int(lb.get("opened_count", 0)) + 1
    _state()["test_lockbox"] = lb
    return lb


def lockbox_open_count() -> int:
    """How many times held-out metrics have been computed against this seal."""
    lb = get_lockbox()
    if lb is None:
        return 0
    try:
        return int(lb.get("opened_count", 0) or 0)
    except (TypeError, ValueError):
        return 0


def quarantine_is_active() -> bool:
    """True only when a lockbox exists AND is being enforced.

    The question every page-level caption about held-out rows has to ask
    before it asserts anything. `train_row_mask` returns an all-True mask both
    when the quarantine is OFF and when there is no lockbox at all, and a
    caption guarded on exploratory mode alone therefore claimed an exclusion
    that had not happened (`MINE-005`).
    """
    return get_lockbox() is not None and not is_exploratory()


def lockbox_absence_reason() -> Optional[Dict[str, Any]]:
    """Why there is no lockbox, when we know — for the page to render.

    Absence is a STATE, not a None. Something refused to seal, or nothing ever
    tried; either way the researcher is owed the difference between "held out"
    and "nothing was held out".
    """
    if get_lockbox() is not None:
        return None
    return _state().get("_lockbox_not_sealed") or None


def ensure_lockbox(df: pd.DataFrame, target_col: str, task_type: str,
                   fraction: Optional[float] = None,
                   seed: Optional[int] = None,
                   group_col: Optional[str] = None,
                   stratify_cols: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """Create or refresh the lockbox; rebuild only when its inputs change.

    Returns the lockbox dict, or None when a lockbox cannot be drawn
    (no target, too few rows). A rebuild that REPLACES an existing lockbox
    invalidates downstream results — models evaluated against a different
    test set are no longer comparable.
    """
    def _cannot_seal(reason: str, **detail) -> Optional[Dict[str, Any]]:
        """No seal drawn — record WHY, so the page can render the absence.

        Only when there is no lockbox at all: an earlier seal that still
        stands is not an absence.
        """
        lb = get_lockbox()
        if lb is None:
            _state()["_lockbox_not_sealed"] = {"reason": reason, **detail}
        return lb

    if df is None or not target_col or target_col not in df.columns:
        return _cannot_seal("no_target")

    fraction = float(fraction if fraction is not None
                     else _state().get("test_lockbox_fraction", DEFAULT_TEST_FRACTION))
    seed = int(seed if seed is not None else _state().get("random_seed", 42))

    y = df[target_col]
    eligible = df.index[y.notna()]
    if len(eligible) < _MIN_ROWS_FOR_LOCKBOX:
        return _cannot_seal("too_few_rows", n_eligible=int(len(eligible)),
                            minimum=_MIN_ROWS_FOR_LOCKBOX)

    # A duplicated row label cannot be sealed. Membership is decided by LABEL
    # (module docstring), so sealing one label seals every row carrying it,
    # while `n_test` and `fraction` count the labels DRAWN: a 15% chip over a
    # 42% holdout (`IMPORT-207`). `ml/splits.resolve_split_rows` already
    # refuses a duplicated label for the same reason; refusing here keeps the
    # two consistent, and refusing is louder than a number nobody can check.
    if df.index.has_duplicates:
        _dupes = df.index[df.index.duplicated()].unique()
        return _cannot_seal("duplicate_row_labels",
                            n_duplicated=int(len(_dupes)),
                            examples=[str(x) for x in list(_dupes[:3])])

    # A DECLARED subject column wins over the heuristic, and a declared "one
    # row per participant" stops the heuristic from guessing at all — that is
    # the difference between an answer and an absence of one. Read here rather
    # than only from the caller so every seal site inherits the declaration.
    group_kind = "subject"
    _declared, _declared_col = declared_subject_col()
    if _declared and _declared_col and not group_col:
        group_col = _declared_col
    if not group_col:
        if _declared:
            # The user said there is no subject column. Believe them, and do
            # not let a leftover `unclear` from a previous frame turn their
            # answer into "we could not tell".
            _state().pop("_lockbox_repetition_unclear", None)
        else:
            ranked = rank_grouping_candidates(df)
            if ranked:
                group_col = ranked[0]["column"]
                group_kind = ranked[0]["kind"]
    else:
        group_kind = _id_kind(group_col) or "subject"
    if group_col and group_col not in df.columns:
        group_col = None
    # A declaration that names a column this frame does not have is not an
    # answer any more; say so rather than quietly reverting to a row split.
    _state().pop("_lockbox_declared_missing", None)
    if _declared and _declared_col and _declared_col not in df.columns:
        _state()["_lockbox_declared_missing"] = str(_declared_col)

    sig = _lockbox_signature(df, target_col, task_type, fraction, seed,
                            f"{group_col}|{'+'.join(sorted(stratify_cols or []))}")
    existing = get_lockbox()
    if existing and existing.get("signature") == sig:
        return existing

    # Every cohort run inherits its slice of ONE split, drawn before the study
    # was ever divided. Redrawing mid-run silently re-partitions the study: rows
    # sealed since upload become trainable, and two runs can no longer be
    # compared because run 2's test people may have been run 1's training
    # people. Refuse, and let the page say so rather than proceeding quietly.
    if existing is not None:
        from utils.cohorts import active_cohort
        run = active_cohort()
        if run is not None:
            # The refusal fires on ANY change of inputs, including a change of
            # OUTCOME. The sealed rows were chosen among those with a value for
            # the OLD outcome; if the new one was assayed on a sub-sample —
            # ordinary in nutrition and omics — most of them cannot be scored at
            # all, and the chip would still report the old count. Carry the real
            # number so the page can state it.
            _sealed = [lbl for lbl in existing["labels"] if lbl in df.index]
            _state()["_lockbox_redraw_refused"] = {
                "column": run["column"], "label": run["label"],
                "drawn_for": existing.get("target_col"),
                "target": target_col,
                "target_changed": bool(existing.get("target_col")
                                       and existing.get("target_col") != target_col),
                "n_sealed": int(len(existing["labels"])),
                "n_scoreable": int(y.loc[_sealed].notna().sum()) if _sealed else 0,
            }
            return existing

    from sklearn.model_selection import train_test_split

    def _build_stratum():
        """Composite stratum for the split, and the columns it ended up using.

        The outcome always — a test set holding twice the event rate makes every
        metric meaningless. Plus any demographic the researcher named, because a
        test set that is 75% men when the cohort is 71% men reports a number that
        does not describe the study population.
        """
        parts: List[pd.Series] = []
        used: List[str] = []
        if task_type == "classification":
            parts.append(y.loc[eligible].astype(str))
            used.append(target_col)
        for col in (stratify_cols or []):
            if col in df.columns and col != target_col:
                parts.append(df.loc[eligible, col].astype(str))
                used.append(col)
        # Every extra variable multiplies the cells, and one singleton cell makes
        # the split impossible. Drop the least important variable until it fits,
        # rather than failing or silently falling back to no stratification.
        while parts:
            combined = parts[0]
            for extra in parts[1:]:
                combined = combined.str.cat(extra, sep="|")
            counts = combined.value_counts()
            if counts.min() >= 2 and (counts * fraction).min() >= 1:
                return combined, used
            parts.pop()
            used.pop()
        return None, []

    grouped = False
    test_labels = None
    _state().pop("_lockbox_grouping_abandoned", None)
    # Constitution §03: the seal states its own basis, and there are three
    # states rather than two. `undetermined` is what this is when the detector
    # could not read the data's grain — a different claim from "there is no
    # repetition", and the one `IMPORT-020` proved was indistinguishable from
    # success. Assume the honest default and narrow it below.
    seal_basis = SEAL_CROSS_SECTIONAL
    if not group_col and _state().get("_lockbox_repetition_unclear"):
        seal_basis = SEAL_UNDETERMINED
    if group_col:
        groups = df.loc[eligible, group_col]
        n_groups = int(groups.nunique(dropna=False))
        if n_groups >= _MIN_GROUPS_FOR_GROUPED_LOCKBOX:
            from sklearn.model_selection import GroupShuffleSplit
            try:
                gss = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
                _, test_pos = next(gss.split(eligible, groups=groups))
                test_labels = eligible[test_pos]
                grouped = True
                seal_basis = SEAL_GROUPED
            except Exception:
                test_labels = None
        if test_labels is None:
            # Too few groups to hold any out. Falling back to a row-wise split
            # reinstates the exact leak the detector exists to catch — the same
            # person on both sides — so the page has to say so. There is no
            # finer column to fall back to: anything finer than the unit that
            # repeats is inside it, and splitting by that IS the leak.
            _state()["_lockbox_grouping_abandoned"] = {
                "column": group_col, "n_groups": n_groups,
                "noun": group_noun(group_kind, group_col),
                "minimum": _MIN_GROUPS_FOR_GROUPED_LOCKBOX,
            }
            seal_basis = SEAL_ABANDONED
            group_col = None

    # Stratification is decided AFTER the grouped attempt resolves. Deciding it
    # earlier meant that a group column with too few subjects to split by fell
    # back to an ordinary split carrying NO stratification at all — the one
    # case that produced a test set unrepresentative of both the outcome and
    # the demographics, silently.
    stratify: Optional[pd.Series] = None
    strata_used: List[str] = []
    if test_labels is None:
        stratify, strata_used = _build_stratum()
        try:
            _, test_labels = train_test_split(
                eligible, test_size=fraction, random_state=seed, stratify=stratify
            )
        except ValueError:
            stratify, strata_used = None, []
            _, test_labels = train_test_split(
                eligible, test_size=fraction, random_state=seed
            )

    # GroupShuffleSplit's test_size is a fraction of GROUPS, so the row
    # fraction it lands on differs from the one requested — the audit measured
    # 17.1% held out while the lockbox reported 15%. Report what was held out.
    _actual_fraction = (len(test_labels) / len(eligible)) if len(eligible) else fraction
    lockbox = {
        "labels": list(test_labels),
        "fraction": float(_actual_fraction),
        "fraction_requested": float(fraction),
        "seed": seed,
        "n_total": int(len(eligible)),
        "n_test": int(len(test_labels)),
        "signature": sig,
        # The outcome the split was drawn for. Eligibility is `y.notna()`, so a
        # lockbox is only meaningful for the outcome it was drawn on.
        "target_col": target_col,
        "stratified": stratify is not None,
        "strata": list(strata_used),
        # What was ASKED for, so a silently-dropped stratum can be named. The
        # composite is trimmed until the split is possible, and a researcher who
        # asked for a sex-balanced holdout was previously told only "stratified".
        "strata_requested": [c for c in ([target_col] if task_type == "classification" else [])
                             + list(stratify_cols or []) if c],
        "group_col": group_col if grouped else None,
        "n_test_groups": (int(df.loc[test_labels, group_col].nunique())
                          if grouped else None),
        # What the groups ARE. Calling 10 recruitment sites "10 subjects" tells
        # the researcher the holdout is a random sample of people when it is a
        # sample of whole sites — a different, harder test than they asked for.
        "group_kind": group_kind if grouped else None,
        "group_noun": group_noun(group_kind, group_col) if grouped else None,
        # ── constitution §03 · the seal states its own basis ──────────────
        # Three states, never two. `group_col: None` used to carry two
        # different claims — "this study has one row per person" and "we could
        # not tell" — and a consumer reading the record rather than the chip
        # could not separate them. That is the half of `IMPORT-021` that stayed
        # open and the whole of what `IMPORT-020` exploited.
        "seal_basis": seal_basis,
        # HOW we know, not just what we concluded. Everything today is
        # `detected`; §02 adds `user_stated` when the grain question ships and
        # `inherited_from_assembly` for a project that arrived through
        # multi-file assembly having already answered it. Written now so those
        # land without migrating a persisted, round-tripped artifact.
        "basis_source": BASIS_USER_STATED if _declared else BASIS_DETECTED,
        # Opening the sealed set is a COUNTED event, not a promise the chip
        # repeats. Zero here; `record_lockbox_open` is called where held-out
        # metrics are computed (`SWEEP-008`).
        "opened_count": 0,
        "opened_at": [],
        # What made the basis undetermined, so the disclosure can name it.
        "undetermined_because": (list(_state().get("_lockbox_repetition_unclear") or [])
                                 if seal_basis == SEAL_UNDETERMINED else None),
    }

    if existing is not None and existing.get("labels") != lockbox["labels"]:
        # Different test set → previous results are not comparable
        from utils.session_state import reset_downstream_results
        reset_downstream_results(clear_feature_engineering=False)
        # Let the page disclose the redraw — a silent reset reads as data loss
        _state()["_lockbox_redrawn"] = True

    _state().pop("_lockbox_not_sealed", None)
    _state()["test_lockbox"] = lockbox
    return lockbox


def train_row_mask(index: pd.Index) -> pd.Series:
    """Boolean Series over `index`: True for rows a target-aware step may see.

    In exploratory mode (or with no lockbox) every row is visible. Works on
    any frame that preserves the original row labels (df_engineered,
    filtered_data, column subsets).
    """
    lb = get_lockbox()
    if lb is None or is_exploratory():
        return pd.Series(True, index=index)
    test_set = set(lb["labels"])
    return pd.Series([lbl not in test_set for lbl in index], index=index)


def _scoreable_here(labels) -> Optional[int]:
    """How many of `labels` have a value for the outcome now configured.

    None when the question cannot be answered (no target, no frame) — silence
    is right there; a guessed count is not.
    """
    try:
        dc = _state().get("data_config")
        target = getattr(dc, "target_col", None) if dc else None
        if not target:
            return None
        from utils.session_state import get_data
        df = get_data(full_study=True)
        if df is None or target not in df.columns:
            return None
        present = [lbl for lbl in labels if lbl in df.index]
        return int(df.loc[present, target].notna().sum())
    except Exception:
        return None


def render_lockbox_status(context: str = "") -> None:
    """The quiet, consistent status chip shown on workflow pages.

    The one genuinely UI function in this module, so it is the one place that
    imports the host directly. Everything above it decides; this renders.
    """
    import streamlit as st

    if is_exploratory():
        st.warning(
            "🔓 **Exploratory mode** — the test-set quarantine is OFF. "
            "Target-aware steps see all rows; downstream metrics are for "
            "exploration and will be flagged as such in any export.",
            icon="🔓",
        )
        return
    lb = get_lockbox()
    if lb is None:
        # ABSENCE IS A STATE, not a None. This used to `return` and render
        # nothing, so every page that said "held-out test rows are excluded"
        # said it over a mask that excluded nothing (`MINE-005`). Exploratory
        # mode is loud because it is chosen; a lockbox that was never drawn is
        # not chosen, which is a reason to be louder rather than quieter.
        _dc = _state().get("data_config")
        if not getattr(_dc, "target_col", None):
            # No outcome chosen yet: there is no modeling problem to protect,
            # nothing has been asserted, and a warning here would be noise on a
            # page the user has not configured. Silence is honest; the moment a
            # target exists, absence becomes a finding.
            return
        _why = lockbox_absence_reason() or {}
        _reason = _why.get("reason")
        if _reason == "duplicate_row_labels":
            _detail = (
                f"The data has {_why.get('n_duplicated', '?')} repeated row "
                f"label(s) (e.g. {', '.join(_why.get('examples') or []) or '—'}), "
                f"so a label no longer names one row: sealing one would seal "
                f"every row carrying it, and the held-out fraction would not be "
                f"the one reported. Re-load the file with unique row labels — a "
                f"plain CSV export, or a JSON export without an index — and the "
                f"set will be sealed."
            )
        elif _reason == "too_few_rows":
            _detail = (
                f"Only {_why.get('n_eligible', '?')} row(s) have a value for the "
                f"outcome; at least {_why.get('minimum', _MIN_ROWS_FOR_LOCKBOX)} "
                f"are needed before holding any out."
            )
        elif _reason == "no_target":
            _detail = ("No outcome has been chosen yet, so there is nothing to "
                       "hold a test set out for.")
        else:
            _detail = ("No test set was sealed in this session — a restored save "
                       "file without a lockbox does this, as does a change that "
                       "cleared it.")
        st.warning(
            "🔓 **No held-out test set is in force.** " + _detail +
            " Until one is, target-aware steps (EDA target views, feature "
            "engineering fits, feature selection) see every row, and any "
            "performance measured afterwards is not held-out performance.",
            icon="🔓",
        )
        return
    extra = f" {context}" if context else ""

    # What the seal can honestly say about being opened. "Opened once" was
    # printed unconditionally beside two re-runnable train buttons, so a
    # researcher who iterated against the test metric was told the opposite of
    # what they had done (`SWEEP-008`).
    _opens = lockbox_open_count()
    if _opens == 0:
        _open_phrase = "not opened yet — it opens at Train & Compare"
    elif _opens == 1:
        _open_phrase = "opened once, at Train & Compare"
    else:
        _open_phrase = f"**opened {_opens} times** at Train & Compare"

    if _opens > 1:
        st.warning(
            f"⚠️ The sealed test set has been **opened {_opens} times** — "
            f"models have been scored against it on {_opens} separate training "
            f"runs. A held-out estimate is unbiased only for a single, final "
            f"evaluation; once a choice (features, preprocessing, models) is "
            f"made after seeing a held-out number, that number is part of the "
            f"model selection and reads better than it will on new data. Report "
            f"it as such, and say in the Methods that the set was accessed "
            f"{_opens} times."
        )

    # Held out is not the same as scoreable, and that is true on EVERY path
    # through this function — so it is computed here, once, above the branch,
    # rather than inside one of them.
    #
    # It used to be computed only inside the cohort-run branch below. The
    # ordinary path printed `lb['n_test']` unconditionally, so a row-dropping
    # step after the seal (the page-05 plausibility filter, `STATE-101`) left
    # the chip reporting 60 held-out rows where evaluation had 53. The
    # principle was stated correctly in `_scoreable_here`'s own docstring and
    # applied in one place out of two — `STATE-102`, and the fifth instance of
    # a rule stated in one place and used in another. `test_scoreable_locality`
    # asserts this call sits above the branch, so the next reader cannot
    # reintroduce the gap by adding a third path.
    _n_sealed_total = len(lb.get("labels") or [])
    _n_scoreable_total = _scoreable_here(lb.get("labels") or [])

    # A cohort run works on a subset, and the study-wide n is then simply not
    # this run's test set: "n=135" beside a 490-row run is a number the
    # researcher would write down and be wrong about.
    from utils.cohorts import active_cohort
    run = active_cohort()
    if run is not None:
        here = set(lb["labels"]) & set(run["labels"])
        n_here = len(here)
        # Held out is not the same as scoreable. If the outcome changed since
        # the split was sealed — and the lockbox refuses to re-draw mid-run —
        # the rows without a value for the CURRENT outcome cannot be scored, and
        # reporting the sealed count is a number a researcher would write down
        # and be wrong about.
        n_scoreable = _scoreable_here(here)
        if n_scoreable is not None and n_scoreable < n_here:
            st.warning(
                f"⚖️ Test set for this run ({run['column']} = {run['label']}): "
                f"**{n_scoreable:,} of {n_here:,}** held-out rows have a value for "
                f"the current outcome. The split was sealed for "
                f"`{lb.get('target_col')}`, and it is not re-drawn during a "
                f"one-group run, so performance will be measured on "
                f"{n_scoreable:,} rows — not {n_here:,}. To hold out a set drawn "
                f"for this outcome, go back to analyzing everyone first."
            )
            return
        st.caption(
            f"🔒 Test set for this run ({run['column']} = {run['label']}): "
            f"n={n_here:,} — this run's share of the {lb['n_test']:,} rows drawn "
            f"once at upload, before the study was split, so every run is "
            f"evaluated against the same held-out people. Sealed set "
            f"{_open_phrase}.{extra}"
        )
        return

    _asked = [c for c in (lb.get("strata_requested") or [])]
    _got = [c for c in (lb.get("strata") or [])]
    _dropped = [c for c in _asked if c not in _got]
    if _dropped:
        st.warning(
            f"⚖️ The held-out set could not be balanced on "
            f"{', '.join(f'`{c}`' for c in _dropped)} — too few people in some "
            f"combination of those groups to put any in both halves. It IS "
            f"balanced on {', '.join(f'`{c}`' for c in _got) if _got else 'nothing'}. "
            f"Check that the test set still resembles your study before "
            f"reporting performance by subgroup."
        )
    elif len(_got) > 1:
        st.caption(f"⚖️ Held-out set balanced on {', '.join(f'`{c}`' for c in _got)}.")

    _abandoned = _state().get("_lockbox_grouping_abandoned")
    if _abandoned:
        st.warning(
            f"⚠️ Rows repeat per `{_abandoned['column']}`, but there are only "
            f"{_abandoned['n_groups']} {_abandoned['noun']} — too few to hold any "
            f"out (at least {_abandoned['minimum']} are needed). The held-out set "
            f"was drawn by row instead, so the same "
            f"{_abandoned['noun'].rstrip('s')} can appear on both sides and "
            f"held-out performance will read better than it is. Treat these "
            f"numbers as exploratory."
        )

    # Constitution §03: an undetermined seal is never rendered as a clean lock.
    # Same treatment IMPORT-021 earned — advisory with exploratory labeling, not
    # a hard block, because a user who genuinely does not know their data's
    # shape should get honest numbers rather than a locked door.
    if lb.get("seal_basis") == SEAL_UNDETERMINED:
        _why = lb.get("undetermined_because") or []
        _cols = ", ".join(f"`{c['column']}`" for c in _why[:2]) or "a column"
        _named = [c for c in _why if c.get("reason") == "unrecognized_name"]
        if _named:
            # The shape says roster; only the SPELLING was unfamiliar. Name the
            # column and point at the control that settles it, because here the
            # user can answer in one click.
            _first = _named[0]
            st.warning(
                f"⚠️ Could not tell whether one person can appear in more than "
                f"one row. `{_first['column']}` repeats like a list of people — "
                f"{_first['n_groups']:,} values across {_first['n_rows']:,} rows, "
                f"about {_first['rows_per']:.1f} rows each — but it is not a "
                f"column name this app recognizes as an identifier, so it was "
                f"not used to draw the split. The held-out set was drawn by row: "
                f"if those rows do repeat people, the same person is on both "
                f"sides and held-out performance will read better than it is. "
                f"Say which column identifies a participant on **Upload & Audit** "
                f"(*Who is a subject?*) and the set will be re-drawn by subject."
            )
        else:
            _rate = max((c.get("rows_per") or 0) for c in _why) if _why else 0
            st.warning(
                f"⚠️ Could not tell whether one person can appear in more than one "
                f"row. {_cols} looks like an identifier but repeats about "
                f"{_rate:.0f} times per value — too often to read as one, and too "
                f"structured to ignore. The held-out set was drawn by row, so if "
                f"these rows do repeat people, the same person is on both sides and "
                f"held-out performance will read better than it is. Treat these "
                f"numbers as exploratory until you confirm the shape."
            )

    _declared_missing = _state().get("_lockbox_declared_missing")
    if _declared_missing:
        st.warning(
            f"⚠️ You named `{_declared_missing}` as the subject column, but it is "
            f"not in the data any more, so the held-out set was NOT drawn by "
            f"subject. Name a column that is still present on Upload & Audit."
        )

    # The same sentence the cohort branch has always had, on the path that
    # never had it. `n_test` is what was SEALED; what performance will be
    # measured on is what still has a value for the current outcome, and those
    # differ after any row-dropping step.
    if _n_scoreable_total is not None and _n_scoreable_total < _n_sealed_total:
        st.warning(
            f"⚖️ **{_n_scoreable_total:,} of {_n_sealed_total:,}** held-out rows "
            f"still have a value for `{lb.get('target_col') or 'the outcome'}`. "
            f"The rest were removed or blanked after the split was sealed, so "
            f"performance will be measured on {_n_scoreable_total:,} rows — not "
            f"{_n_sealed_total:,}. Check what dropped them before reporting a "
            f"held-out number.{extra}"
        )
        return

    if lb.get("group_col"):
        _noun = lb.get("group_noun") or "subjects"
        _one = _noun.rstrip('s') if _noun.endswith('s') else _noun
        st.caption(
            f"🔒 Test set: {lb['fraction']:.0%} (n={lb['n_test']} rows from "
            f"{lb.get('n_test_groups', '?')} {_noun}, split by '{lb['group_col']}' so no "
            f"{_one} appears on both sides) held out since upload — "
            f"{_open_phrase}.{extra}"
        )
        if lb.get("group_kind") == "cluster":
            st.caption(
                f"⚠️ `{lb['group_col']}` groups people, it does not identify them. "
                f"Whole groups were held out, so this is a test of generalizing to "
                f"a NEW {_one} — a harder question than the random holdout you "
                f"asked for. Name a participant column if that is not what you meant."
            )
    else:
        st.caption(
            f"🔒 Test set: {lb['fraction']:.0%} (n={lb['n_test']}"
            f"{', stratified' if lb.get('stratified') else ''}) held out since upload — "
            f"{_open_phrase}.{extra}"
        )

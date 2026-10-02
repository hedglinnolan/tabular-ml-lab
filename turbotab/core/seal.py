"""The seal: the held-out rows every held-out claim rests on (M2_CONTRACT §3; ROADMAP "The lockbox
constitution" §01–§05).

What this module decides, and where each piece runs:

* **The basis** (constitution §03): how the held-out rows were drawn, as one of four recorded
  states, persisted in the ``split`` artifact and rendered on the seal:

  ==================== =====================================================================
  ``grouped``          grouped by a named column: no unit is on both sides
  ``one_row_per_unit`` the user said (or the app stated, from a unique person identifier) that
                       each row is a different unit, and no identifier repeats
  ``abandoned``        repetition found but grouping abandoned (too few units, no column to
                       group by, or each row said to be a unit while an identifier repeats)
  ``undetermined``     the user answered "I don't know" to the grain question
  ==================== =====================================================================

  The seal needs a grain (M2_CONTRACT §12.2): answered, or stated by the Router. Skipping the
  question never yields ``undetermined``; the split question waits for it instead.

  ``abandoned`` and ``undetermined`` carry ``exploratory: true`` and are never drawn as a clean
  lock. ``undetermined`` is its own state, never a missing column, which a reader could not tell
  from a verified seal (the failure ``IMPORT-020`` names). The grain answer decides the basis;
  the data only confirms or contradicts it (constitution §02).
* **The chronological draw**, when the user said the task predicts later from earlier
  (``temporal.temporal``): whole units ordered by their last observation, the latest held out
  (``turbotab/engine.py::draw_holdout``, ``GUIDED-143``: a unit split across the boundary would
  trade one leak for another). A time column that cannot be read refuses; none named discloses.
  The cross-validation folds follow the same ranking (:func:`time_order`): forward chaining by whole
  unit, each fold scored by models fit on the folds before it (``turbotab/core/models/folds.py``).
* **What a holdout of this size can measure**: the split question's options, each with the
  precision of a held-out score on that many rows, ordered so cross-validation alone comes first
  when the usual holdout falls below a stated floor (:func:`holdout_options`). Never a refusal.
* **Withholding**: the fit scores the held-out rows once and keeps the scores out of its public
  artifact (``frames["sealed_scores"]``); :func:`serve_fit` serves ``holdout: null`` and
  ``holdout_sealed: true`` until ``open_seal`` is recorded, whatever artifact it is handed.
* **Refusals**: ``open_seal`` once, on a fresh fit, and never reverted; Decision A (orientation,
  grain, unit, aggregation) refused while the seal is drawn, with the re-seal path as the exit.
  So is any answer that changes what the draw reads once rows are held out (audit RO-02): the
  outcome, its task, a chronological request, a repair to a column the draw reads
  (:func:`draw_columns`), or a revert that undoes one. A repair the draw would read comes first:
  the split waits for a finding that would rewrite the outcome to be repaired or kept.
* **Post-seal marking**: a decision recorded after the seal was opened says so in its sentence
  (:func:`post_seal_sentence`), its record carries ``post_seal: true`` (``DecisionLog.append``),
  and the Results say when the fit changed after the opening (:func:`changed_after_seal`).

Importing this module registers the validators and the ``open_seal`` preview.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from turbotab.core import decisions
from turbotab.core.decisions import ProjectState, Refusal, Revert, SetSplit

BasisState = Literal["grouped", "one_row_per_unit", "abandoned", "undetermined"]
EXPLORATORY_STATES = ("abandoned", "undetermined")

# Fewer units than this and a grouped draw cannot hold any out whole in a meaningful way: the
# legacy lockbox's own floor (``utils/test_lockbox.py::_MIN_GROUPS_FOR_GROUPED_LOCKBOX``), read from
# it so the seal and the grain question cannot disagree about it.
DEFAULT_MIN_GROUPS = 8

# The chronological draw refuses when more than this share of rows has no readable time: ordering
# units by a last observation that is mostly missing would place them arbitrarily, and the seal
# would report a chronology it did not draw (``turbotab/engine.py::draw_holdout``).
MAX_UNDATED_SHARE = 0.10

SEALED_SCORES = "sealed_scores"  # the fit Bundle frame that holds the held-out scores


def min_groups() -> int:
    try:
        from turbotab.grain import _MIN_DISTINCT  # = utils.test_lockbox._MIN_GROUPS_FOR_GROUPED_LOCKBOX

        return int(_MIN_DISTINCT)
    except Exception:  # pragma: no cover - the legacy module is part of this repository
        return DEFAULT_MIN_GROUPS


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)


class SealBasis(_Model):
    """How the held-out rows were drawn: one of four states, never inferred silently."""

    state: BasisState
    # grouped: the column that keeps a unit's rows together; abandoned: the column that repeats.
    column: str | None
    label: str  # "grouped by `participant_id`" · "one row per unit" · "repetition found but …"
    sentence: str  # one plain sentence for the seal
    exploratory: bool  # abandoned or undetermined: never drawn as a clean lock
    # What the basis rests on: the grain answer, the grain as stated from a unique identifier
    # (not asked, M2_CONTRACT §10), the identifier among the confirmed roles, or the combining of
    # each unit's rows into one.
    source: Literal["grain", "stated", "roles", "aggregation"] | None
    n_units: int | None  # units the draw kept whole (grouped), or found repeating (abandoned)


class Chronology(_Model):
    """The chronological draw the temporal answer asked for, and whether it was drawn."""

    drawn: bool
    time_column: str | None
    boundary: str | None  # the earliest last observation among the held-out units
    n_units: int | None  # units held out, latest first
    n_undated: int  # units with no readable time: they train, never held out
    sentence: str


# ── the basis ────────────────────────────────────────────────────────────────


def _basis(state: BasisState, column: str | None, sentence: str, *,
           source: str | None = None, n_units: int | None = None) -> SealBasis:
    label = {
        "grouped": f"grouped by `{column}`",
        "one_row_per_unit": "one row per unit",
        "abandoned": "repetition found but grouping abandoned",
        "undetermined": "undetermined",
    }[state]
    return SealBasis(state=state, column=column, label=label, sentence=sentence,
                     exploratory=state in EXPLORATORY_STATES, source=source, n_units=n_units)


def _units(values: Any) -> int:
    import pandas as pd

    series = pd.Series(values, dtype=object)
    return int(series.nunique(dropna=True)) + int(series.isna().sum())


def _repeating(frame: Any, columns: Sequence[str]) -> list[tuple[int, str]]:
    """``(units, column)`` for each column whose values repeat across these rows, fewest first."""
    out = []
    for c in columns:
        if c not in frame.columns:
            continue
        values = frame[c]
        n_units = int(values.nunique(dropna=True))
        if 0 < n_units < int(values.notna().sum()):
            out.append((n_units, c))
    return sorted(out)


GRAIN_FIRST = ("The held-out rows are drawn by the grain answer: say first whether a unit can "
               "appear in more than one row.")


def decide_basis(state: Any, frame: Any, identifiers: Sequence[str], *,
                 stated: bool = False) -> tuple[SealBasis, str | None]:
    """The basis the grain answer and the data give, and the column to group the draw by.

    ``frame`` holds the identifier columns (and the grain's named column) over the rows the seal
    is drawn from. The grain answer is the authority; the data confirms or contradicts it. The
    seal needs a grain (M2_CONTRACT §12.2): answered, or ``stated`` from a unique identifier.
    ``undetermined`` comes only from the answer "I don't know" (``unknown``), never from silence.
    """
    n = int(len(frame))
    grain = getattr(state, "grain", None)
    if grain is None:
        raise ValueError(GRAIN_FIRST)
    named = getattr(grain, "id_column", None)
    repeating = _repeating(frame, list(dict.fromkeys([*([named] if named else []), *identifiers])))
    aggregated = getattr(state, "unit", None) == "unit" and getattr(state, "aggregation", None) is not None

    def grouped(column: str, source: str) -> tuple[SealBasis, str | None]:
        present = column in frame.columns
        if aggregated and source == "grain":
            units = n
            return (_basis("grouped", column,
                           f"Each row is one `{column}` after combining their rows, so no unit is "
                           f"on both sides of the seal.", source="aggregation", n_units=units),
                    column if present else None)
        units = _units(frame[column]) if present else 0
        if not present:
            return (_basis("abandoned", column,
                           f"The rows were said to repeat by `{column}`, but the table has no such "
                           f"column, so the held-out rows were drawn by row. A unit can sit on both "
                           f"sides: treat held-out scores as exploratory.", source=source), None)
        if units < min_groups():
            return (_basis("abandoned", column,
                           f"`{column}` repeats, but `{units}` units are too few to hold any out "
                           f"whole, so the held-out rows were drawn by row. A unit can sit on both "
                           f"sides: treat held-out scores as exploratory.",
                           source=source, n_units=units), None)
        return (_basis("grouped", column,
                       f"Held out by `{column}`: each of the `{units:,}` units sits wholly on one "
                       f"side, so no unit is both trained on and scored.",
                       source=source, n_units=units), column)

    if grain.grain == "unknown":
        seen = ""
        if repeating:
            units, column = repeating[0]
            seen = (f" `{column}` repeats (`{units:,}` values over `{n:,}` rows), so a unit can "
                    f"sit on both sides.")
        return (_basis("undetermined", None,
                       f"Held out by row, because whether a unit can appear in more than one row "
                       f"was answered as not known.{seen} This is not a verified clean split: "
                       f"treat held-out scores as exploratory.", source="grain"), None)
    if grain.grain == "repeated":
        if named:
            return grouped(named, "grain")
        if repeating:
            return grouped(repeating[0][1], "roles")
        return (_basis("abandoned", None,
                       "Units were said to repeat, but no column names the unit, so the held-out "
                       "rows were drawn by row. A unit can sit on both sides: treat held-out "
                       "scores as exploratory.", source="grain"), None)
    source = "stated" if stated else "grain"
    if repeating:
        units, column = repeating[0]
        said = "stated" if stated else "said"
        return (_basis("abandoned", column,
                       f"Each row was {said} to be a different unit, but `{column}` repeats "
                       f"(`{units:,}` values over `{n:,}` rows); the held-out rows were drawn by "
                       f"row as answered. Treat held-out scores as exploratory.",
                       source=source, n_units=units), None)
    if stated and named:
        return (_basis("one_row_per_unit", None,
                       f"Held out by row: every `{named}` appears once, so each row is a different "
                       f"unit.", source=source), None)
    return (_basis("one_row_per_unit", None,
                   "Held out by row: each row was said to be a different unit, and no "
                   "identifier repeats.", source=source), None)


# ── the chronological draw ───────────────────────────────────────────────────


def temporal_request(state: Any) -> tuple[bool, str | None]:
    """Whether the answers ask for a chronological split, and the column named as time."""
    temporal = getattr(state, "temporal", None)
    if temporal is None or not temporal.temporal:
        return False, None
    column = temporal.time_column
    if column is None:
        repeat = getattr(state, "repeat_kind", None)
        column = getattr(repeat, "time_column", None) if repeat is not None else None
    return True, column


def read_times(values: Any) -> Any:
    """Times as numbers that order correctly: numbers as they are, else parsed dates (ns)."""
    import pandas as pd

    series = pd.Series(values)
    if pd.api.types.is_bool_dtype(series):
        return np.full(len(series), np.nan)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = series
    else:
        parsed = pd.to_datetime(series.astype("string"), errors="coerce", format="mixed", utc=True)
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert(None)
    values = parsed.to_numpy(dtype="datetime64[ns]")
    out = values.astype("int64").astype(float)
    out[np.isnat(values)] = np.nan
    return out


def _show_time(value: float, dated: bool) -> str:
    import pandas as pd

    if dated:
        return str(pd.Timestamp(int(value)).date())
    return f"{value:g}"


def _units_by_time(times: Any, groups: Any | None, seed: int) -> tuple[Any, Any, Any, int]:
    """Each row's unit key, each unit's last time, the dated units latest-last, and the undated count.

    Units tied on their last time are ordered by a draw seeded by ``seed`` over the units in key
    order, so the ranking never depends on the order the rows arrive in. The chronological holdout
    and the time-ordered folds both rank units by this one function.
    """
    import pandas as pd

    times = np.asarray(times, dtype=float)
    n = len(times)
    keys = (np.asarray([str(g) for g in np.asarray(groups, dtype=object)], dtype=object)
            if groups is not None else np.arange(n).astype(str).astype(object))
    last = pd.Series(times).groupby(keys).max()  # NaN for a unit with no readable time
    dated_units = last.dropna()
    tiebreak = np.random.default_rng(seed).random(len(dated_units))
    ranked = dated_units.index.to_numpy()[np.lexsort((tiebreak, dated_units.to_numpy()))]
    return keys, dated_units, ranked, int(last.isna().sum())


def time_order(times: Any, groups: Any | None, seed: int) -> Any:
    """Each row's unit rank by its last observation (0 earliest); NaN when the unit has no time.

    The order the time-ordered folds follow (:mod:`turbotab.core.models.folds`): the same ranking
    the chronological holdout draws its latest units from.
    """
    keys, _, ranked, _ = _units_by_time(times, groups, seed)
    rank = {k: float(i) for i, k in enumerate(ranked.tolist())}
    return np.asarray([rank.get(k, np.nan) for k in keys.tolist()], dtype=float)


def chronological_holdout(times: Any, groups: Any | None, holdout: float, seed: int,
                          column: str, *, dated: bool) -> tuple[Any, Chronology]:
    """The held-out mask: whole units, ordered by their last observation, the latest held out.

    Units with no readable time are never held out (they train) and are counted. Units tied at
    the boundary are ordered by a seeded draw, and the sentence says so.
    """
    keys, dated_units, ranked, n_undated = _units_by_time(times, groups, seed)
    n = len(keys)
    mask = np.zeros(n, dtype=bool)
    word = "units" if groups is not None else "rows"
    if holdout <= 0 or len(dated_units) < 2:
        why = ("nothing is held out" if holdout <= 0 else f"fewer than two {word} have a readable "
               f"`{column}`")
        return mask, Chronology(drawn=False, time_column=column, boundary=None, n_units=None,
                                n_undated=n_undated,
                                sentence=f"No rows were held out by time: {why}.")
    n_hold = min(len(ranked) - 1, max(1, int(round(len(ranked) * holdout))))
    held = set(ranked[-n_hold:].tolist())
    mask = np.isin(keys, np.asarray(list(held), dtype=object))
    boundary_value = float(dated_units.loc[list(held)].min())
    train_last = float(dated_units.loc[ranked[:-n_hold]].max())
    boundary = _show_time(boundary_value, dated)
    tied = train_last >= boundary_value
    tie = (f" `{int((dated_units == boundary_value).sum()):,}` {word} share that last `{column}`; "
           f"which of them are held out was drawn at random." if tied else "")
    undated = (f" `{n_undated:,}` {word} with no readable `{column}` train and are never held out."
               if n_undated else "")
    sentence = (f"The latest `{n_hold:,}` {word} by their last `{column}` are held out, from "
                f"`{boundary}` on; every training {word[:-1]}'s last `{column}` comes "
                f"{'no later' if tied else 'earlier'}.{tie}{undated}")
    return mask, Chronology(drawn=True, time_column=column, boundary=boundary, n_units=n_hold,
                            n_undated=n_undated, sentence=sentence)


# ── the seal's inputs, for the split stage, its preview and the plan ─────────


@dataclass
class SealDraw:
    """Everything the split needs to draw the seal, and the facts it reports about it."""

    basis: SealBasis | None  # None: no grain yet, so no seal (``refusal`` says why)
    chronology: Chronology | None = None
    y: Any | None = None  # classification labels over the universe (stratification)
    groups: Any | None = None  # the grouping column's values over the universe
    grouped_by: str | None = None
    held: Any | None = None  # a held-out mask over the universe (the chronological draw)
    # Each row's unit rank in time over the universe (:func:`time_order`), when the answers ask for
    # time order and the time column reads: the folds then forward-chain by whole unit (MA-11).
    order: Any | None = None
    refusal: str | None = None  # why the seal cannot be drawn as the answers ask
    labels: Any | None = field(default=None, repr=False)  # the outcome over the universe (counts)
    read: list[str] = field(default_factory=list)  # the columns read over the universe, to say so

    @property
    def exploratory(self) -> bool:
        return bool((self.basis is not None and self.basis.exploratory)
                    or (self.chronology is not None and not self.chronology.drawn
                        and self.chronology.time_column is None))

    def split_args(self) -> dict[str, Any]:
        """Keyword arguments for ``stages.rows.draw_split``."""
        return {"y": self.y, "groups": self.groups, "grouped_by": self.grouped_by, "held": self.held,
                "order": self.order}

    def facts(self) -> dict[str, Any]:
        """What the split artifact reports about the seal (its ``basis``, ``chronology``…)."""
        return {
            "basis": self.basis.model_dump(mode="json") if self.basis is not None else None,
            "chronology": self.chronology.model_dump(mode="json") if self.chronology else None,
            "exploratory": bool(self.exploratory),
            "time_ordered_folds": self.order is not None,
        }


def seal_inputs(state: Any, universe: Any, store: Any, task: str | None, *,
                holdout: float, seed: int, structure: Mapping[str, Any] | None = None) -> SealDraw:
    """Read what the seal is drawn from (identifiers, the outcome's classes, time) and decide it.

    ``universe`` is every row the held-out rows are drawn over (every row with the outcome
    measured), in any order; every array returned aligns with it. ``structure`` (the structure
    stage's artifact) supplies the grain stated from a unique identifier when none is answered;
    with neither, nothing is drawn and ``refusal`` says the grain comes first.
    """
    from turbotab.core.stages.working import with_effective_grain

    state, stated = with_effective_grain(state, structure)
    if getattr(state, "grain", None) is None:
        return SealDraw(basis=None, refusal=GRAIN_FIRST)
    ids = np.asarray(universe, dtype=np.int64)
    columns = set(store.columns)
    roles = getattr(state, "roles", None) or {}
    identifiers = [c for c, r in roles.items() if r == "identifier" and c in columns]
    grain = getattr(state, "grain", None)
    named = getattr(grain, "id_column", None) if grain is not None else None
    requested, time_column = temporal_request(state)
    # Classes, or a time-to-event outcome's event, are kept in proportion by the draw.
    classify = task in ("binary", "multiclass", "time_to_event")
    target = getattr(state, "target", None)
    wanted = [*identifiers, *([named] if named in columns else [])]
    if classify and target in columns:
        wanted.append(target)
    if requested and time_column in columns:
        wanted.append(time_column)
    wanted = list(dict.fromkeys(wanted))
    if wanted and len(ids):
        frame = store.materialize(wanted, ids)
    else:
        import pandas as pd

        frame = pd.DataFrame(index=pd.Index(ids, name="row_id"))
    basis, column = decide_basis(state, frame, identifiers, stated=stated)
    draw = SealDraw(basis=basis, read=wanted if len(ids) else [])
    if column is not None:
        values = frame[column].astype(object)
        # A missing identifier is its own unit per row: it cannot be matched to anyone.
        draw.groups = [v if not _isna(v) else f"__missing_{rid}" for rid, v in zip(ids, values)]
        draw.grouped_by = column
    if classify and target in frame.columns:
        draw.y = frame[target].to_numpy(dtype=object)
        draw.labels = draw.y
    if requested:
        draw.chronology, draw.held, draw.refusal, draw.order = _chronology(
            frame, time_column, draw.groups, holdout=holdout, seed=seed, present=time_column in columns)
    return draw


def _chronology(frame: Any, column: str | None, groups: Any | None, *, holdout: float, seed: int,
                present: bool) -> tuple[Chronology, Any | None, str | None, Any | None]:
    """The chronological draw, its disclosure, a refusal when the named column cannot order, and
    each row's unit rank in time for the folds (None when time cannot order them)."""
    if column is None:
        return (Chronology(drawn=False, time_column=None, boundary=None, n_units=None, n_undated=0,
                           sentence="Later outcomes were said to be predicted from earlier ones, but "
                                    "no column was named as time, so the held-out rows were drawn "
                                    "at random rather than by time: treat held-out scores as "
                                    "exploratory."), None, None, None)
    if not present:
        return (Chronology(drawn=False, time_column=column, boundary=None, n_units=None, n_undated=0,
                           sentence=f"`{column}` is named as time but is not in the table."),
                None,
                f"`{column}` was named as the time column, but the table has no column by that "
                f"name, so the latest rows cannot be held out.", None)
    import pandas as pd

    raw = frame[column]
    times = read_times(raw)
    dated = not pd.api.types.is_numeric_dtype(raw) or pd.api.types.is_bool_dtype(raw)
    unreadable = int(np.isnan(times).sum())
    if len(times) and unreadable == len(times):
        return (Chronology(drawn=False, time_column=column, boundary=None, n_units=None,
                           n_undated=0, sentence=f"No value of `{column}` reads as a time."),
                None,
                f"`{column}` was named as the time column, but none of its values reads as a date "
                f"or a number, so the held-out rows cannot be the latest ones.", None)
    if len(times) and unreadable > MAX_UNDATED_SHARE * len(times):
        return (Chronology(drawn=False, time_column=column, boundary=None, n_units=None,
                           n_undated=0, sentence=f"Too many rows have no readable `{column}`."),
                None,
                f"`{unreadable:,}` of `{len(times):,}` rows have no readable `{column}`. Ordering "
                f"units by their last one would place those units arbitrarily, and the seal would "
                f"report a chronology it did not draw.", None)
    mask, chronology = chronological_holdout(times, groups, holdout, seed, column, dated=dated)
    order = time_order(times, groups, seed)
    # Time orders the folds when at least two units have a readable time, held out or not.
    ordered = order if int(np.unique(order[~np.isnan(order)]).size) >= 2 else None
    return chronology, (mask if chronology.drawn else None), None, ordered


def _isna(value: Any) -> bool:
    import pandas as pd

    try:
        return bool(pd.isna(value))  # None, NaN, NaT and pd.NA alike
    except (TypeError, ValueError):
        return False


# ── what a holdout of this size can measure ──────────────────────────────────

Z95 = 1.959964
FLOOR = 100
HOLDOUTS = (0.0, 0.1, 0.2, 0.3)  # the split question's options (teaching content, SPLIT)
USUAL = 0.2  # the holdout the floor is checked at: the usual choice
BINARY_FLOOR_SOURCE = ("Collins, Ogundimu & Altman, Stat Med 2016;35:214–226; Vergouwe et al., "
                       "J Clin Epidemiol 2005;58:475–483")
# Collins et al. resampled Cox models (QRISK2, Cox Framingham): "externally validating a prognostic
# model requires a minimum of 100 events and ideally 200 (or more) events".
TIME_TO_EVENT_FLOOR_SOURCE = "Collins, Ogundimu & Altman, Stat Med 2016;35:214–226"
# The R² a plan assumes, as the AUC's width assumes 0.75: a held-out R²'s precision depends on the
# R² itself, and a plan comes before any fit. 0.2 is a modest prediction R² for a diet outcome.
PLAN_R2 = 0.2
PRECISION_NOTE = ("Widths are approximate 95% intervals of a score on that many held-out rows: "
                  "R² by its large-sample standard error 2·√R²·(1 − R²)/√n at an R² of 0.2, AUC by "
                  "Hanley & McNeil (1982) at an AUC of 0.75, macro-F1 by a proportion's interval "
                  "on the rarest class.")


class SealFloor(_Model):
    """Below this many held-out rows (or events), cross-validation alone is offered first."""

    unit: Literal["rows", "events", "rows in the rarest class"]
    n: int
    text: str
    source: str | None  # where the floor comes from, when the literature gives it
    convention: bool  # True: a stated convention, not a cited rule


class HoldoutOption(_Model):
    holdout: float
    label: str
    n_holdout: int  # about this many rows held out
    measures: str  # what a held-out score on that many rows can tell (≤ 16 words)
    below_floor: bool


class SealPlan(_Model):
    """The ``seal_plan`` artifact: what the split question can offer on this table."""

    task: str
    n_measured: int  # rows with the outcome recorded: the base the seal is drawn over
    n_analyzed: int  # rows in the analysis now: a held-out score is made on the held-out ones
    # The basis the seal would carry if it were drawn now; None until the grain is answered or
    # stated, when ``refusal`` says the grain comes first (M2_CONTRACT §12.2).
    basis: SealBasis | None
    chronology: Chronology | None
    exploratory: bool
    floor: SealFloor
    options: list[HoldoutOption]  # in the order the question offers them
    cv_first: bool
    reason: str  # why that order
    precision_note: str
    refusal: str | None  # why the seal cannot be drawn as the answers stand, if it cannot
    # True when the answers ask for time order and the time column reads: the cross-validation
    # folds then forward-chain by whole unit (each scored by a model fit on earlier units).
    time_ordered_folds: bool = False


def floor_for(task: str | None) -> SealFloor:
    if task == "binary":
        return SealFloor(unit="events", n=FLOOR, convention=False, source=BINARY_FLOOR_SOURCE,
                         text=f"A held-out score needs at least {FLOOR} events and {FLOOR} "
                              f"non-events to be estimated with useful precision.")
    if task == "time_to_event":
        return SealFloor(unit="events", n=FLOOR, convention=False, source=TIME_TO_EVENT_FLOOR_SOURCE,
                         text=f"A held-out score needs at least {FLOOR} events to be estimated "
                              f"with useful precision.")
    if task == "multiclass":
        return SealFloor(unit="rows in the rarest class", n=FLOOR, convention=True, source=None,
                         text=f"A convention: at least {FLOOR} held-out rows in the rarest class, "
                              f"the binary-outcome rule applied to each class.")
    return SealFloor(unit="rows", n=FLOOR, convention=True, source=None,
                     text=f"A convention: at least {FLOOR} held-out rows, by analogy with the "
                          f"{FLOOR}-event rule for binary outcomes.")


def _auc_halfwidth(n1: float, n0: float, auc: float = 0.75) -> float:
    """Hanley & McNeil (1982), Radiology 143:29–36: the SE of an AUC with n1 events, n0 others."""
    if n1 < 1 or n0 < 1:
        return math.inf
    q1, q2 = auc / (2 - auc), 2 * auc * auc / (1 + auc)
    var = (auc * (1 - auc) + (n1 - 1) * (q1 - auc * auc) + (n0 - 1) * (q2 - auc * auc)) / (n1 * n0)
    return Z95 * math.sqrt(max(var, 0.0))


def measure(task: str | None, n: int, counts: Sequence[int] | None = None) -> tuple[str, bool]:
    """What a held-out score on ``n`` rows (with these class counts) can tell, and if below floor."""
    from turbotab.core.voice import tick

    rows = tick(f"{n:,}")
    if n <= 0:
        return ("Every row trains and is scored by cross-validation; no untouched final score.",
                False)
    if task == "binary":
        k = sorted(int(c) for c in (counts or []))
        events, others = (k[0], k[-1]) if len(k) >= 2 else (0, n)
        width = _auc_halfwidth(events, others)
        below = events < FLOOR or others < FLOOR
        what = f"AUC known to about ±{width:.2f}" if width < 0.5 else "too few to measure an AUC"
        return (f"About {rows} held-out rows, {tick(f'{events:,}')} in the rarer class: {what}.",
                below)
    if task == "time_to_event":  # counts: [events, the rest] (:func:`outcome_counts`)
        events = int(counts[0]) if counts else 0
        return (f"About {rows} held-out rows, {tick(f'{events:,}')} with the event: a C-index "
                f"rests on its events.", events < FLOOR)
    if task == "multiclass":
        rarest = min((int(c) for c in counts or []), default=0)
        width = Z95 * math.sqrt(0.25 / rarest) if rarest else math.inf
        what = f"macro-F1 known to about ±{width:.2f}" if width < 0.5 else "too few to measure"
        return (f"About {rows} held-out rows, {tick(f'{rarest:,}')} in the rarest class: {what}.",
                rarest < FLOOR)
    width = Z95 * r2_se(n)
    what = f"R² known to about ±{width:.2f}" if width < 0.5 else "too few to measure an R²"
    return f"About {rows} held-out rows: {what}.", n < FLOOR


def r2_se(n: int, r2: float = PLAN_R2) -> float:
    """The large-sample standard error of an R² measured on ``n`` held-out rows: 2·√R²·(1 − R²)/√n.

    For a model whose predictions are calibrated on new rows, a held-out R² is 1 − A/B with
    A = mean (y − ŷ)² and B = mean (y − ȳ)². By the delta method, with (y − ŷ, y − ȳ) bivariate
    normal and corr² = 1 − R² (the residual is uncorrelated with the prediction),
    Var(1 − A/B) ≈ 4·(1 − R²)²·(1 − corr²)/n = 4·R²·(1 − R²)²/n: the large-sample variance of a
    squared correlation. The formula this replaces, (1 − R²)·√(2/n) evaluated at R² = 0, overstated
    the standard error about twofold at R² = 0.2 (audit A19/E11).
    """
    if n <= 0:
        return math.inf
    return 2.0 * math.sqrt(max(r2, 0.0)) * (1.0 - r2) / math.sqrt(n)


def _label(h: float) -> str:
    return "Cross-validation only" if h == 0 else f"Hold out {h:.0%}"


def holdout_options(task: str | None, n_rows: int,
                    class_counts: Sequence[int] | None = None) -> tuple[list[HoldoutOption], bool, str]:
    """The split question's options in the order to offer them, and why that order.

    Above the floor the usual holdout leads and cross-validation alone comes last; below it,
    cross-validation alone leads, with its reason. Every option stays on offer.
    """
    from turbotab.core.voice import tick

    counts = list(class_counts or [])
    options = []
    for h in HOLDOUTS:
        n = int(round(h * n_rows))
        held_counts = [int(round(h * c)) for c in counts]
        text, below = measure(task, n, held_counts)
        options.append(HoldoutOption(holdout=h, label=_label(h), n_holdout=n, measures=text,
                                     below_floor=bool(h > 0 and below)))
    usual = next(o for o in options if o.holdout == USUAL)
    smallest =min((int(round(USUAL * c)) for c in counts), default=0)
    if task == "time_to_event":
        smallest = int(round(USUAL * counts[0])) if counts else 0
        leaves = f"about {tick(f'{smallest:,}')} held-out events"
        floor_words = f"{FLOOR} events"
    elif task in ("binary", "multiclass"):
        which = "rarer class" if task == "binary" else "rarest class"
        leaves = f"about {tick(f'{smallest:,}')} held-out rows in the {which}"
        floor_words = (f"{FLOOR} events and {FLOOR} non-events" if task == "binary"
                       else f"{FLOOR} in every class")
    else:
        leaves = f"about {tick(f'{usual.n_holdout:,}')} held-out rows"
        floor_words = f"{FLOOR} rows"
    if usual.below_floor:
        reason = (f"Of {tick(f'{n_rows:,}')} rows analyzed, a {USUAL:.0%} holdout leaves {leaves}, "
                  f"below the floor of {floor_words}; cross-validation reuses every row, so it "
                  f"comes first.")
        return [options[0], *options[1:]], True, reason
    reason = (f"A {USUAL:.0%} holdout leaves {leaves}, at or above the floor of {floor_words}; it "
              f"comes first, and cross-validation alone stays on offer.")
    return [usual, *[o for o in options[1:] if o is not usual], options[0]], False, reason


def class_counts(labels: Any) -> list[int]:
    import pandas as pd

    if labels is None:
        return []
    return [int(c) for c in pd.Series(np.asarray(labels, dtype=object)).value_counts(dropna=True)]


def outcome_counts(task: str | None, labels: Any, event: Any = None) -> list[int]:
    """:func:`class_counts`, or for a time-to-event outcome ``[events, the rest]``: the level the
    user named as the event (``set_event``), else 1, as the fit codes it."""
    if task != "time_to_event":
        return class_counts(labels)
    if labels is None:
        return []
    from turbotab.core.stages.rows import _level_key

    keys = [_level_key(v) for v in np.asarray(labels, dtype=object) if not _isna(v)]
    hit = _level_key(event) if event is not None else "1"
    events = sum(k == hit for k in keys)
    return [events, len(keys) - events]


def plan(state: Any, universe: Any, store: Any, task: str | None,
         analyzed: Any | None = None, structure: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The ``seal_plan`` artifact (see :class:`SealPlan`).

    The seal is drawn over ``universe`` (every row with the outcome measured), but a held-out score
    is made on the held-out rows still in the analysis (``analyzed``: the cohort's rows), so what
    each size can measure is counted on those. It answers before the split is chosen, so it never
    reads the split answer: the basis and the chronology it reports are a usual draw's.
    """
    universe = np.asarray(universe, dtype=np.int64)
    analyzed = universe if analyzed is None else np.asarray(analyzed, dtype=np.int64)
    draw = seal_inputs(state, universe, store, task, holdout=USUAL, seed=0, structure=structure)
    counts: list[int] = []
    if task in ("binary", "multiclass", "time_to_event") and getattr(state, "target", None):
        frame = store.materialize([state.target], analyzed)
        counts = outcome_counts(task, frame[state.target].to_numpy(dtype=object),
                                getattr(state, "event", None))
    options, cv_first, reason = holdout_options(task, int(len(analyzed)), counts)
    return SealPlan(
        task=str(task), n_measured=int(len(universe)), n_analyzed=int(len(analyzed)),
        basis=draw.basis, chronology=draw.chronology, exploratory=draw.exploratory,
        floor=floor_for(task), options=options, cv_first=cv_first, reason=reason,
        precision_note=PRECISION_NOTE, refusal=draw.refusal,
        time_ordered_folds=draw.order is not None,
    ).model_dump(mode="json")


# ── withholding the held-out scores ──────────────────────────────────────────


def sealed_scores_frame(models: Sequence[Mapping[str, Any]], scores: Mapping[str, Mapping[str, Any]]) -> Any:
    """The fit Bundle frame holding the held-out scores (``family``, ``metric``, ``value``)."""
    import pandas as pd

    rows = [{"family": key, "metric": metric, "value": (None if value is None else float(value))}
            for key, by_metric in scores.items() if by_metric for metric, value in by_metric.items()]
    return pd.DataFrame(rows, columns=["family", "metric", "value"])


def read_sealed_scores(cache_root: str | Path, key: str) -> dict[str, dict[str, float | None]] | None:
    """The held-out scores a fit artifact holds, read from its sealed frame alone (None: none)."""
    import pandas as pd

    from turbotab.core.graph import artifact_dir

    path = artifact_dir(cache_root, "fit", key) / "frames" / f"{SEALED_SCORES}.parquet"
    if not path.is_file():
        return None
    return scores_by_family(pd.read_parquet(path))


def scores_by_family(frame: Any) -> dict[str, dict[str, float | None]]:
    """``{family: {metric: value}}`` from a sealed-scores frame (a fit Bundle's, or read back)."""
    out: dict[str, dict[str, float | None]] = {}
    for row in frame.itertuples(index=False):
        value = None if row.value is None or (isinstance(row.value, float) and math.isnan(row.value)) \
            else float(row.value)
        out.setdefault(str(row.family), {})[str(row.metric)] = value
    return out


def serve_fit(data: Mapping[str, Any], *, opened: bool,
              scores: Callable[[], Mapping[str, Mapping[str, Any]] | None]) -> dict[str, Any]:
    """The fit artifact as a client may see it now.

    Held-out scores are withheld (``holdout: null``, ``holdout_sealed: true``) until the seal is
    opened, whatever the artifact holds (an older artifact kept them in its data); once opened
    they are the scores the fit computed, unchanged. ``scores()`` reads them from the sealed frame.
    """
    out = copy.deepcopy(dict(data))
    models = out.get("models") or []
    kept = {}
    for m in models:
        kept[m.get("family")] = m.get("holdout")
        m["holdout"] = None
    has_holdout = int(out.get("n_holdout") or 0) > 0
    if not has_holdout:
        out["holdout_sealed"] = False
        return out
    if not opened:
        out["holdout_sealed"] = True
        return out
    sealed = scores() or {}
    for m in models:
        found = sealed.get(m.get("family"))
        m["holdout"] = dict(found) if found else kept.get(m.get("family"))
    out["holdout_sealed"] = False
    return out


# ── when the seal was opened, and what changed since ─────────────────────────


def opening(records: Sequence[Any]) -> Any | None:
    """The record that opened the seal (it is never reverted), or None."""
    for record in sorted(records, key=lambda r: r.seq):
        if record.decision.kind == "open_seal":
            return record
    return None


def state_at_opening(records: Sequence[Any]) -> ProjectState | None:
    opened = opening(records)
    if opened is None:
        return None
    return decisions.fold([r for r in records if r.seq <= opened.seq])


def keys_for(graph: Any, state: ProjectState, fingerprint: str) -> dict[str, str | None]:
    """Every stage's key under ``state``, computed as the engine computes it (``Engine._compute_keys``)."""
    from turbotab.core.graph import stage_key

    values = state.model_dump(mode="json")
    keys: dict[str, str | None] = {}
    missing: dict[str, list[str]] = {}
    for stage in graph.order():
        miss = [slot for slot in stage.requires if values.get(slot) is None]
        for dep in stage.deps:
            miss.extend(missing[dep])
        missing[stage.name] = miss
        if miss:
            keys[stage.name] = None
            continue
        keys[stage.name] = stage_key(stage, {d: keys[d] for d in stage.deps},
                                     {slot: values.get(slot) for slot in stage.reads}, fingerprint)
    return keys


def slots_read_by(graph: Any, stage: str) -> set[str]:
    """Every slot ``stage`` or anything upstream of it reads."""
    names = graph.upstream(stage) | {stage}
    return {slot for name in names for slot in graph[name].reads}


def post_seal_changes(records: Sequence[Any], slots: set[str]) -> list[str]:
    """Ids of the live records made after the opening that touch one of ``slots``."""
    ordered = sorted(records, key=lambda r: r.seq)
    try:
        cancelled = decisions.reverted(ordered)
    except Refusal:
        cancelled = {}
    by_id = {r.id: r for r in ordered}
    out = []
    for record in ordered:
        if not getattr(record, "post_seal", False) or record.id in cancelled:
            continue
        decision = record.decision
        if isinstance(decision, Revert):
            target = by_id.get(decision.decision_id)
            while target is not None and isinstance(target.decision, Revert):
                target = by_id.get(target.decision.decision_id)
            slot = decisions.SLOTS.get(target.decision.kind) if target is not None else None
        else:
            slot = decisions.SLOTS.get(decision.kind)
        if slot in slots:
            out.append(record.id)
    return out


def changed_after_seal(graph: Any, records: Sequence[Any], fingerprint: str, served_key: str | None) -> bool:
    """Whether the fit served (``served_key``) differs from the fit the seal was opened on."""
    then = state_at_opening(records)
    if then is None or served_key is None:
        return False
    return keys_for(graph, then, fingerprint).get("fit") != served_key


def post_seal_sentence(text: str | None, state_before: Any) -> str | None:
    """The record's sentence, marked when it is recorded after the seal was opened."""
    if not text or not getattr(state_before, "seal_opened", False):
        return text
    body = text[0].lower() + text[1:] if text[:1].isupper() and not text[1:2].isupper() else text
    return f"After the held-out rows were opened, {body}"


# ── validators ───────────────────────────────────────────────────────────────


def _ctx(ctx: Any, name: str) -> Any:
    if ctx is None:
        return None
    if isinstance(ctx, Mapping):
        return ctx.get(name)
    return getattr(ctx, name, None)


def _records(ctx: Any) -> list[Any] | None:
    found = _ctx(ctx, "records")
    if callable(found):
        try:
            found = found()
        except Exception:  # noqa: BLE001 - no records: nothing is checked against them
            return None
    return list(found) if found is not None else None


def split_writer(records: Sequence[Any] | None) -> str | None:
    """The id of the live record that drew the current seal (the latest live ``set_split``)."""
    if not records:
        return None
    ordered = sorted(records, key=lambda r: r.seq)
    try:
        cancelled = decisions.reverted(ordered)
    except Refusal:
        return None
    live = [r for r in ordered if r.decision.kind == "set_split" and r.id not in cancelled]
    return live[-1].id if live else None


# Decision A (OPENING_SEQUENCE §01): each changes what a row is, so a seal drawn before it names
# rows that no longer exist.
DECISION_A: dict[str, tuple[str, str]] = {
    "set_orientation": ("orientation", "which way round the table is"),
    "set_grain": ("grain", "whether a unit can appear in more than one row"),
    "set_unit": ("unit", "what one row of the analysis is"),
    "set_aggregation": ("aggregation", "how each unit's rows are combined"),
}
DECISION_A_SLOTS = tuple(slot for slot, _ in DECISION_A.values())


def _cross_validation_only(state: Any) -> bool:
    split = getattr(state, "split", None)
    return split is not None and float(getattr(split, "holdout", 0) or 0) == 0


def _reseal_exit(records: Sequence[Any] | None, cv_only: bool = False) -> dict[str, Any]:
    writer = split_writer(records)
    label = ("Re-draw: withdraw the folds, change this, then draw them again" if cv_only
             else "Re-seal: withdraw the held-out rows, change this, then draw them again")
    return {"label": label, "decision": Revert(decision_id=writer) if writer else None}


def _sealed_refusal(what: str, records: Sequence[Any] | None, state: Any = None) -> Refusal:
    """Decision A's refusal, worded for what is drawn: held-out rows, or the folds alone when every
    row trains under cross-validation (M2_CONTRACT §12.3)."""
    if _cross_validation_only(state):
        return Refusal(
            "sealed",
            f"The cross-validation folds are drawn, and changing {what} changes what a row is: the "
            f"folds name rows as they were. Withdraw the split first, then draw it again.",
            exits=[_reseal_exit(records, cv_only=True)],
        )
    return Refusal(
        "sealed",
        f"The held-out rows are drawn, and changing {what} changes what a row is: the seal names "
        f"rows as they were. Withdraw the seal first, then draw it again.",
        exits=[_reseal_exit(records)],
    )


def _decision_a_waits_for_a_reseal(decision: Any, ctx: Any) -> None:
    state = _ctx(ctx, "state")
    if state is None or getattr(state, "split", None) is None:
        return
    slot, what = DECISION_A[decision.kind]
    value = decisions._SLOT_VALUE[decision.kind](decision)
    if getattr(state, slot, None) == value:
        return  # the same answer again changes nothing
    raise _sealed_refusal(what, _records(ctx), state)


# ── what the draw reads holds still until a re-seal (audit RO-02) ────────────
# The held-out rows are drawn over every row with the outcome measured, stratified by its classes
# when it is categorical, grouped by the column naming the unit when rows repeat, and ordered by
# time when a chronological split is asked for (``seal_inputs``). Changing any of these after the
# draw draws them again: setting three impossible `sbp` values to missing moved 119–130 of 200
# held-out rows into training, and rows whose outcomes had trained the fits already shown became
# "sealed" behind a lock that still read clean (docs/turbotab-next/audit, RO-02). Exclusions and
# the missing-values answer never move a row (``stages.rows.draw_split``). So, like Decision A,
# each such change waits for a re-seal, and the re-seal is the exit. Only a seal that holds rows
# out is guarded: under cross-validation alone nothing is held out.


def holds_rows_out(state: Any) -> bool:
    split = getattr(state, "split", None)
    return split is not None and float(getattr(split, "holdout", 0) or 0) > 0


def draw_columns(state: Any) -> list[str]:
    """The columns whose values the held-out draw reads: the outcome (which rows have it measured,
    and its classes), the column naming the unit when rows repeat, and the time column of a
    chronological draw."""
    out: list[str] = []
    target = getattr(state, "target", None)
    if target:
        out.append(target)
    grain = getattr(state, "grain", None)
    if grain is not None and grain.grain == "repeated":
        if grain.id_column:
            out.append(grain.id_column)
        else:  # ``decide_basis`` then groups by a repeating identifier
            out += [c for c, r in (getattr(state, "roles", None) or {}).items() if r == "identifier"]
    requested, column = temporal_request(state)
    if requested and column:
        out.append(column)
    return list(dict.fromkeys(out))


def _fresh_artifact(ctx: Any, stage: str) -> Any:
    reader = _ctx(ctx, "artifact")
    if not callable(reader):
        return None
    try:
        found = reader(stage)
    except Exception:  # noqa: BLE001 - an unreadable artifact checks nothing
        return None
    return getattr(found, "data", found)


def _task_now(state: Any, ctx: Any) -> str | None:
    """The task the outcome is read as under ``state``: answered, else detected (None: unknown)."""
    if state is None:
        return None
    if state.task is not None:
        return state.task
    info = _fresh_artifact(ctx, "target_info")
    if isinstance(info, Mapping) and state.target is not None and info.get("column") == state.target:
        return info.get("task")
    if state.target is not None and _ctx(ctx, "target") == state.target:
        return _ctx(ctx, "task")
    return None


def _draw_values(state: Any, findings: Any) -> dict[str, str]:
    """The SQL each applied repair puts in place of a column the draw reads (absent: untouched)."""
    from turbotab.core.repairs import column_expressions

    expressions = column_expressions(findings, state)
    return {c: expressions[c] for c in draw_columns(state) if c in expressions}


def _redrawn_by(now: Any, after: Any, ctx: Any) -> str | None:
    """How going from ``now`` to ``after`` would draw the held-out rows again, as a clause; None
    when everything the draw reads stays as it is."""
    target = now.target
    if after.target != target:
        return (f"The held-out rows are drawn over the rows with `{target}` measured, and making "
                f"`{after.target}` the outcome would draw them again over other rows")
    before_task, after_task = _task_now(now, ctx), _task_now(after, ctx)
    if after_task != before_task:
        return (f"The held-out rows are drawn for `{target}` read as "
                f"{before_task or 'it is read now'}, and reading it as "
                f"{after_task or 'something else'} would draw them again")
    (asked, column), (asks, new_column) = temporal_request(now), temporal_request(after)
    if (asked, column) != (asks, new_column):
        if asked and asks:
            return (f"The held-out rows are the latest by `{column}`, and ordering them by "
                    f"`{new_column}` would draw them again")
        if asks:
            return ("The held-out rows are drawn at random, and drawing them by time instead would "
                    "draw them again")
        return (f"The held-out rows are the latest by `{column or 'time'}`, and drawing them at "
                f"random instead would draw them again")
    findings = _fresh_artifact(ctx, "findings")
    before_values, after_values = _draw_values(now, findings), _draw_values(after, findings)
    changed = [c for c in draw_columns(now) if before_values.get(c) != after_values.get(c)]
    if changed:
        return (f"The held-out rows are drawn by `{changed[0]}`, and this changes its values, so "
                f"it could draw them again")
    return None


def _redraw_refusal(clause: str, records: Sequence[Any] | None,
                    extra: Sequence[Mapping[str, Any]] = ()) -> Refusal:
    return Refusal(
        "sealed",
        f"{clause}, moving rows across the seal. Withdraw the seal first, then draw it again.",
        exits=[*extra, _reseal_exit(records)],
    )


def _state_with(state: Any, decision: Any, ctx: Any) -> Any | None:
    """``state`` with ``decision`` recorded, for the kinds this guard reads; None for any other."""
    from turbotab.core.decisions import TemporalSpec

    kind = decision.kind
    if kind == "set_target":
        if decision.column == state.target:
            return None  # the same outcome again: its task answer still stands (``fold``)
        return state.model_copy(update={"target": decision.column, "task": None})
    if kind == "set_task":
        if decision.column != state.target:
            return None  # answers for another column: refused for that reason
        return state.model_copy(update={"task": decision.task})
    if kind == "set_temporal":
        from turbotab.core.sequence import _temporal_names_its_time_column

        done = _temporal_names_its_time_column(decision, ctx)
        return state.model_copy(update={"temporal": TemporalSpec(temporal=done.temporal,
                                                                 time_column=done.time_column)})
    if kind in ("apply_repair", "defer_finding", "dismiss_finding"):
        if kind == "apply_repair":
            from turbotab.core.repairs import _fill_params

            decision = _fill_params(decision, ctx)
        entries = dict(state.findings or {})
        entries[decision.finding_id] = decisions._SLOT_VALUE[kind](decision)
        return state.model_copy(update={"findings": entries})
    return None


def _same_rows_another_way(decision: Any, ctx: Any) -> list[dict[str, Any]]:
    """For a repair that would move rows across the seal, the same finding's options that leave the
    draw alone (impossible outcome values: exclude those rows, which never moves a row)."""
    from turbotab.core.repairs import option_columns

    if decision.kind != "apply_repair":
        return []
    found = None
    for f in _findings(ctx):
        if f.get("id") == decision.finding_id:
            found = f
    reads = set(draw_columns(_ctx(ctx, "state")))
    exits = []
    for option in (found or {}).get("repairs") or []:
        if option.get("key") == decision.option or option.get("effect") != "rows":
            continue
        if not reads & option_columns(decision.finding_id, str(option.get("key")),
                                      (option.get("decision") or {}).get("params") or {}):
            continue
        try:
            decisions.validate(option["decision"], ctx)
        except Refusal:
            continue
        exits.append({"label": f"{option.get('label') or option.get('key')} instead: the held-out "
                               f"rows stay as drawn", "decision": option["decision"]})
    return exits


def _findings(ctx: Any) -> list[dict[str, Any]]:
    from turbotab.core.repairs import findings_of

    return findings_of(_fresh_artifact(ctx, "findings"))


def _the_draw_holds_until_a_reseal(decision: Any, ctx: Any) -> None:
    state = _ctx(ctx, "state")
    if state is None or not holds_rows_out(state):
        return
    after = _state_with(state, decision, ctx)
    if after is None:
        return
    clause = _redrawn_by(state, after, ctx)
    if clause is not None:
        raise _redraw_refusal(clause, _records(ctx), _same_rows_another_way(decision, ctx))


DRAW_READS = ("set_target", "set_task", "set_temporal", "apply_repair", "defer_finding",
              "dismiss_finding")


# ── the repairs to what the draw reads come before it (audit RO-02) ──────────


def _settled(finding: Mapping[str, Any], state: Any, done: Mapping[tuple[str, str], str]) -> bool:
    from turbotab.core.repairs import _covered

    d = (getattr(state, "findings", None) or {}).get(str(finding.get("id")))
    if d is not None and d.action in ("applied", "dismissed"):
        return True
    return _covered(finding, done) is not None


def unsettled_on_the_draw(state: Any, findings: Any) -> list[dict[str, Any]]:
    """Findings whose repair would rewrite a column the draw reads, neither applied, dismissed nor
    done by another finding's repair: the ones to settle before the held-out rows are drawn."""
    from turbotab.core.repairs import _applied, findings_of

    reads = set(draw_columns(state))
    done: dict[tuple[str, str], str] = {}
    for fam, fid, option, params in _applied(state, findings):
        for mark in fam.marks(option, params):
            done.setdefault(mark, fid)
    out = []
    for finding in findings_of(findings):
        touches = any(reads & _option_value_columns(finding, o) for o in finding.get("repairs") or [])
        if touches and not _settled(finding, state, done):
            out.append(dict(finding))
    return out


def _the_draw_reads_settled_values(decision: Any, ctx: Any) -> None:
    """Draw the held-out rows once, on the values the analysis will use: a finding that would
    rewrite the outcome (impossible values, missing-value codes) is settled first."""
    from turbotab.core.decisions import DismissFinding, SplitSpec

    state = _ctx(ctx, "state")
    if state is None or decision.holdout <= 0:
        return
    if state.split == SplitSpec(holdout=decision.holdout, seed=decision.seed, folds=decision.folds):
        return  # the same answer again draws nothing new
    findings = _fresh_artifact(ctx, "findings")
    pending = unsettled_on_the_draw(state, findings) if findings is not None else []
    if not pending:
        return
    finding = pending[0]
    reads = set(draw_columns(state))
    columns = sorted({c for f in pending for o in f.get("repairs") or []
                      for c in reads & set(_option_value_columns(f, o))})
    named = ", ".join(f"`{c}`" for c in columns) or "a column the draw reads"
    what = "A finding" if len(pending) == 1 else f"{len(pending)} findings"
    exits = []
    for option in finding.get("repairs") or []:
        if not reads & _option_columns(finding, option):
            continue  # about another column (a predictor set aside): not what the draw reads
        try:
            decisions.validate(option["decision"], ctx)
        except Refusal:
            continue  # a repair the drawn seal refuses: the re-seal exit below comes first
        exits.append({"label": str(option.get("label") or option.get("key")),
                      "decision": option["decision"]})
    exits.append({"label": "Keep the values as they are",
                  "decision": DismissFinding(finding_id=str(finding["id"]))})
    if holds_rows_out(state):
        exits.append(_reseal_exit(_records(ctx)))
    raise Refusal(
        "settle_first",
        f"{what} about {named} {'is' if len(pending) == 1 else 'are'} not settled, and the held-out "
        f"rows are drawn by {'it' if len(columns) <= 1 else 'them'}: repair or keep the values "
        f"first, so the seal is drawn once, on the values the analysis uses.",
        exits=exits,
    )


def _option_value_columns(finding: Mapping[str, Any], option: Mapping[str, Any]) -> set[str]:
    return _option_columns(finding, option, effects=("values",))


def _option_columns(finding: Mapping[str, Any], option: Mapping[str, Any],
                    effects: Sequence[str] = ("values", "rows", "columns")) -> set[str]:
    from turbotab.core.repairs import option_columns

    return option_columns(str(finding["id"]), str(option.get("key")),
                          (option.get("decision") or {}).get("params") or {}, effects=effects)


def _probe(records: Sequence[Any], decision: Any) -> list[Any]:
    seq = max((r.seq for r in records), default=0) + 1
    probe = decisions.DecisionRecord(id="__probe__", seq=seq, at=datetime.now(timezone.utc),
                                     decision=decision)
    return [*records, probe]


def _revert_keeps_the_seal(decision: Revert, ctx: Any) -> None:
    records = _records(ctx)
    if not records:
        return
    target = next((r for r in records if r.id == decision.decision_id), None)
    if target is None:
        return  # the log refuses an unknown revert itself
    if target.decision.kind == "open_seal":
        raise Refusal(
            "seal_stays_open",
            "The held-out rows were opened and their scores seen; that cannot be undone, so the "
            "opening stays in the record.",
        )
    try:
        now, after = decisions.fold(records), decisions.fold(_probe(records, decision))
    except Refusal:
        return  # the log refuses it with its own reason
    if after.split is None:
        return  # this revert withdraws the seal: the re-seal path
    for kind, (slot, what) in DECISION_A.items():
        if getattr(now, slot) != getattr(after, slot):
            raise _sealed_refusal(what, records, now)
    if holds_rows_out(now) and now.split == after.split:
        clause = _redrawn_by(now, after, ctx)
        if clause is not None:
            raise _redraw_refusal(clause, records)


def _open_seal_once_on_a_fresh_fit(decision: Any, ctx: Any) -> None:
    state = _ctx(ctx, "state")
    if state is None:
        return
    if state.seal_opened:
        raise Refusal(
            "seal_already_open",
            "The held-out rows were opened once already; their scores stand in the record, and a "
            "seal opens only once.",
        )
    spec = state.split
    if spec is None:
        raise Refusal("no_seal", "No rows are held out yet, so there is nothing to open.",
                      exits=[{"label": "Choose how many rows to hold out", "decision": None}])
    hold = [{"label": f"Hold out {USUAL:.0%} first",
             "decision": SetSplit(holdout=USUAL, seed=spec.seed, folds=spec.folds)}]
    if spec.holdout == 0:
        raise Refusal("nothing_sealed",
                      "Every row trains under cross-validation alone, so there are no held-out rows "
                      "to open.", exits=hold)
    artifact = _ctx(ctx, "artifact")
    if not callable(artifact):
        return
    try:
        fit = artifact("fit")
    except Exception:  # noqa: BLE001 - an artifact that cannot be read is not a fresh fit
        fit = None
    if not isinstance(fit, Mapping):
        raise Refusal(
            "fit_not_fresh",
            "The models are not fitted for the current answers yet; the held-out rows open only on "
            "a finished, current fit.",
            exits=[{"label": "Wait for the fit to finish", "decision": None}],
        )
    if not int(fit.get("n_holdout") or 0):
        raise Refusal("nothing_sealed", "This fit holds no rows out, so there is nothing to open.",
                      exits=hold)


decisions.register_validator("open_seal", _open_seal_once_on_a_fresh_fit)
decisions.register_validator("revert", _revert_keeps_the_seal)
for _kind in DECISION_A:  # first: once sealed, the seal is the reason, whatever the data says
    decisions.register_validator(_kind, _decision_a_waits_for_a_reseal, first=True)
for _kind in DRAW_READS:  # after the kind's own checks: a malformed answer says what is wrong first
    decisions.register_validator(_kind, _the_draw_holds_until_a_reseal)
decisions.register_validator("set_split", _the_draw_reads_settled_values)


# ── the open_seal consequence ────────────────────────────────────────────────


def open_seal_views(decision: Any, ctx: Any) -> list[Any]:
    """What opening does: the held-out rows are scored once, and the scores then stand."""
    from turbotab.core.consequences import CAPTION_WORDS, RowFlowView, RowStep, clip_words, fmt_count

    split = ctx.artifact("split")
    data = getattr(split, "data", split)
    if not isinstance(data, Mapping) or not data.get("n_holdout"):
        return []
    n_train, n_hold = int(data["n_train"]), int(data["n_holdout"])
    train = RowStep(key="train", label="Training rows", n=n_train, dropped=n_hold,
                    reason="held out for the final check")
    ctx.read["note"] = ("Opening shows each model's score on the held-out rows, once. A later "
                        "change still recomputes, and the Record marks it as made after the opening.")
    return [RowFlowView(
        title="The held-out rows, scored once",
        caption=clip_words(f"The {fmt_count(n_hold)} held-out rows are scored once; the scores then "
                           f"stand in the record.", CAPTION_WORDS),
        emphasis=["holdout"],
        before=[train, RowStep(key="holdout", label="Held-out rows: sealed", n=n_hold)],
        after=[train, RowStep(key="holdout", label="Held-out rows: scored once", n=n_hold)],
    )]


def _register_preview() -> None:
    from turbotab.core.consequences import register_consequence

    register_consequence("open_seal", open_seal_views)


_register_preview()

__all__ = [
    "BasisState", "Chronology", "DECISION_A", "GRAIN_FIRST", "HoldoutOption", "SEALED_SCORES",
    "SealBasis",
    "SealDraw", "SealFloor", "SealPlan", "changed_after_seal", "chronological_holdout",
    "decide_basis", "draw_columns", "floor_for", "holdout_options", "holds_rows_out", "keys_for",
    "measure", "open_seal_views", "r2_se", "time_order",
    "opening", "plan", "post_seal_changes", "post_seal_sentence", "read_sealed_scores",
    "read_times", "seal_inputs", "sealed_scores_frame", "serve_fit", "slots_read_by",
    "split_writer", "state_at_opening", "unsettled_on_the_draw",
]

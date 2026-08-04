"""turbotab.features — the transform catalogue, split by clause 06's litmus.

Lockbox constitution §06 gives an automatable test and two dispositions:

    Does this transform's output for row *i* depend on any other row?

* **No — structural repair.** Row-local, deterministic, label-free. Zero
  leakage pathway, so it **executes immediately** on the working table and
  posts a receipt.
* **Yes — statistical transform.** It learns from a distribution. **Recorded as
  a decision now and executed inside per-model pipelines fit on training folds
  only.** Materializing one on the working table pre-split is the canonical
  preprocessing leak.

**The router defaults to deferral when unsure.**

So the classification lives on the catalogue entry rather than in the code that
applies it. A transform cannot be executed without its `scope` being read,
because `apply` refuses anything that is not `ROW_LOCAL` — the litmus is a
precondition, not a convention.

Two entries genuinely split rather than resolve, and they are the interesting
ones:

* **Binning** depends on where the edges come from. Fixed cut-points the user
  supplies are row-local. Quantile and k-means edges are learned from the
  column's distribution — computed over the full table they have seen the
  sealed rows. *Uniform* is stateful too, and more subtly: its min and max come
  from the data.
* **Ordinal encoding** depends on where the ORDER comes from. A declared order
  (`mild < moderate < severe`) is row-local; one derived from frequency is not.

Neither is resolved by picking a side. Each is two entries, and the deferral is
the default of the pair.

Nothing here imports scikit-learn. `turbotab/requirements.txt` states that the
engine path needs pandas and numpy and nothing else, and the row-local half of
this catalogue is arithmetic. The deferred half does not execute here at all —
it produces a *spec* that a per-model pipeline consumes, which is precisely
what clause §06 asks for and is also why this module stays dependency-free.
"""
from __future__ import annotations

import math
import string
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# The two dispositions, named so a reader of a spec can see which one applies
# without knowing the catalogue.
ROW_LOCAL = "row_local"
STATEFUL = "stateful"


class FeatureRefusal(Exception):
    """The catalogue was asked for something it cannot honestly do."""


# ── the refusals `_compute` raises, hoisted so there is ONE copy ─────────────
#
# `GUIDED-198`. These two sentences are the reason the app cannot derive the
# parameter, stated in the engine's own words, and the interface has to be able
# to SHOW that reason beside the control it renders. Hoisting them out of
# `_compute` is what makes "quote it" possible without a second copy drifting
# from the first — the alternative is a `because` on the descriptor that says
# roughly the same thing, which is this project's most-repeated defect one layer
# over.
#
# The text is unchanged from where it was written, and `COPY_DECK.md` carries
# both with `source="turbotab/features.py"`.
EDGES_REFUSAL = (
    "Binning by supplied cut-points needs at least two edges. "
    "Without them the edges would have to come from the data, "
    "which is a different transform and defers.")
ORDER_REFUSAL = (
    "Encoding in a stated order needs the order. Deriving it from "
    "the data is a different transform and defers.")

#: The most levels an order can be STATED over, and it is a routing answer
#: rather than a cap. An order is supplied one level at a time — that is what
#: makes it a statement of the researcher's knowledge rather than a reading of
#: the data — and past this many the control is a list nobody fills by hand. The
#: transform that derives the order from the data already exists and is named in
#: the refusal, so the column is still reachable; what is refused is pretending
#: an unfillable control is an offer.
ORDER_MAX_LEVELS = 12


@dataclass(frozen=True)
class Parameter:
    """One value a transform needs, the user supplies, and the app cannot derive.

    **`GUIDED-198`, and the finding is what `needs` USED to be.** It was a tuple
    of NAMES — `("n_bins",)` — and a name is not renderable. The page read
    `n_inputs` and never `needs`, so six of eighteen transforms offered a button
    whose only possible answer was a 400, on every fixture, for the life of the
    step.

    A name would have forced the page to invent the control: *"n_bins is a whole
    number from 2 to 10"* written in `index.html` is a second copy of a rule that
    lives here, which is the defect this codebase repeats most. So the SERVER
    describes the parameter and the page renders what it is told:

    * `kind` — `integer`, `numbers` or `levels`. The only vocabulary the page
      knows, and each maps to one control.
    * `label` — what goes beside the control.
    * `because` — **why the app cannot derive it**, and for `edges` and `order`
      this is `_compute`'s own `FeatureRefusal`, quoted rather than rewritten.
      A control that appears without its reason is a demand; one that carries it
      is a question the researcher can answer.
    * `minimum` / `min_items` — the bound, HERE and enforced here
      (`_check_params`). A bound the interface renders and nothing keeps is a
      rule the app states falsely.
    * `from_column` — the legitimate values are the CHOSEN COLUMN's own distinct
      levels, so they are unknowable until a column is picked. `order` is the
      only one, and it is the one that tests whether "state the precondition"
      generalizes: `/features` serves the levels, so the precondition is met
      rather than announced.
    """

    name: str
    kind: str                                # integer | numbers | levels
    label: str
    because: str
    minimum: Optional[float] = None          # `integer`: the smallest value
    min_items: Optional[int] = None          # `numbers`/`levels`: fewest entries
    hint: str = ""
    from_column: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """What an interface can RENDER. `min_items` deliberately stays here.

        `minimum` goes out because the page puts it on the control as `min` and
        the browser keeps it. `min_items` has no such rendering — the count it
        requires is already stated to the user inside `hint` and `because`, and
        it is enforced in `_check_params`. Shipping it anyway would be a field
        the server composes and nothing reads, which is §07.1 at payload
        granularity and exactly what `fieldsweep` exists to find.
        """
        return {"name": self.name, "kind": self.kind, "label": self.label,
                "because": self.because, "minimum": self.minimum,
                "hint": self.hint, "from_column": self.from_column}


#: Keyed by parameter NAME rather than by transform, because three transforms
#: need `n_bins` and one rule for it is the point. A `needs` entry with no row
#: here raises at import through `_PARAMETER_FOR`, so a transform cannot ship a
#: parameter the interface has no way to render.
PARAMETERS: Dict[str, Parameter] = {p.name: p for p in [
    Parameter(
        "edges", "numbers", "The cut-points, lowest first",
        EDGES_REFUSAL, min_items=2,
        hint="Two or more numbers, separated by commas, in increasing order."),
    Parameter(
        "order", "levels", "The order, lowest first",
        ORDER_REFUSAL, min_items=2, from_column=True),
    Parameter(
        "n_bins", "integer", "How many bins",
        # `_sentence`'s own argument, not a new one: dropping the token gives
        # "grouped into equal-sized bins", which reads as THE APP CHOSE FOR YOU
        # — and `pipeline_plan` then fits four, so the silence would sit
        # directly on top of an undisclosed default.
        "How many groups to make is a question about the resolution you want, "
        "not a property of the column. If it is left unstated the pipeline "
        "fits four, and a number the app picked silently is a decision you "
        "did not make.",
        minimum=2),
    Parameter(
        "n_components", "integer", "How many components",
        # Also `_sentence`'s: for `pca` the missing value is the sentence's
        # grammatical SUBJECT.
        "How many components to keep is the methods sentence's subject: "
        "without it the line reads \"principal components will be computed\", "
        "which asserts a number nobody stated. If it is left unstated the "
        "pipeline fits two.",
        minimum=1),
]}


def _PARAMETER_FOR(name: str) -> Parameter:
    param = PARAMETERS.get(name)
    if param is None:                                    # pragma: no cover
        raise FeatureRefusal(
            f"'{name}' is declared in a transform's `needs` and has no "
            f"descriptor, so no interface can render a control for it. "
            f"Known: {', '.join(sorted(PARAMETERS))}.")
    return param


@dataclass(frozen=True)
class Transform:
    """One catalogue entry, carrying its own clause-§06 classification.

    `scope` is not documentation. `apply()` refuses to execute anything that is
    not `ROW_LOCAL`, so a stateful transform cannot be materialized on the
    working table even by a caller that means well.
    """

    key: str
    label: str
    scope: str                              # ROW_LOCAL | STATEFUL
    # Why this scope, in the terms of the litmus. Held on the entry so the
    # interface can show the reasoning rather than assert the classification.
    because: str
    # The methods-prose sentence. For a deferred transform this carries the
    # TIMING, which is simultaneously the receipt, the schedule and the
    # manuscript line: "will be selected within each training fold".
    sentence: str
    # How hard this makes the model to explain. Classic carries this in every
    # tab's guidance expander and a rebuild loses it by default.
    explainability_cost: str = "low"        # low | medium | high
    n_inputs: int = 1                       # columns the user must name
    needs: Sequence[str] = ()               # extra parameters, by name
    _fn: Optional[Callable[..., pd.Series]] = None

    @property
    def defers(self) -> bool:
        return self.scope != ROW_LOCAL

    @property
    def parameters(self) -> List[Parameter]:
        """`needs`, resolved to the descriptors an interface can render.

        `needs` stays the names — `_unfilled` compares against them and so do
        three tests — and this is the same list one level up. One source, two
        readings, rather than two lists to drift.
        """
        return [_PARAMETER_FOR(n) for n in self.needs]

    def to_dict(self) -> Dict[str, Any]:
        # `needs` GOES OUT AS DESCRIPTORS. On the wire it was a list of bare
        # names that nothing read; a name cannot be rendered, which is why
        # nothing read it (`GUIDED-198`). Trap #7 in reverse — the structured
        # form was lossier than the reason sitting beside it in this module.
        return {"key": self.key, "label": self.label, "scope": self.scope,
                "because": self.because, "sentence": self.sentence,
                "defers": self.defers, "n_inputs": self.n_inputs,
                "needs": [p.to_dict() for p in self.parameters],
                "explainability_cost": self.explainability_cost}


# ── row-local: arithmetic on one row ─────────────────────────────────────────

def _log(s: pd.Series) -> pd.Series:
    return np.log(s.where(s > 0))


def _log1p(s: pd.Series) -> pd.Series:
    return np.log1p(s.where(s > -1))


def _sqrt(s: pd.Series) -> pd.Series:
    return np.sqrt(s.where(s >= 0))


def _inverse(s: pd.Series) -> pd.Series:
    return 1.0 / s.where(s != 0)


_ROW_LOCAL_WHY = (
    "Row-local: the value computed for a row uses only that row's own cells, "
    "so it cannot carry information from any other row — including the "
    "held-out ones.")

_STATEFUL_WHY_SUFFIX = (
    " Computing it over the whole table would fit it on the held-out rows too, "
    "which is the canonical preprocessing leak — so it is recorded now and "
    "fitted inside each training fold.")


CATALOGUE: Dict[str, Transform] = {t.key: t for t in [
    # ── row-local ────────────────────────────────────────────────────────────
    Transform("log", "log(x)", ROW_LOCAL, _ROW_LOCAL_WHY,
              "`log({a})` was computed from `{a}` directly; values at or below "
              "zero are undefined and become missing.",
              explainability_cost="low", _fn=_log),
    Transform("log1p", "log(x + 1)", ROW_LOCAL, _ROW_LOCAL_WHY,
              "`log1p({a})` was computed from `{a}` directly, which is defined "
              "at zero.",
              explainability_cost="low", _fn=_log1p),
    Transform("sqrt", "sqrt(x)", ROW_LOCAL, _ROW_LOCAL_WHY,
              "`sqrt({a})` was computed from `{a}` directly; negative values "
              "are undefined and become missing.",
              explainability_cost="low", _fn=_sqrt),
    Transform("square", "x squared", ROW_LOCAL, _ROW_LOCAL_WHY,
              "`{a}` squared was computed from `{a}` directly.",
              explainability_cost="medium", _fn=lambda s: s ** 2),
    Transform("cube", "x cubed", ROW_LOCAL, _ROW_LOCAL_WHY,
              "`{a}` cubed was computed from `{a}` directly.",
              explainability_cost="medium", _fn=lambda s: s ** 3),
    Transform("inverse", "1 / x", ROW_LOCAL, _ROW_LOCAL_WHY,
              "`1/{a}` was computed from `{a}` directly; zeros are undefined "
              "and become missing.",
              explainability_cost="medium", _fn=_inverse),
    Transform("ratio", "A / B", ROW_LOCAL, _ROW_LOCAL_WHY,
              "The ratio `{a} / {b}` was computed row by row; rows where `{b}` "
              "is zero are undefined and become missing.",
              explainability_cost="low", n_inputs=2,
              _fn=lambda a, b: a / b.where(b != 0)),
    Transform("product", "A x B (interaction)", ROW_LOCAL, _ROW_LOCAL_WHY,
              "The interaction `{a} x {b}` was computed row by row.",
              explainability_cost="high", n_inputs=2,
              _fn=lambda a, b: a * b),
    Transform("difference", "A - B", ROW_LOCAL, _ROW_LOCAL_WHY,
              "The difference `{a} - {b}` was computed row by row.",
              explainability_cost="low", n_inputs=2,
              _fn=lambda a, b: a - b),
    Transform("missing_indicator", "Is this value missing?", ROW_LOCAL,
              _ROW_LOCAL_WHY + " Whether a cell is blank is a fact about that "
              "cell, not about the column's distribution.",
              "A binary indicator was added recording whether `{a}` was "
              "missing, so a model can use the fact of the blank as signal.",
              explainability_cost="low",
              _fn=lambda s: s.isna().astype("int8")),
    Transform("bin_fixed", "Bin by cut-points I supply", ROW_LOCAL,
              _ROW_LOCAL_WHY + " The edges come from the user, not from the "
              "data, so no other row is consulted.",
              "`{a}` was grouped into bins at cut-points {edges}, which were "
              "specified rather than derived from the data.",
              explainability_cost="low", needs=("edges",)),
    Transform("ordinal_declared", "Encode categories in an order I state",
              ROW_LOCAL,
              _ROW_LOCAL_WHY + " The order comes from the user's knowledge of "
              "the variable, not from the data's shape.",
              "`{a}` was encoded in the order {order}, which was stated rather "
              "than inferred.",
              explainability_cost="low", needs=("order",)),

    # ── stateful: recorded now, fitted in-fold ───────────────────────────────
    Transform("bin_quantile", "Bin into equal-sized groups (quantiles)",
              STATEFUL,
              "The bin edges are quantiles of the column, so every row's bin "
              "depends on where the other rows fall." + _STATEFUL_WHY_SUFFIX,
              "`{a}` will be grouped into {n_bins} equal-sized bins, with the "
              "cut-points computed within each training fold.",
              explainability_cost="medium", needs=("n_bins",)),
    Transform("bin_uniform", "Bin into equal-width groups", STATEFUL,
              "The edges are spaced between the column's minimum and maximum, "
              "and both come from the data — so an extreme value in any row "
              "moves every other row's bin." + _STATEFUL_WHY_SUFFIX,
              "`{a}` will be grouped into {n_bins} equal-width bins, with the "
              "range computed within each training fold.",
              explainability_cost="medium", needs=("n_bins",)),
    Transform("bin_kmeans", "Bin by clustering the values", STATEFUL,
              "The cluster centres are fitted to the column's whole "
              "distribution." + _STATEFUL_WHY_SUFFIX,
              "`{a}` will be grouped into {n_bins} clustered bins, fitted "
              "within each training fold.",
              explainability_cost="high", needs=("n_bins",)),
    Transform("ordinal_frequency", "Encode categories by how common they are",
              STATEFUL,
              "The order is derived from counts across the whole column, so "
              "one row's code depends on every other row." + _STATEFUL_WHY_SUFFIX,
              "`{a}` will be encoded by category frequency, computed within "
              "each training fold.",
              explainability_cost="medium"),
    Transform("standardize", "Center and scale", STATEFUL,
              "The mean and standard deviation are properties of the "
              "column." + _STATEFUL_WHY_SUFFIX,
              "`{a}` will be centered and scaled using the mean and standard "
              "deviation of each training fold.",
              explainability_cost="low"),
    Transform("pca", "Principal components", STATEFUL,
              "Components are fitted to the covariance of the whole table, so "
              "every component encodes every row." + _STATEFUL_WHY_SUFFIX,
              "{n_components} principal components will be computed, fitted "
              "within each training fold.",
              explainability_cost="high", needs=("n_components",)),
]}


def column_levels(df: pd.DataFrame,
                  exclude: Sequence[str] = ()) -> List[Dict[str, Any]]:
    """Each column's distinct values, or the reason an order cannot be stated.

    **This is `GUIDED-198`'s hard half.** `ordinal_declared` needs an `order`,
    and the legitimate values of an order are the CHOSEN COLUMN's own levels —
    unknowable until a column is picked, which is why the parameter had no
    control and the transform 400'd on every press. The row's brief gives two
    admissible answers: serve the levels once a column is known, or state the
    precondition and leave it refusing. **This serves the levels**, because a
    control that states a precondition nothing can satisfy has moved the defect
    rather than fixed it, and the levels are one `unique()` away.

    **Every column appears, and that is the point.** The shelf is never
    shortened: the picker used to offer `ordinal_declared` the NUMERIC columns
    only — which is both too narrow (a category is usually text) and too wide (a
    496-level identifier has no statable order), and it could not succeed on
    either. So every column gets a row, and a row is one of two things:

    * `levels` — the distinct values, sorted, and an order can be stated.
    * `refusal` — the sentence saying why not, naming the count and, where the
      count is the problem, the transform that derives the order from the data
      instead. A routing answer rather than an omission (`_NOT_OFFERED`'s rule).

    Sorted by their string form and NOT by a guessed semantic order: the whole
    premise of `ordinal_declared` is that the app does not know the order. An
    alphabetical list is a list; a list the app arranged would be an assertion.

    Missing values are excluded — a blank is not a level, and `_compute`'s
    `.map()` leaves an unlisted value as missing either way.
    """
    skip = {str(c) for c in exclude}
    out: List[Dict[str, Any]] = []
    for column in df.columns:
        name = str(column)
        if name in skip:
            continue
        present = df[column].dropna()
        distinct = sorted({str(v) for v in present.unique()})
        row: Dict[str, Any] = {"column": name, "n_levels": len(distinct)}
        if len(distinct) < 2:
            row["refusal"] = (
                f"`{name}` has {len(distinct)} distinct "
                f"{'value' if len(distinct) == 1 else 'values'} in this table, "
                f"so there is no order to state. An order is a statement about "
                f"how two levels rank, and there are not two levels.")
        elif len(distinct) > ORDER_MAX_LEVELS:
            row["refusal"] = (
                f"`{name}` has {len(distinct)} distinct values. An order is "
                f"stated one level at a time, and stating {len(distinct)} of "
                f"them by hand is not a thing this control can honestly ask "
                f"for. If the ranking you want is by how common each value is, "
                f"that is 'Encode categories by how common they are', which "
                f"derives it from the data and is recorded as a deferred "
                f"decision.")
        else:
            row["levels"] = distinct
        out.append(row)
    return out


def row_local_keys() -> List[str]:
    return [k for k, t in CATALOGUE.items() if not t.defers]


def deferred_keys() -> List[str]:
    return [k for k, t in CATALOGUE.items() if t.defers]


# ── deliberately not offered, with the routing answer instead ────────────────
# A gap that becomes routing is worth more than a transform. `feat-polynomial`
# is `classic-only` in the register on two arguments, and a user who reaches for
# it deserves both of them plus somewhere to go — not "unknown key", which reads
# as an omission and teaches nothing.
#
# Keyed on what a caller might ASK for, which makes this a routing table and not
# a detector: it fires on these spellings and no others. That is acceptable here
# and would not be for a contradiction check, because the keys arrive from an
# interface offering the catalogue rather than from free text.
_NOT_OFFERED: Dict[str, str] = {
    "polynomial": (
        "Generating a whole polynomial basis is not offered here, and the "
        "reason is a routing answer rather than a missing feature.\n\n"
        "Two arguments, and they are different. First: degree 2 over ten "
        "numeric columns produces 55 new terms — 10 squares and 45 pairwise "
        "products — that nobody chose one at a time, each carrying "
        "explainability cost. Mass generation is the opposite of this "
        "interview's premise. Second: on a 140-row study those 55 terms are "
        "p/n ≈ 0.39, which is the overfitting regime; the expansion is most "
        "attractive on exactly the small studies where it does the most harm.\n\n"
        "If your question really is about interactions, the route is a model "
        "that captures them rather than columns that manufacture them. Trees "
        "and gradient boosting get interactions for free, so this is a model "
        "choice at the modeling step, not a feature choice here.\n\n"
        "If you want ONE interaction because you already reason about it "
        "clinically, that is what `product`, `ratio` and `difference` are — "
        "named, chosen, and each posting its own receipt."),
}
# Spellings that route to the same answer. Not a detector; see above.
_NOT_OFFERED_ALIASES: Dict[str, str] = {
    "poly": "polynomial",
    "polynomial_features": "polynomial",
    "polynomialfeatures": "polynomial",
    "interactions": "polynomial",
    "all_interactions": "polynomial",
}


def not_offered(key: str) -> Optional[str]:
    """The routing answer for a capability this door declines to build.

    `None` when the key is simply unknown. Separate from the catalogue lookup
    so an interface can ask "is there guidance for this?" without provoking an
    exception it then has to catch.
    """
    canonical = _NOT_OFFERED_ALIASES.get(str(key).lower(), str(key).lower())
    return _NOT_OFFERED.get(canonical)


def get(key: str) -> Transform:
    t = CATALOGUE.get(key)
    if t is None:
        routed = not_offered(key)
        if routed:
            raise FeatureRefusal(routed)
        raise FeatureRefusal(
            f"'{key}' is not in the transform catalogue. Known: "
            f"{', '.join(sorted(CATALOGUE))}.")
    return t


def classify(key: str) -> str:
    """The litmus answer for one transform. The one place that decides."""
    return get(key).scope


def new_column_name(key: str, columns: Sequence[str],
                    params: Optional[Dict[str, Any]] = None) -> str:
    """A name a researcher would recognize, and one that does not collide."""
    params = params or {}
    a = str(columns[0]) if columns else "x"
    b = str(columns[1]) if len(columns) > 1 else ""
    return {
        "log": f"log_{a}", "log1p": f"log1p_{a}", "sqrt": f"sqrt_{a}",
        "square": f"{a}_squared", "cube": f"{a}_cubed", "inverse": f"inv_{a}",
        "ratio": f"{a}_per_{b}", "product": f"{a}_x_{b}",
        "difference": f"{a}_minus_{b}",
        "missing_indicator": f"{a}_is_missing",
        "bin_fixed": f"{a}_binned",
        "ordinal_declared": f"{a}_ordinal",
    }.get(key, f"{a}_{key}")


def preview(df: pd.DataFrame, key: str, columns: Sequence[str],
            params: Optional[Dict[str, Any]] = None,
            n: int = 6) -> Dict[str, Any]:
    """Compute the transform on a COPY and describe it. Never persists.

    A CHOICE gets a before/after preview (`DESIGN_LANGUAGE.md` §09), and the
    preview must be the real computation rather than a description of one —
    otherwise it is a claim about what would happen, which is the thing this
    project keeps finding to be wrong.

    **`GUIDED-171`: EVERY OPERAND, not the first one.** This used to compute
    `before = df[columns[0]]`, so a two-column formula previewed as one:
    `ratio` on `weight_kg` and `height_cm` returned `before: 95.8, after:
    0.5682` and never showed the 168.6 the division used. The `after` was
    right and unexplainable — a before/after table whose *before* is missing
    half of what the *after* was computed from is not a preview of that
    transform, and it is the surface where the user consents to it.

    The payload said nothing at all about the second column, either: the
    `sentence` read *"The ratio `weight_kg / height_cm` was computed row by
    row"* while the structured form beside it named no columns whatsoever.
    Trap #7 — the machine-readable form lossier than the sentence — so
    `inputs` is the columns the computation consumed, in the order it consumed
    them, and `operands` is each row's value for each of them.

    `before` stays, and stays the first operand. It is `operands[0]` by
    construction rather than a second computation of the same value, and it is
    kept because it has shipped consumers — the shelf is never shortened, and a
    field removed from a payload is a reader broken somewhere the removal
    cannot see.
    """
    t = get(key)
    params = dict(params or {})
    _check_params(t, params)
    _require_columns(df, columns, t)

    if t.defers:
        # Clause §06 permits exactly one override: a read-only preview NOT
        # persisted to the modeling table, labeled "preview, not applied". It
        # is computed on TRAINING ROWS ONLY, because a preview fitted on the
        # whole column would show the user a picture of their held-out data.
        return _deferred_preview(df, t, columns, params, n)

    out = _compute(df, t, columns, params)
    # `columns[:t.n_inputs]` is `_compute`'s own slice, not a second opinion
    # about which columns this transform reads. A preview naming operands the
    # computation did not use would be the same defect pointing the other way.
    inputs = [str(c) for c in columns[:t.n_inputs]]
    operands = [df[c] for c in inputs]
    shown = operands[0].head(n).index
    return {
        "key": key, "scope": t.scope, "applied": False,
        "new_column": new_column_name(key, columns, params),
        "sentence": _sentence(t, columns, params),
        "explainability_cost": t.explainability_cost,
        "because": t.because,
        "inputs": inputs,
        "rows": [{"label": _plain(i),
                  "operands": [_plain(s.loc[i]) for s in operands],
                  "before": _plain(operands[0].loc[i]),
                  "after": _plain(out.loc[i])} for i in shown],
        "n_undefined": int(out.isna().sum() - df[list(columns)].isna().any(axis=1).sum()),
        "n_rows": int(len(out)),
    }


def apply(df: pd.DataFrame, key: str, columns: Sequence[str],
          params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute a ROW-LOCAL transform on the working table, and post a receipt.

    **Refuses anything stateful.** That refusal is clause §06 made executable:
    a caller cannot materialize a distribution-dependent transform pre-split
    even by asking for it directly, so the litmus is a precondition rather than
    a convention somebody follows.
    """
    t = get(key)
    if t.defers:
        raise FeatureRefusal(
            f"'{t.label}' learns from the column's distribution, so applying it "
            f"to the working table now would fit it on the held-out rows too. "
            f"It is recorded as a decision and fitted inside each training "
            f"fold instead. {t.because}")
    params = dict(params or {})
    _check_params(t, params)
    _require_columns(df, columns, t)

    name = new_column_name(key, columns, params)
    if name in df.columns:
        raise FeatureRefusal(
            f"'{name}' already exists in this table. Remove it first, or the "
            f"new column would silently replace it.")

    out = df.copy()
    out[name] = _compute(df, t, columns, params)
    return {
        "frame": out,
        "receipt": {
            "key": key, "scope": ROW_LOCAL, "column": name,
            "inputs": [str(c) for c in columns],
            "sentence": _sentence(t, columns, params),
            "explainability_cost": t.explainability_cost,
            "n_undefined": int(out[name].isna().sum()),
        },
    }


def declare(key: str, columns: Sequence[str],
            params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Record a STATEFUL transform as a decision, without executing it.

    Returns the spec a per-model pipeline consumes. The sentence carries the
    timing as methods prose, which is simultaneously the receipt, the schedule
    and the manuscript line.
    """
    t = get(key)
    if not t.defers:
        raise FeatureRefusal(
            f"'{t.label}' is row-local, so it executes immediately rather than "
            f"being declared. Use apply().")
    params = dict(params or {})
    _check_params(t, params)
    return {
        "key": key, "scope": STATEFUL, "columns": [str(c) for c in columns],
        "params": params,
        "sentence": _sentence(t, columns, params),
        "because": t.because,
        "explainability_cost": t.explainability_cost,
        "fit_on": "training folds only",
    }


# ── internals ────────────────────────────────────────────────────────────────

def _check_params(t: Transform, params: Dict[str, Any]) -> None:
    """Every SUPPLIED parameter against the descriptor the catalogue publishes.

    **Absence is deliberately not checked here.** `_compute` refuses a missing
    `edges` and a missing `order`, and `_sentence` refuses a missing `n_bins` or
    `n_components` — in the words `GUIDED-175` settled and `COPY_DECK.md`
    records. Duplicating that here would give one condition two sentences.

    What this is for is the other half, and it only became reachable at
    `GUIDED-198`: **once the page can send a parameter, it can send a wrong
    one.** `Parameter` publishes `minimum`, `min_items` and — for `edges` —
    an order the values have to be in, and a bound an interface renders while
    nothing enforces it is a rule the app states falsely.

    `pd.cut` is the sharp case rather than a hypothetical: `bins=[10, 5]` raises
    a bare `ValueError` that no handler in `api.py` catches, so a user typing
    two cut-points backwards would have got a 500 with a traceback where a
    sentence belongs.
    """
    for name in t.needs:
        param = _PARAMETER_FOR(name)
        if name not in params or params[name] is None:
            continue                                    # absence, refused above
        value = params[name]

        if param.kind == "integer":
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise FeatureRefusal(
                    f"`{param.name}` has to be a whole number and "
                    f"{value!r} is not one. {param.because}") from None
            if number != int(number):
                raise FeatureRefusal(
                    f"`{param.name}` has to be a whole number; {value!r} is "
                    f"not. {param.because}")
            if param.minimum is not None and number < param.minimum:
                raise FeatureRefusal(
                    f"`{param.name}` has to be at least "
                    f"{int(param.minimum)}; {int(number)} was supplied. "
                    f"{param.because}")
            continue

        if not isinstance(value, (list, tuple)):
            raise FeatureRefusal(
                f"`{param.name}` is a list of values and {value!r} is not a "
                f"list. {param.because}")
        if param.min_items is not None and len(value) < param.min_items:
            raise FeatureRefusal(
                f"`{param.name}` needs at least {param.min_items} entries and "
                f"{len(value)} were supplied. {param.because}")

        if param.kind == "numbers":
            for entry in value:
                try:
                    float(entry)
                except (TypeError, ValueError):
                    raise FeatureRefusal(
                        f"`{param.name}` is a list of numbers and {entry!r} is "
                        f"not one. {param.because}") from None
            ordered = [float(e) for e in value]
            if any(b <= a for a, b in zip(ordered, ordered[1:])):
                raise FeatureRefusal(
                    f"`{param.name}` has to increase, and {list(value)} does "
                    f"not. Cut-points out of order describe no bins at all, so "
                    f"they are refused rather than reordered — the order you "
                    f"typed is part of what you are stating.")
            continue

        # `levels`. A repeated level is the one that returns a WRONG value
        # rather than refusing: `_compute` builds `{str(v): i}`, so a duplicate
        # silently keeps the last position and every row at the level it
        # displaced is encoded as the wrong rank.
        seen = [str(v) for v in value]
        if len(set(seen)) != len(seen):
            repeated = sorted({v for v in seen if seen.count(v) > 1})
            raise FeatureRefusal(
                f"`{param.name}` names {', '.join(repr(r) for r in repeated)} "
                f"more than once. Each level holds one position in an order, "
                f"and a repeat would encode every row at the displaced level "
                f"with the wrong rank.")


def _require_columns(df: pd.DataFrame, columns: Sequence[str],
                     t: Transform) -> None:
    if len(columns) < t.n_inputs:
        raise FeatureRefusal(
            f"'{t.label}' needs {t.n_inputs} column(s); got {len(columns)}.")
    for c in columns[:t.n_inputs]:
        if c not in df.columns:
            raise FeatureRefusal(f"No column named '{c}' in this table.")


def _compute(df: pd.DataFrame, t: Transform, columns: Sequence[str],
             params: Dict[str, Any]) -> pd.Series:
    if t.key == "bin_fixed":
        edges = params.get("edges")
        if not edges or len(edges) < 2:
            raise FeatureRefusal(EDGES_REFUSAL)
        return pd.cut(df[columns[0]], bins=list(edges),
                      labels=False, include_lowest=True)
    if t.key == "ordinal_declared":
        order = params.get("order")
        if not order:
            raise FeatureRefusal(ORDER_REFUSAL)
        lookup = {str(v): i for i, v in enumerate(order)}
        return df[columns[0]].astype(str).map(lookup).astype("float64")
    args = [df[c] for c in columns[:t.n_inputs]]
    if t._fn is None:                                    # pragma: no cover
        raise FeatureRefusal(f"'{t.key}' has no implementation.")
    return t._fn(*args)


_ABSENT = object()


def _unfilled(template: str, fields: Dict[str, Any]) -> List[str]:
    """The placeholders this template needs and `fields` cannot honestly fill.

    In template order, deduplicated. Read with the same parser `str.format`
    uses rather than with a regex, so `{{` escapes and any future `{n:.1f}`
    format spec stay the formatter's business instead of drifting away from a
    hand-written pattern.

    A field supplied as `None` or as blank text counts as unfilled: rendering
    it produces `The ratio `weight` /  was computed`, which is the same defect
    with the braces removed.
    """
    out: List[str] = []
    for _, name, _, _ in string.Formatter().parse(template):
        if not name:                    # literal text, or a positional `{}`
            continue
        key = name.split(".")[0].split("[")[0]
        value = fields.get(key, _ABSENT)
        if (value is _ABSENT or value is None
                or (isinstance(value, str) and not value.strip())):
            if key not in out:
                out.append(key)
    return out


def _sentence(t: Transform, columns: Sequence[str],
              params: Dict[str, Any]) -> str:
    """The methods sentence, fully substituted — or a refusal. Never the template.

    `GUIDED-175`. This was:

        try: return t.sentence.format(**fields)
        except KeyError: return t.sentence

    so a transform whose parameter had not been supplied shipped its TEMPLATE.
    The product owner drove it and read, on screen, *"`{a}` will be grouped
    into {n_bins} clustered bins."* — and that string is the decision sentence,
    which is the manuscript's methods line at a different level of formality.

    **The class is a fourth branch nobody authorized.** The governing rule
    gives three: the app may assert truly, it may be silent, and it may refuse.
    Template syntax on screen is none of them — not false, not silent, not a
    refusal, but noise where a sentence was promised. The old handler is also
    the project's silent-degradation shape: a `except: return something of the
    right type` that no test notices because a `str` came back.

    **The second half, which is easy to miss:** a `KeyError` on `n_bins`
    discarded the substitution of `{a}` as well, so the column the user HAD
    chosen was thrown away with the parameter they had not. Whatever this
    function does when it cannot compose the sentence, it does not lose what it
    knows — the refusal below names the column.

    **Option (a), refuse, is what this does**, and the refusal states which
    parameter is outstanding. Three reasons over the alternatives:

    * The refusal is the branch the rule already authorizes, and it is already
      wired at every one of the four call sites — `preview` and `apply` and
      `declare` all raise `FeatureRefusal` into handlers that exist
      (`api.preview_feature`, `project.defer_feature`, `api._decision`). No new
      sentence kind has to be rendered anywhere.
    * This module already answers a missing parameter this way for the OTHER
      half of the same split pair: `_compute` refuses `bin_fixed` without its
      `edges` and `ordinal_declared` without its `order`. The deferred half
      returning a template was the inconsistency, not the refusal.
    * A sentence is a decision's own claim about itself. `declare` writes one
      into the record, so any composed-anyway sentence becomes a decision the
      user never made and has to be un-made later.

    **(b), drop the clause that needs the value, was rejected as impossible for
    one entry and untrue for three.** For `pca` the missing `{n_components}` is
    the sentence's grammatical SUBJECT — dropping it leaves *"principal
    components will be computed"*, which asserts an unstated number. For the
    three binning entries dropping the token gives *"grouped into equal-sized
    bins"*, which reads as *the app chose for you* — and `pipeline_plan` then
    fits `n_bins=4`, so the silence would sit directly on top of an undisclosed
    default.

    **(c), return a sentence naming the outstanding parameter, was rejected as
    the RETURN value and kept as the refusal's content.** The returned string is
    recorded and exported; a methods line reading "the number of bins is
    outstanding" is a decision recorded before it was made. The same words are
    honest as a refusal, which is why they are in the message below.
    """
    fields = {"a": str(columns[0]) if columns else "x",
              "b": str(columns[1]) if len(columns) > 1 else "",
              **{k: v for k, v in params.items()}}
    outstanding = _unfilled(t.sentence, fields)
    if outstanding:
        named = ", ".join(f"`{k}`" for k in outstanding)
        on = (" on " + ", ".join(f"`{c}`" for c in columns)) if columns else ""
        raise FeatureRefusal(
            f"'{t.label}'{on} cannot be described yet: "
            f"{named} {'has' if len(outstanding) == 1 else 'have'} not been "
            f"supplied. The sentence this step writes is the methods sentence, "
            f"so it is refused rather than shipped with the parameter still "
            f"unfilled. Supply {named} and it reads in full.")
    return t.sentence.format(**fields)


def _deferred_preview(df: pd.DataFrame, t: Transform, columns: Sequence[str],
                      params: Dict[str, Any], n: int) -> Dict[str, Any]:
    """The one permitted override, labeled and scoped.

    Clause §06: *"a read-only preview not persisted to the modeling table is
    the only permitted override, and it is labeled 'preview, not applied'."*
    Computed on TRAINING ROWS only — a preview fitted on the whole column would
    be showing the researcher a picture of their own held-out data, which is
    the leak this transform defers to avoid, arriving through the preview
    instead.
    """
    return {
        "key": t.key, "scope": STATEFUL, "applied": False,
        "preview_not_applied": True,
        "new_column": None,
        "sentence": _sentence(t, columns, params),
        "because": t.because,
        "explainability_cost": t.explainability_cost,
        "fit_on": "training folds only",
        "rows": [],
        "note": ("Not computed here. This transform learns from the column's "
                 "distribution, so it is fitted inside each training fold at "
                 "modeling time — there is no single set of values to show "
                 "before then."),
    }


def _plain(v: Any) -> Any:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if math.isnan(float(v)) else round(float(v), 4)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v if isinstance(v, (int, float, bool, str)) else str(v)

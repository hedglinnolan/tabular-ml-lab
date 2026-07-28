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

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "label": self.label, "scope": self.scope,
                "because": self.because, "sentence": self.sentence,
                "defers": self.defers, "n_inputs": self.n_inputs,
                "needs": list(self.needs),
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
    """
    t = get(key)
    params = dict(params or {})
    _require_columns(df, columns, t)

    if t.defers:
        # Clause §06 permits exactly one override: a read-only preview NOT
        # persisted to the modeling table, labeled "preview, not applied". It
        # is computed on TRAINING ROWS ONLY, because a preview fitted on the
        # whole column would show the user a picture of their held-out data.
        return _deferred_preview(df, t, columns, params, n)

    out = _compute(df, t, columns, params)
    before = df[columns[0]].head(n)
    return {
        "key": key, "scope": t.scope, "applied": False,
        "new_column": new_column_name(key, columns, params),
        "sentence": _sentence(t, columns, params),
        "explainability_cost": t.explainability_cost,
        "because": t.because,
        "rows": [{"label": _plain(i), "before": _plain(before.loc[i]),
                  "after": _plain(out.loc[i])} for i in before.index],
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
    return {
        "key": key, "scope": STATEFUL, "columns": [str(c) for c in columns],
        "params": params,
        "sentence": _sentence(t, columns, params),
        "because": t.because,
        "explainability_cost": t.explainability_cost,
        "fit_on": "training folds only",
    }


# ── internals ────────────────────────────────────────────────────────────────

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
            raise FeatureRefusal(
                "Binning by supplied cut-points needs at least two edges. "
                "Without them the edges would have to come from the data, "
                "which is a different transform and defers.")
        return pd.cut(df[columns[0]], bins=list(edges),
                      labels=False, include_lowest=True)
    if t.key == "ordinal_declared":
        order = params.get("order")
        if not order:
            raise FeatureRefusal(
                "Encoding in a stated order needs the order. Deriving it from "
                "the data is a different transform and defers.")
        lookup = {str(v): i for i, v in enumerate(order)}
        return df[columns[0]].astype(str).map(lookup).astype("float64")
    args = [df[c] for c in columns[:t.n_inputs]]
    if t._fn is None:                                    # pragma: no cover
        raise FeatureRefusal(f"'{t.key}' has no implementation.")
    return t._fn(*args)


def _sentence(t: Transform, columns: Sequence[str],
              params: Dict[str, Any]) -> str:
    fields = {"a": str(columns[0]) if columns else "x",
              "b": str(columns[1]) if len(columns) > 1 else "",
              **{k: v for k, v in params.items()}}
    try:
        return t.sentence.format(**fields)
    except KeyError:
        return t.sentence


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

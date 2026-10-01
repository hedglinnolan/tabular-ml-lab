"""Nested nutrients: columns that are parts of another column (``fat_sat`` ⊂ ``fat_total``).

A dietary export often carries a macronutrient's total beside its components: saturated,
monounsaturated and polyunsaturated fat beside total fat, sugars beside carbohydrate. Moving energy
through the total while its parts stay fixed is not a coherent substitution (the parts would no
longer fit inside the total), and pairing a total with its own part moves nothing at all.

A column is read as nested in another when both hold:

* **the names say so** — the child names a subtype of the parent's macronutrient (``sat``,
  ``mufa``, ``poly``, ``sugar``, ``starch``, ``animal``…) and the parent names that macronutrient
  with no subtype (``fat_total``, ``total_fat``, ``carb``);
* **the data agree** — the child is at most the parent on at least :data:`MIN_INSIDE` of the rows
  where both are recorded.

Fiber is deliberately not read as part of carbohydrate: food tables disagree on whether total
carbohydrate includes it.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import numpy as np

MIN_INSIDE = 0.99  # share of rows on which a part may not exceed its total
MIN_ROWS = 10

# Subtype words by macronutrient. A child's name must carry one; a parent's must carry none.
SUBTYPES: dict[str, frozenset[str]] = {
    "fat": frozenset({"sat", "saturated", "sfa", "mon", "mono", "monounsaturated", "mufa", "poly",
                      "polyunsaturated", "pufa", "trans", "tfa"}),
    "carbohydrate": frozenset({"sugar", "sugars", "starch", "sucrose", "fructose", "lactose"}),
    "protein": frozenset({"animal", "plant", "vegetable", "dairy"}),
}
# Words that name a subtype of one macronutrient on their own (``sfa_g``, ``sugar``).
SPECIFIC: dict[str, frozenset[str]] = {
    "fat": frozenset({"sfa", "mufa", "pufa", "saturated", "monounsaturated", "polyunsaturated"}),
    "carbohydrate": frozenset({"sugar", "sugars", "starch", "sucrose", "fructose", "lactose"}),
    "protein": frozenset(),
}


def _tokens(name: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", str(name).lower()) if t}


def _role(column: str) -> str | None:
    from turbotab.core.methods.energy import nutrient_role

    try:
        return nutrient_role(column)
    except ValueError:
        return None


def _macro_of_child(column: str) -> str | None:
    """The macronutrient ``column`` names a subtype of, from its name alone."""
    tokens = _tokens(column)
    role = _role(column)
    for macro, words in SUBTYPES.items():
        if tokens & SPECIFIC[macro] or (role == macro and tokens & words):
            return macro
    return None


def _unit_class(column: str) -> str | None:
    from turbotab.core.methods.energy import unit_of

    unit = unit_of(column)
    if unit == "density":
        return None  # a share of energy is not an amount a total can hold
    return "grams" if unit in ("grams", "unmarked") else unit


def candidates(columns: Sequence[str]) -> dict[str, list[str]]:
    """Parent -> the children its name admits, from names alone (the data have not spoken yet)."""
    parents: dict[str, list[str]] = {}
    for c in columns:
        role = _role(c)
        if role in SUBTYPES and not _tokens(c) & SUBTYPES[role] and _unit_class(c) is not None:
            parents.setdefault(role, []).append(c)
    out: dict[str, list[str]] = {}
    for macro, options in parents.items():
        parent = sorted(options, key=lambda c: ("total" not in _tokens(c), columns.index(c)))[0]
        children = [c for c in columns if c != parent and _macro_of_child(c) == macro
                    and _unit_class(c) == _unit_class(parent)]
        if children:
            out[parent] = children
    return out


def nested_components(frame: Any, columns: Sequence[str] | None = None) -> dict[str, str]:
    """Child -> parent for every name-admitted pair the data confirm, over ``frame``'s rows."""
    import pandas as pd

    columns = [str(c) for c in (frame.columns if columns is None else columns) if c in frame.columns]
    out: dict[str, str] = {}
    for parent, children in candidates(columns).items():
        p = pd.to_numeric(frame[parent], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        for child in children:
            c = pd.to_numeric(frame[child], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
            both = np.isfinite(p) & np.isfinite(c)
            if both.sum() < MIN_ROWS:
                continue
            inside = c[both] <= p[both] * (1 + 1e-9) + 1e-9
            if float(inside.mean()) >= MIN_INSIDE:
                out[child] = parent
    return out


def parts_of(nested: Mapping[str, str]) -> dict[str, list[str]]:
    """Parent -> its children, in the order they were found."""
    out: dict[str, list[str]] = {}
    for child, parent in nested.items():
        out.setdefault(parent, []).append(child)
    return out


SHARE_TOTALS = (100.0, 1.0)  # percent, or a proportion
SHARE_TOLERANCE = 0.005  # relative: 100 ± 0.5


def _is_share(column: str) -> bool:
    from turbotab.core.methods.energy import unit_of

    name = str(column).lower()
    return unit_of(column) == "density" and ("pct" in name or "percent" in name)


def compositions(frame: Any, columns: Sequence[str] | None = None) -> list[str]:
    """Energy shares that sum to a fixed total (100%) on every row, or [] when none do.

    Shares of energy (``protein_pct_kcal`` …) that add up to 100% are one fewer free quantity
    than columns: beside an intercept, any one is fixed by the others, so a linear model cannot
    estimate them all. The usual remedy leaves one out as the reference.
    """
    import pandas as pd

    names = [str(c) for c in (frame.columns if columns is None else columns)
             if c in frame.columns and _is_share(str(c))]
    if len(names) < 2:
        return []
    values = frame[names].apply(pd.to_numeric, errors="coerce").dropna()
    if len(values) < MIN_ROWS:
        return []
    total = values.sum(axis=1).to_numpy(dtype=float)
    for fixed in SHARE_TOTALS:
        if float((np.abs(total - fixed) <= SHARE_TOLERANCE * fixed).mean()) >= MIN_INSIDE:
            return names
    return []


def reference_share(frame: Any, shares: Sequence[str]) -> str:
    """The share to leave out: the largest on average, so the others read as replacing it."""
    import pandas as pd

    means = frame[list(shares)].apply(pd.to_numeric, errors="coerce").mean()
    return str(means.idxmax())


__all__ = ["MIN_INSIDE", "SUBTYPES", "candidates", "compositions", "nested_components", "parts_of",
           "reference_share"]

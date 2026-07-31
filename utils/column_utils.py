"""
Column name utilities for data processing.
"""
from typing import List, Any, Optional

import pandas as pd


def color_by_category(
    series: "pd.Series",
    is_classification_target: bool = False,
    max_discrete_levels: int = 10,
) -> bool:
    """Whether a column should get one color per value rather than a colorbar.

    Deciding this from dtype alone is the trap: a classification target coded
    0/1 is numeric, so a dtype check hands it a continuous colorbar and the two
    classes come out as two shades of the same ramp instead of two colors.

    Args:
        series: the column being used for color.
        is_classification_target: True when this column is the modeling target
            and the task is classification. Decisive on its own — an integer
            class code is categorical no matter how many levels it has.
        max_discrete_levels: above this many distinct values, a numeric column
            is treated as continuous even if every value is a whole number.
    """
    if is_classification_target:
        return True
    if not pd.api.types.is_numeric_dtype(series):
        return True
    values = series.dropna()
    if values.empty:
        return False
    # Whole numbers over a small range are codes, not measurements.
    return values.nunique() <= max_discrete_levels and bool((values % 1 == 0).all())


def make_unique_columns(cols: List[Any]) -> List[str]:
    """Deduplicate column names by appending _1, _2, etc. for duplicates."""
    seen = {}
    result = []
    for c in cols:
        c_str = str(c)
        if c_str in seen:
            seen[c_str] += 1
            result.append(f"{c_str}_{seen[c_str]}")
        else:
            seen[c_str] = 0
            result.append(c_str)
    return result

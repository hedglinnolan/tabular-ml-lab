"""A family's inner cross-validation, grouped when the split is grouped (M2_CONTRACT §3).

Elastic net tunes its penalty by an inner cross-validation over the rows it is fit on
(``ElasticNetCV`` / ``LogisticRegressionCV``). With repeated rows per unit, plain inner folds put
one unit's rows on both sides of an inner split: the penalty is then tuned on rows that share a
person with the rows scoring it, which favors too little shrinkage. The outer folds and the seal are
grouped already; this makes the inner folds grouped the same way.

Generic on purpose: any model step with a ``cv`` parameter gets grouped inner splits, so a later
family with its own inner search is covered without a switch on family keys.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _missing(value: Any) -> bool:
    import pandas as pd

    try:
        return bool(pd.isna(value))  # None, NaN, NaT, pd.NA
    except (TypeError, ValueError):
        return False


def inner_splits(groups: Any, n_splits: int, seed: int = 0) -> list[tuple[Any, Any]] | None:
    """Grouped (train, test) position splits over these rows; None when there are too few units."""
    from sklearn.model_selection import GroupKFold

    # A missing identifier is its own unit, as the split treats it: it cannot be matched to anyone.
    labels = np.asarray([f"__missing_{i}" if _missing(g) else str(g)
                         for i, g in enumerate(np.asarray(groups, dtype=object))], dtype=object)
    units = len(np.unique(labels))
    if units < 2:
        return None
    k = max(2, min(int(n_splits), units))
    folds = GroupKFold(n_splits=k, shuffle=True, random_state=seed)
    return [(train, test) for train, test in folds.split(np.zeros(len(labels)), groups=labels)]


def with_grouped_inner_cv(pipeline: Any, groups: Any | None, seed: int = 0) -> Any:
    """``pipeline`` with its model step's inner cross-validation grouped by ``groups``.

    ``groups`` aligns with the rows the pipeline is about to be fit on. Nothing changes when there
    are no groups or the model step has no inner cross-validation. Returns the pipeline.
    """
    if groups is None:
        return pipeline
    name, model = pipeline.steps[-1]
    params = model.get_params(deep=False)
    if "cv" not in params:
        return pipeline
    current = params["cv"]
    if isinstance(current, int):
        n_splits = current
    elif isinstance(current, (list, tuple)):
        n_splits = len(current)
    else:
        n_splits = int(getattr(current, "n_splits", 5) or 5)
    splits = inner_splits(groups, n_splits, seed)
    if splits is not None:
        pipeline.set_params(**{f"{name}__cv": splits})
    return pipeline


__all__ = ["inner_splits", "with_grouped_inner_cv"]

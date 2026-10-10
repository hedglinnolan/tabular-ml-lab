"""Every split a model makes inside its own fit, drawn as the outer folds are (M2_CONTRACT §3).

Two families split the rows they are fit on: elastic net tunes its penalty by an inner
cross-validation (``ElasticNetCV`` / ``LogisticRegressionCV``), and boosted trees on more than
10,000 rows hold out a tenth of them to stop early. Left to scikit-learn, both split by position:

* **Repeated rows.** One person's rows land on both sides of an inner split, and the penalty is
  tuned (or the trees stopped) on rows that share a person with the rows scoring it.
* **Time** (audit MA-11). Inner folds shuffle time even when the outer folds respect it.
* **File order** (audit A15). An unshuffled inner ``KFold`` cuts the rows into contiguous runs, so
  a file sorted by the outcome tunes the penalty on folds that each hold one slice of its range:
  0.0299 on a random-order file against 0.0004 on the same rows sorted by the outcome.

:func:`fit_pipeline` fits a pipeline with its inner splits drawn the way the outer ones are: by
whole unit when rows repeat; forward-chaining by whole unit when the folds are time-ordered
(:mod:`turbotab.core.models.folds`); otherwise shuffled by a seed over a key computed from each
row's contents, so the same rows give the same splits in any order. A classification outcome keeps
its classes in proportion where the splitter allows. Boosted trees get the same treatment for their
early-stopping rows (A16): a tenth of the units — the latest ones when time-ordered — passed as
``X_val``.

Generic on purpose: any model step with a ``cv`` parameter, or with ``early_stopping`` and
``validation_fraction``, is covered without a switch on family keys.
"""
from __future__ import annotations

from typing import Any

import numpy as np

EARLY_STOPPING_ROWS = 10_000  # scikit-learn's own threshold for early_stopping="auto"
HASH_COLUMNS = 64  # the columns a row's key is computed from (with the outcome): cheap at any width


def _missing(value: Any) -> bool:
    import pandas as pd

    try:
        return bool(pd.isna(value))  # None, NaN, NaT, pd.NA
    except (TypeError, ValueError):
        return False


def unit_labels(groups: Any) -> np.ndarray:
    """Group labels as strings; a missing identifier is its own unit, as the split treats it."""
    return np.asarray([f"__missing_{i}" if _missing(g) else str(g)
                       for i, g in enumerate(np.asarray(groups, dtype=object))], dtype=object)


def _single_precision(column: Any) -> Any:
    """A floating-point column in single precision; any other column as it is."""
    import pandas as pd

    if not pd.api.types.is_float_dtype(column.dtype):
        return column
    with np.errstate(over="ignore"):  # beyond single precision's range: ±inf, one key
        return column.astype("Float32" if isinstance(column.dtype, pd.api.extensions.ExtensionDtype)
                             else np.float32)


def row_keys(X: Any, y: Any = None) -> np.ndarray:
    """A key per row from its contents (up to :data:`HASH_COLUMNS` columns and the outcome).

    Identical rows share a key, so a bootstrap resample's copies of one row never sit on both sides
    of a split. Rows that differ in those columns or the outcome get different keys (barring a
    64-bit hash collision); rows that differ only beyond them share one and stay in one fold, which
    costs nothing. The key never depends on the order the rows arrive in.

    **Numbers are hashed in single precision**, so the key, and every split drawn from it, is the
    same on every platform. A value computed on two platforms can differ in its last bit (another
    summation order, another libm, a fused multiply-add): the WP7 prediction fixture's simulated
    values do between macOS and Linux, and hashed exactly, every key and every inner fold differed,
    so the elastic net tuned its penalty on other rows. In single precision such a value keeps its
    key unless it sits within a bit of a rounding boundary (about one value in 500 million). Rows
    equal to single precision, about seven significant digits, share a key and stay in one fold.
    """
    import pandas as pd

    frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
    frame = frame.iloc[:, :HASH_COLUMNS].reset_index(drop=True)
    frame = frame.apply(_single_precision)
    keys = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64)
    if y is not None:
        values = np.asarray(y)
        target = (_single_precision(pd.Series(values)) if values.dtype.kind == "f"
                  else pd.Series(np.asarray(y, dtype=object)).astype(str))
        with np.errstate(over="ignore"):  # unsigned arithmetic wraps, as a hash should
            keys = keys * np.uint64(1_000_003) ^ pd.util.hash_pandas_object(
                target.reset_index(drop=True), index=False).to_numpy(dtype=np.uint64)
    return keys


def _canonical(keys: np.ndarray) -> np.ndarray:
    """Positions sorted by key (stable): the order every seeded splitter below works in."""
    return np.argsort(keys, kind="stable")


def _unit_label_of(keys: np.ndarray, y: Any) -> dict[str, str]:
    """Each unit's most common class (for stratifying whole units)."""
    import pandas as pd

    frame = pd.DataFrame({"unit": keys, "y": pd.Series(np.asarray(y, dtype=object)).astype(str)})
    return frame.groupby("unit", sort=True)["y"].agg(lambda s: s.mode().iloc[0]).to_dict()


def inner_splits(groups: Any = None, n_splits: int = 5, seed: int = 0, *, keys: Any = None,
                 order: Any = None, y: Any = None) -> list[tuple[Any, Any]] | None:
    """(train, test) position splits over these rows; None when there are too few units.

    The unit is ``groups`` when given, else ``keys`` (one per row, :func:`row_keys`), else the
    row's position. With ``order`` the splits forward-chain over whole units in time
    (:func:`~turbotab.core.models.folds.forward_blocks`): ``n_splits`` scored folds over
    ``n_splits + 1`` blocks. Otherwise whole units are shuffled into ``n_splits`` folds by ``seed``,
    stratified by ``y`` when it is given and every class has enough units.
    """
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    from turbotab.core.models.folds import forward_blocks, forward_pairs

    if groups is not None:
        labels = unit_labels(groups)
    elif keys is not None:
        labels = np.asarray(keys, dtype=object)
    else:
        n = len(order) if order is not None else (len(y) if y is not None else 0)
        labels = np.asarray([f"{i:012d}" for i in range(n)], dtype=object)
    units = len(np.unique(labels))
    if units < 2:
        return None
    if order is not None:
        blocks, made, _ = forward_blocks(order, labels, int(n_splits) + 1, labels=y)
        pairs = forward_pairs(blocks)
        return pairs if made >= 2 and pairs else None
    k = max(2, min(int(n_splits), units))
    canon = _canonical(labels)
    sorted_labels = labels[canon]
    idx = np.arange(len(labels))
    parts = None
    if y is not None:
        unit_y = _unit_label_of(labels, y)
        counts = np.unique(np.asarray(list(unit_y.values()), dtype=object), return_counts=True)[1]
        if len(counts) >= 2 and counts.min() >= k:
            y_sorted = np.asarray(y, dtype=object)[canon].astype(str)
            parts = StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(
                idx, y_sorted, groups=sorted_labels)
    if parts is None:
        parts = GroupKFold(k, shuffle=True, random_state=seed).split(idx, groups=sorted_labels)
    return [(np.sort(canon[train]), np.sort(canon[test])) for train, test in parts]


def _n_splits(current: Any) -> int:
    if isinstance(current, int):
        return current
    if isinstance(current, (list, tuple)):
        return len(current)
    return int(getattr(current, "n_splits", 5) or 5)


def with_inner_cv(pipeline: Any, *, groups: Any = None, keys: Any = None, order: Any = None,
                  y: Any = None, seed: int = 0) -> Any:
    """``pipeline`` with its model step's inner cross-validation drawn by :func:`inner_splits`.

    Everything aligns with the rows the pipeline is about to be fit on. Nothing changes when the
    model step has no ``cv`` parameter. Returns the pipeline.
    """
    name, model = pipeline.steps[-1]
    params = model.get_params(deep=False)
    if "cv" not in params:
        return pipeline
    from sklearn.base import is_classifier

    splits = inner_splits(groups, _n_splits(params["cv"]), seed, keys=keys, order=order,
                          y=y if is_classifier(model) else None)
    if splits is not None:
        pipeline.set_params(**{f"{name}__cv": splits})
    return pipeline


def _steps_with_cv(pipeline: Any) -> list[tuple[str, Any]]:
    """The steps before the model that split their own rows (EXPLORE's inner-CV form choice and
    the selection step): those with a ``cv`` parameter."""
    out = []
    for name, step in getattr(pipeline, "steps", [])[:-1]:
        try:
            params = step.get_params(deep=False)
        except Exception:  # noqa: BLE001 - a step without parameters splits nothing
            continue
        if "cv" in params:
            out.append((name, step))
    return out


def with_step_cv(pipeline: Any, *, groups: Any = None, keys: Any = None, order: Any = None,
                 y: Any = None, seed: int = 0) -> Any:
    """Each earlier step's inner cross-validation drawn by :func:`inner_splits`, as the model's is
    (Wave 2, EXPLORE): a step sees every row the pipeline is fit on, so the positions align. A step
    whose splits cannot be drawn (too few units) keeps its own count of folds."""
    for name, step in _steps_with_cv(pipeline):
        current = step.get_params(deep=False)["cv"]
        splits = inner_splits(groups, _n_splits(current), seed, keys=keys, order=order, y=y)
        pipeline.set_params(**{f"{name}__cv": splits if splits is not None else _n_splits(current)})
    return pipeline


def with_grouped_inner_cv(pipeline: Any, groups: Any | None, seed: int = 0) -> Any:
    """The inner cross-validation grouped by ``groups`` (unchanged without groups).

    Kept for callers that only know the groups; :func:`fit_pipeline` is the complete path.
    """
    if groups is None:
        return pipeline
    return with_inner_cv(pipeline, groups=groups, seed=seed)


def _stops_early(model: Any, n_rows: int) -> bool:
    params = model.get_params(deep=False)
    if "early_stopping" not in params or "validation_fraction" not in params:
        return False
    if params.get("validation_fraction") is None:
        return False
    flag = params["early_stopping"]
    return bool(flag is True or (flag == "auto" and n_rows > EARLY_STOPPING_ROWS))


def validation_rows(share: float, *, groups: Any = None, keys: Any = None, order: Any = None,
                    y: Any = None, seed: int = 0) -> np.ndarray:
    """A boolean mask: about ``share`` of the units, held aside for early stopping.

    The latest units when ``order`` is given; else whole units drawn by ``seed`` (stratified by
    ``y`` when every class has two units or more), in an order fixed by the units' labels.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    from turbotab.core.models.folds import UNDATED

    if groups is not None:
        labels = unit_labels(groups)
    elif keys is not None:
        labels = np.asarray(keys, dtype=object)
    else:
        n = len(order) if order is not None else len(y)
        labels = np.asarray([f"{i:012d}" for i in range(n)], dtype=object)
    names = np.unique(labels)
    n_val = max(1, int(round(share * len(names))))
    if order is not None:
        o = np.asarray(order, dtype=float)
        last = pd.Series(np.where(np.isnan(o), UNDATED, o)).groupby(labels).max()
        ranked = pd.DataFrame({"unit": last.index.to_numpy(), "order": last.to_numpy()})
        ranked = ranked.sort_values(["order", "unit"], kind="mergesort")
        return np.isin(labels, ranked["unit"].to_numpy()[-n_val:])
    stratify = None
    if y is not None:
        unit_y = _unit_label_of(labels, y)
        strata = np.asarray([unit_y[u] for u in names], dtype=object)
        counts = np.unique(strata, return_counts=True)[1]
        if len(counts) >= 2 and counts.min() >= 2 and n_val >= len(counts):
            stratify = strata
    _, chosen = train_test_split(names, test_size=n_val, random_state=seed, stratify=stratify)
    return np.isin(labels, chosen)


def _take(data: Any, mask: np.ndarray) -> Any:
    import pandas as pd

    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.iloc[np.flatnonzero(mask)]
    return np.asarray(data)[mask]


def fit_pipeline(pipeline: Any, X: Any, y: Any, *, groups: Any = None, order: Any = None,
                 seed: int = 0) -> Any:
    """Fit ``pipeline`` on ``X``, ``y`` with every inner split drawn as the outer folds are.

    ``groups`` (the unit per row) and ``order`` (each row's unit rank in time, from the split) align
    with ``X``. The model step's inner cross-validation gets :func:`inner_splits`; a model that
    stops early on part of its rows (boosted trees above 10,000 rows) is handed
    :func:`validation_rows` as ``X_val`` instead of drawing its own by position, and the steps before
    it are fit on the other rows only (RECIPES F11). Returns the fitted pipeline.
    """
    from sklearn.base import is_classifier

    from turbotab.core.models.metrics import survival_baseline

    y_arr = np.asarray(y)
    model = pipeline.steps[-1][1]
    splits = ("cv" in model.get_params(deep=False) or _stops_early(model, len(y_arr))
              or bool(_steps_with_cv(pipeline)))
    keys = row_keys(X, y_arr) if splits and groups is None and order is None else None
    if not _stops_early(model, len(y_arr)):
        with_inner_cv(pipeline, groups=groups, keys=keys, order=order, y=y_arr, seed=seed)
        with_step_cv(pipeline, groups=groups, keys=keys, order=order, y=y_arr, seed=seed)
        # A time to event keeps the baseline hazard of the rows it was fit on (MS6), so it predicts
        # a risk by the horizon wherever it is scored.
        return survival_baseline(pipeline.fit(X, y), X, y_arr)
    # RECIPES F11 (§4.3): the stopping units are drawn first, and the steps before the model, their
    # inner splits included, are fit on the remaining rows only; the stopping rows are transformed
    # by those fitted steps, so a step that reads the outcome never sees them.
    share = float(model.get_params(deep=False)["validation_fraction"])
    held = validation_rows(share, groups=groups, keys=keys, order=order,
                           y=y_arr if is_classifier(model) else None, seed=seed)
    rest = ~held

    def kept(a: Any) -> Any:
        return None if a is None else np.asarray(a)[rest]

    with_inner_cv(pipeline, groups=kept(groups), keys=kept(keys), order=kept(order),
                  y=y_arr[rest], seed=seed)
    with_step_cv(pipeline, groups=kept(groups), keys=kept(keys), order=kept(order),
                 y=y_arr[rest], seed=seed)
    head = pipeline[:-1] if len(pipeline.steps) > 1 else None
    X_fit, X_val = _take(X, rest), _take(X, held)
    if head is not None:
        X_fit = head.fit_transform(X_fit, _take(y, rest))
        X_val = head.transform(X_val)
    model.set_params(early_stopping=True)
    model.fit(X_fit, y_arr[rest], X_val=X_val, y_val=y_arr[held])
    return survival_baseline(pipeline, X, y_arr)


__all__ = ["EARLY_STOPPING_ROWS", "HASH_COLUMNS", "fit_pipeline", "inner_splits", "row_keys",
           "unit_labels", "validation_rows", "with_grouped_inner_cv", "with_inner_cv"]

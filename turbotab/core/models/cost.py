"""What fitting each family will take, measured before the fit (M2_CONTRACT §12.6).

Honest cost at scale: elastic net at 20,000 columns takes minutes, and the shelf says so before the
user chooses it. The estimate is measured the way the substitution band's is (one refit timed,
then multiplied by the refits the band makes): each family's own pipeline, built exactly as the
design stage builds it for the whole table, is fit once on a sample of the training rows and
columns and timed. Coordinate descent, histogram building and least squares all cost about one
pass over the cells per iteration, so the time is scaled by the cells the real fit reads, then by
the fits the fit stage makes: one per fold on (k − 1)/k of the rows plus the refit on all of them,
which comes to k fits of the whole table.

A table that fits inside the timing sample is timed whole, so the scaling is exact there. Above
it the number is an estimate and is worded as one ("about 5 minutes").
"""
from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_CELLS = 200_000  # rows × columns in the timing sample
SAMPLE_COLUMNS = 1_000  # columns in the timing sample, at most
SAMPLE_ROWS = 5_000  # rows in the timing sample, at most: a tall table's timing stays about a second
MIN_SAMPLE_ROWS = 60  # fewer and an inner cross-validation cannot run
WIDE_COLUMNS = 200  # the estimate names the columns above this (the roles search threshold)
TALL_ROWS = 100_000  # …and the rows above this
# Below this a fit's length answers no question the user has: it is stated plainly, its cause is
# not named, and the models card says nothing about it.
NOTEWORTHY_SECONDS = 10.0


@dataclass(frozen=True)
class Estimate:
    seconds: float
    text: str  # "about 5 minutes at `20,004` columns"


def sample_shape(n_rows: int, n_columns: int) -> tuple[int, int]:
    """(rows, columns) of the timing sample for a training table of this shape."""
    columns = max(1, min(int(n_columns), SAMPLE_COLUMNS))
    rows = min(int(n_rows), SAMPLE_ROWS, max(MIN_SAMPLE_ROWS, SAMPLE_CELLS // columns))
    return rows, columns


def duration(seconds: float) -> str:
    """A measured duration in words, rounded as an estimate is: ``about 5 minutes``."""
    if seconds < 1:
        return "under a second"
    if seconds < 60:
        n = int(round(seconds)) if seconds < 10 else int(5 * round(seconds / 5))
        return f"about {n} second{'s' if n != 1 else ''}"
    if seconds < 3600:
        n = max(1, int(round(seconds / 60)))
        return f"about {n} minute{'s' if n != 1 else ''}"
    n = max(1, int(round(seconds / 3600)))
    return f"about {n} hour{'s' if n != 1 else ''}"


def say(seconds: float, n_rows: int, n_columns: int) -> str:
    """The estimate as the shelf and the models card state it, naming what makes it large (only
    when it is: "under a second at `394` columns" would explain a length nobody has)."""
    from turbotab.core.voice import tick

    text = duration(seconds)
    if seconds < NOTEWORTHY_SECONDS:
        return text
    if n_columns > WIDE_COLUMNS:
        return f"{text} at {tick(f'{n_columns:,}')} columns"
    if n_rows >= TALL_ROWS:
        return f"{text} on {tick(f'{n_rows:,}')} rows"
    return text


def time_one_fit(pipeline: Any, X: Any, y: Any) -> float | None:
    """Seconds one fit of ``pipeline`` takes on ``X``/``y``; None when the fit cannot run."""
    started = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(X, y)
    except Exception:  # noqa: BLE001 - a sample this fit cannot take says nothing about the cost
        log.debug("a timing fit failed", exc_info=True)
        return None
    return time.perf_counter() - started


def estimate_fits(store: Any, state: Any, task: str, train_ids: Any, families: Sequence[Any],
                  folds: int, *, cancelled: Any = None) -> dict[str, Estimate | None]:
    """``{family key: Estimate | None}`` for fitting each family on these training rows."""
    from turbotab.core.models.pipeline import build_pipeline, design_spec, model_predictors, modeling_frame
    from turbotab.core.stages.modeling import coded_outcome

    out: dict[str, Estimate | None] = {f.key: None for f in families}
    predictors = model_predictors(state)
    train_ids = np.asarray(train_ids, dtype=np.int64)
    n_rows, n_columns = int(len(train_ids)), len(predictors)
    target = getattr(state, "target", None)
    if not predictors or not n_rows or not target:
        return out
    rows, columns = sample_shape(n_rows, n_columns)
    rng = np.random.default_rng(0)
    sample_ids = np.sort(rng.choice(train_ids, size=rows, replace=False)) if rows < n_rows else train_ids
    chosen = (sorted(rng.choice(n_columns, size=columns, replace=False).tolist())
              if columns < n_columns else list(range(n_columns)))
    sampled = [predictors[i] for i in chosen]
    # read as the fit stage reads them (booleans as 0/1, the named event coded 1)
    frame = modeling_frame(store, [*sampled, target], sample_ids)
    frame = frame.loc[frame[target].notna()]
    if not len(frame):
        return out
    X = frame[sampled]
    y = coded_outcome(task, frame[target].to_numpy(), getattr(state, "event", None))
    spec = design_spec(state, X, sampled, energy=None)  # adjusting energy costs next to nothing
    scale = (n_rows / len(y)) * (n_columns / max(1, len(sampled)))
    for family in families:
        if callable(cancelled) and cancelled():
            break
        pipeline = build_pipeline(spec, family, task, getattr(state, "purpose", None), n_rows, n_columns)
        seconds = time_one_fit(pipeline, X[spec.inputs], y)
        if seconds is None:
            continue
        total = seconds * scale * max(1, int(folds))
        out[family.key] = Estimate(seconds=round(total, 1), text=say(total, n_rows, n_columns))
    return out


__all__ = ["NOTEWORTHY_SECONDS", "Estimate", "duration", "estimate_fits", "sample_shape", "say",
           "time_one_fit"]

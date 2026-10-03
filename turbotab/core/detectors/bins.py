"""Histogram bins aligned to the data's resolution, one rule for both histograms (audit MI-01, C9).

The datastore's histogram split [min, max] into 30 equal bins; on values recorded to a step (whole
years, a 0–40 score, HbA1c to 0.1) a bin then held two grid values or one, a sawtooth with empty
bins that is an artifact of the binning, not of the data. ``consequences._histogram_pair`` used
``numpy.histogram`` on the same range with its own edge convention, so the two pictures of one
column disagreed.

The rule here, used by both: find the column's **resolution** r, the coarsest step in
{1, 2, 2.5, 5} × 10^k that every value is a whole multiple of. When the grid between min and max
has K = (max − min)/r + 1 points and K is at most ``MAX_GRID`` × bins, every bin is a whole number
of grid steps wide, m = ⌈K / bins⌉, and the edges sit half a step off the grid, so no value lies on
an edge and each bin holds exactly m grid values. Otherwise the values are read as continuous and
the range is split into equal bins. A value goes to bin ⌊(x − start) / width⌋, clamped to the last
bin, in both implementations.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

STEPS = tuple(sorted({m * 10.0 ** k for k in range(-6, 7) for m in (1.0, 2.0, 2.5, 5.0)},
                     reverse=True))
TOLERANCE = 1e-6
MAX_GRID = 50


def resolution(values: Any) -> float | None:
    """The coarsest conventional step every finite value is a multiple of, or None."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    x = np.unique(x)
    for s in STEPS:
        q = x / s
        r = np.round(q)
        # A non-zero value that rounds to zero steps is finer than the step, never a multiple.
        if np.all((np.abs(q - r) <= TOLERANCE * np.maximum(1.0, np.abs(q))) & ((r != 0) | (x == 0))):
            return float(s)
    return None


def layout(lo: float, hi: float, step: float | None, bins: int) -> tuple[float, float, int, list[float]]:
    """``(start, width, n_bins, edges)`` for values in [lo, hi] on a grid of ``step`` (or None)."""
    if hi == lo:
        return lo - 0.5, 1.0, 1, [lo - 0.5, lo + 0.5]
    if step is not None and step > 0:
        k = int(round((hi - lo) / step)) + 1
        if k <= MAX_GRID * bins:
            m = max(1, math.ceil(k / bins))
            width = m * step
            nb = max(1, math.ceil(k / m))
            start = lo - step / 2
            return start, width, nb, [start + i * width for i in range(nb + 1)]
    width = (hi - lo) / bins
    return lo, width, bins, [lo + i * width for i in range(bins)] + [hi]


def assign(x: Any, start: float, width: float, nb: int) -> np.ndarray:
    """The bin of each finite value: ⌊(x − start) / width⌋, clamped to [0, nb − 1]."""
    x = np.asarray(x, dtype=float)
    b = np.floor((x - start) / width).astype(np.int64)
    return np.clip(b, 0, nb - 1)


def counts(x: Any, start: float, width: float, nb: int) -> list[int]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return np.bincount(assign(x, start, width, nb), minlength=nb).astype(int).tolist()


def resolution_sql(x: str) -> str:
    """One DuckDB aggregate per step: whether every finite value of ``x`` is a multiple of it."""
    return ", ".join(
        f"bool_and(abs({x} / {s!r} - round({x} / {s!r})) <= {TOLERANCE!r} * greatest(1.0, "
        f"abs({x} / {s!r})) AND (round({x} / {s!r}) <> 0 OR {x} = 0))" for s in STEPS)


__all__ = ["STEPS", "assign", "counts", "layout", "resolution", "resolution_sql"]

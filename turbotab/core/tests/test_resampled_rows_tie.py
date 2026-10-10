"""An undersampling or oversampling draw on rows whose two classes are exactly as common.

``levers.resampled_rows`` named the minority with ``argmin`` and the majority with ``argmax``;
with the counts tied both picked the first level, so the draw held that level's rows twice and
none of the other's, and the wrapped logistic refused one class. C6a phase 3 met it when the
imbalance correction's path search fits each grid point through the wrapper (RECIPES §4.3) and an
inner split's recalibration fold held as many events as non-events. The expected rows are worked
by hand: with the classes tied there is nothing to draw up or down, so every row once.
"""
from __future__ import annotations

import numpy as np
import pytest

from turbotab.core.methods.levers import resampled_rows


@pytest.mark.parametrize("method", ["undersample", "oversample"])
def test_tied_classes_keep_every_row_once(method):
    y = np.array([1, 0, 1, 0, 0, 1])
    rows = resampled_rows(y, method, np.random.default_rng(0))
    assert rows.tolist() == [0, 1, 2, 3, 4, 5]


@pytest.mark.parametrize("method, n_rows", [("undersample", 4), ("oversample", 8)])
def test_unequal_classes_draw_to_the_other_count(method, n_rows):
    y = np.array([0, 0, 0, 1, 0, 1])  # 4 non-events, 2 events
    rows = resampled_rows(y, method, np.random.default_rng(0))
    counts = np.bincount(y[rows], minlength=2)
    assert len(rows) == n_rows and counts[0] == counts[1]

"""``effects.column_basis`` fills one preallocated buffer with its orthonormal basis, where it grew
a copy by a column each time (INBOX:1021, speed only): its kept columns are R's. A refactor guard,
so each test passes on the base too. Truth: by derivation (columns in general position, a zero
column, an exact copy) and R's ``qr(x, tol = 1e-7)``, ``lm``'s limited pivoting, whose first
``rank`` pivots are the kept columns in order.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbotab.core.models.effects import column_basis
from turbotab.core.tests.acceptance.r_reference import needs_r, run_r


def _wide() -> np.ndarray:
    """6 rows, 10 columns in general position but the third, all zero, and the fifth, a copy of
    the first: the first six others span every row, so the kept columns are 0, 1, 3, 5, 6, 7 and
    the buffer, 6 columns wide, fills before the last two are read."""
    X = np.random.default_rng(3).normal(size=(6, 10))
    X[:, 2] = 0.0
    X[:, 4] = X[:, 0]
    return X


def _every_level() -> np.ndarray:
    """The intercept, a 3-level category coded by every level, a numeric column and a 4-level
    category coded by every level, 300 rows: the last level of each category is aliased."""
    rng = np.random.default_rng(5)
    return np.column_stack([np.ones(300), np.eye(3)[rng.integers(0, 3, 300)], rng.normal(size=300),
                            np.eye(4)[rng.integers(0, 4, 300)]])


@pytest.mark.parametrize(("make", "want"), [(_wide, [0, 1, 3, 5, 6, 7]),
                                            (_every_level, [0, 1, 2, 4, 5, 6, 7])])
def test_the_kept_columns_are_the_derived_ones(make, want):
    assert column_basis(make()) == want


@needs_r
def test_the_kept_columns_are_rs_qr(tmp_path):
    """R's ``qr(x, tol = 1e-7)`` on the same two matrices: its first ``rank`` pivots."""
    frames = {"w": pd.DataFrame(_wide()), "e": pd.DataFrame(_every_level())}
    r = run_r("""
        kept <- function(path) {
            q <- qr(as.matrix(read.csv(path)), tol = 1e-7)
            I(q$pivot[seq_len(q$rank)] - 1)
        }
        out(list(w = kept(w_csv), e = kept(e_csv)))
    """, frames, tmp_path)
    assert column_basis(_wide()) == r["w"]
    assert column_basis(_every_level()) == r["e"]

"""`DRIVE-044`, `046`, `048`, `049` — four sentences run 5 read that were not true.

Each is small and each is the same family: **a statement placed where a reader
takes it as being about their own numbers, while it is about something else.**

* **`DRIVE-044`** — the calibration caption appended *"(a slope below 1
  indicates predictions that are too extreme)"* as literal text, unconditional
  on the slope printed immediately before it. Run 5 read it beside slope
  **1.141**, which is the opposite problem. The irony is that it sits inside
  the one caption run 5 praised for disclosing its own inadequacy.
* **`DRIVE-046`** — Table 1's Overall column pooled the whole uploaded table,
  so path 1's manuscript failed its own validator: *"Expected analysis N=6297,
  Table 1 overall N=21849."* The strata were right; only Overall pooled.
* **`DRIVE-048`** — the imbalance finding says *"Accuracy can be misleading"*
  and the held-out table then leads with Accuracy 0.88, sitting **at** the
  87.77% base rate. Both true, three cards apart.
* **`DRIVE-049`** — *"No model is selected"* under a button reading *"Fit 2
  model(s)"*. Not stale: the panel was answering about the RECORD's selection
  while the button answered about the page's, and the page never records one.

**The app caught `DRIVE-046` itself** — `passed: false`, rather than exporting
silently — which is the refusal apparatus working and is why that one is medium
rather than high.
"""
# The tests here that drove the retired legacy app (turbotab/api.py, the page, the
# figure and manuscript modules, the gates) were removed with it, BLUEPRINT §9.1; git keeps them.
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import engine                                             # noqa: E402
from turbotab.project import AnalysisProject                    # noqa: E402


# ── DRIVE-046 · Table 1's Overall column ────────────────────────────────────

def _partly_labeled(n=400, labeled=250):
    rng = np.random.default_rng(4)
    outcome = pd.Series(rng.choice(["case", "ctl"], n, p=[0.8, 0.2]),
                        dtype=object)
    outcome.iloc[labeled:] = None
    frame = pd.DataFrame({"x": rng.normal(0, 1, n).round(2),
                          "z": rng.normal(0, 1, n).round(2),
                          "y": outcome})
    project = AnalysisProject.from_dataframe(frame, "p.csv")
    project.set_target("y", "classification", "high", [])
    engine.record_fix(project, "positive_class__y", choice="case")
    project.set_grain("one_row_per_person")
    project.set_eligibility("everyone")
    return project, n, labeled


# ── DRIVE-048 · the score every metric must beat ────────────────────────────


def test_a_regression_run_reports_no_base_rate():
    """`None`, never `0.0` — a base rate on a continuous outcome is not a
    quantity, and zero is a score."""
    from turbotab import training as T

    rng = np.random.default_rng(9)
    n = 300
    frame = pd.DataFrame({"x": rng.normal(0, 1, n), "y": rng.normal(0, 1, n)})
    project = AnalysisProject.from_dataframe(frame, "p.csv")
    project.set_target("y", "regression", "high", [])
    project.set_grain("one_row_per_person")
    project.set_eligibility("everyone")
    idx = list(project.df.index)
    project.seal_lockbox(idx[:75], fraction=0.25)
    assert T.train(project, ["ridge"]).majority_class_rate is None

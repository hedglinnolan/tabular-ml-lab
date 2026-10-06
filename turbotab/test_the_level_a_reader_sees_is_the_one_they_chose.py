"""`DRIVE-040`, the half `L61` did not carry — three surfaces, named.

`L61-D1` closed this row on the figure caption, and the caption is right:
*"945 observations with 829 events of True."* The row is about **the encoded
value reaching a reader**, and after that fix it still did, at three places run
5 read on screen:

* the **PCA group annotation** — `<NA> 15,552, 1 5,527, 0 770`
* **Table 1's column headers** — `0 (n=770)` / `1 (n=5527)`
* the **event noticing card**, which flips from `False`/`True` to `0.0`/`1.0`
  the moment the answer is recorded

## What had to be built first, because the row assumed otherwise

The row said *"the name is available… verify that before building"*. Measured:
it was **half** available. `chosen_level_text` spells the level a caller asks
about, and the recorded decision carried `event_level` — the EVENT's name. All
three surfaces above render **both** levels, and the comparison level's name
was on the record nowhere: after the repair the live finding's own `spellings`
are recomputed against the encoded column to `{'0': '0', '1': '1'}`, so the
original words survived only inside the decision's prose sentence.

So `comparison_level_text` is new, `engine.record_fix` records both, and
`training.outcome_level_names` is the one reader. That is the *bigger finding*
the row anticipated, and it is small only because the sentence that already
spells both levels was one function away.

## What this file will not let happen

**Silence, never a guess.** Every project sealed before `L62` has no
`comparison_level` on its record, and every surface below must then render what
it rendered before — the encoded value. A renderer that filled the gap by
sorting, or by assuming `0` is the level that is not the event, would be
putting a word in a user's column that they never typed.
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
from turbotab import training as T                            # noqa: E402
from turbotab.project import AnalysisProject                  # noqa: E402

#: The event is the MAJORITY, as run 5's was, so a surface that happened to
#: render the more common level first would not pass by accident.
EVENT, COMPARISON = "True", "False"


def _frame(n: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(19)
    return pd.DataFrame({
        "age": rng.normal(50, 12, n).round(1),
        "bmi": rng.normal(27, 5, n).round(1),
        "meds": rng.choice([True, False], n, p=[0.85, 0.15]),
    })


def _answered(*, answer: bool = True) -> AnalysisProject:
    project = AnalysisProject.from_dataframe(_frame(), "p.csv")
    project.set_target("meds", "classification", "high", [])
    if answer:
        engine.record_fix(project, "positive_class__meds", choice="true")
    project.set_grain("one_row_per_person")
    project.set_eligibility("everyone")
    return project


# ── the record, which is what the three surfaces read ───────────────────────


def test_a_record_without_the_comparison_names_only_the_event():
    """**Every project sealed before `L62` is this project.**

    `engine.record_fix` recorded `event_level` from `L61` and
    `comparison_level` only from `L62`, so an older record has one of the two.
    The mapping must carry what it has and omit what it does not, because a
    partial answer is still an answer and a filled-in gap is a fabrication.
    """
    project = _answered()
    decision = T.event_decision(project)
    decision.payload.pop("comparison_level")
    assert T.outcome_level_names(project) == {1: EVENT}

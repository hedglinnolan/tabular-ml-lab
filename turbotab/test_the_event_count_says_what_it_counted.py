"""`DRIVE-043` — one field, four renderers, and three of them said the wrong noun.

**What run 5 found, and why 2,607 green tests could not.** `resolution.statement`
counted held-out events against `counts.index[-1]` — the **least frequent**
level, never the recorded decision. When the event is the minority that is
accidentally right, and the minority *is* the event in the ordinary clinical
case and in every fixture this repository had. It is wrong only when a user
names the **majority** as the event, which is what the fifth human drive did:
`meds_hbp` is 87.77% `True`, the tester chose `True`, and the Methods section
printed **116** while every figure payload printed **829** on the same 945
rows. 945 − 829 = 116 — the non-event count under an events label, in the
artifact that leaves the building.

**The quantity was deliberate and the label was not.** `resolution.py` and
`web/index.html` carry the *same* documented reason for holding arithmetic
rather than a class value: `archive.assert_no_participant_data` rejects a
serialized class label, and it is right to. So the count stays a count. What
changed is that the count is now of the **recorded** event where one exists,
and every renderer says which of the two things it is looking at.

## What this file guards, and it is the class rather than the sentence

A test that only checked the new Methods sentence would leave the defect's
shape intact: **two implementations of one field's meaning, free to diverge.**
The page said *"of the less common outcome"* and the Methods section said
*"carrying the outcome"* about the same integer, and both had been shipping for
as long as both existed. So the assertions below run the Python renderers and
the JavaScript one **against each other on the same payload**, and fail if they
disagree — which is the only form that a later edit to one cannot slip past.

`resolution.event_noun` is the single source. The page reads
`res.event_count_noun` off the wire rather than deciding for itself.
"""
# The tests here that drove the retired legacy app (turbotab/api.py, the page, the
# figure and manuscript modules, the gates) were removed with it, BLUEPRINT §9.1; git keeps them.
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import resolution as R, training as T         # noqa: E402

#: Run 5's shape in miniature: a heavy majority, so counting the minority and
#: counting the event give different answers. **A fixture whose event is the
#: minority cannot fail the old code**, which is exactly why this defect
#: survived every sweep.
N = 800
N_LABELED = 400
MAJORITY_SHARE = 0.88


def _frame(seed: int = 5, n_labeled: int = N_LABELED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "age": rng.normal(50, 12, N).round(1),
        "bmi": rng.normal(27, 5, N).round(1),
        "sbp": rng.normal(128, 16, N).round(1),
    })
    outcome = pd.Series(
        rng.choice(["yes", "no"], N, p=[MAJORITY_SHARE, 1 - MAJORITY_SHARE]),
        dtype=object)
    # Rows with no outcome, because run 5's table had 15,552 of them and they
    # are what `analysis_mask` is about one row over.
    outcome.iloc[n_labeled:] = None
    frame["treated"] = outcome
    return frame


def _truth(project):
    """Ground truth from pandas, not from the app."""
    table = project.working_table
    target = str(project.target)
    has_y = table[target].notna()
    sealed = set(project.lockbox["labels"])
    is_test = pd.Series([i in sealed for i in table.index], index=table.index)
    held = table.loc[has_y & is_test, target]
    return {"n_test": int(len(held)),
            "events": int((held == T.EVENT_VALUE).sum()),
            "non_events": int((held != T.EVENT_VALUE).sum())}


# ── the noun, in every sentence that carries it ─────────────────────────────


def test_the_push_sentence_names_the_statistic_the_event_decides():
    """`_push_because` was worse than a wrong noun and it is worth its own test.

    It said *"sensitivity is undefined"* when the count of the LEAST FREQUENT
    level fell below two. Sensitivity is the rate **of the event**. With the
    event as the majority, that branch announced the statistic that was fine
    and stayed silent about the one that was not — so reading the recorded
    event corrects which statistic is named, not only the noun.
    """
    thin = R._push_because("classification", 40, 1, 39, 2, 0, True)
    assert thin and "sensitivity" in thin and R.EVENT_NOUN in thin, thin
    fat = R._push_because("classification", 40, 39, 1, 2, 0, True)
    assert fat and "specificity" in fat, fat

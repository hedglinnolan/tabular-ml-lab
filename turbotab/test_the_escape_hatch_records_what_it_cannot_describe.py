"""`My design isn't described here` — the answer space is closed, the world is not.

A nested case-control with matched pairs has no correct answer among *no*, *yes
with this column* and *not sure*. Its rows are neither independent participants
nor repeated measures of one person; they are **matched sets**, and a split has
to keep a set together for a reason none of the three describes. A crossover
trial is the same shape from another direction.

Forcing one of the three produces exactly the confidently-wrong answer the
constitution forbids — and it produces it **in the record**, where a reader takes
it as a description of the study.

Same shape as *"I'm not sure"*, and for the same reason: **uncertainty must never
cost more than a wrong confident answer.**

Two independent hatches, because the two questions are independent: the design
may be undescribable while the unit is clear, or the reverse.
"""
# The tests here that drove the retired legacy app (turbotab/api.py, the page, the
# figure and manuscript modules, the gates) were removed with it, BLUEPRINT §9.1; git keeps them.
from __future__ import annotations

import os
import sys
from pathlib import Path


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import grain as G, repeats as R                           # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"


# ── it is offered at both questions ──────────────────────────────────────────

def test_both_questions_offer_it_and_the_page_can_submit_it():
    from ml import router
    plan = router.plan([], target="y", detection=None, step="data", deferred={},
                       answered=["state_lens", "choose_target"],
                       recommendations=[], signals=None, missing_columns=[])
    grain = next(q.to_dict() for q in plan if q.key == "state_grain")
    assert "My design isn't described here" in grain["options"]
    assert G.DESIGN_NOT_DESCRIBED in grain["option_values"]

    plan = router.plan([], target="y", detection=None, step="data", deferred={},
                       answered=["state_lens", "choose_target", "state_grain",
                                 "state_repeat_kind"],
                       recommendations=[], signals=None, missing_columns=[],
                       repeats={"reading": R.REPEATS, "sentence": "…",
                                "confidence": "medium", "kind": R.REPEATS,
                                "unit": None, "menu": None})
    unit = next(q.to_dict() for q in plan if q.key == "state_unit_of_analysis")
    assert "My design isn't described here" in unit["options"]
    assert R.UNIT_NOT_DESCRIBED in unit["option_values"]

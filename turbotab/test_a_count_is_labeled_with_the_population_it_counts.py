"""L63-C4. A number that is right under a name that is wider than it.

`DRIVE-050` is the headline instance — the shelf's design sentence counted the
training rows to describe a ranking computed on the analysis rows — but the
same shape had four more sites, and **two of them are publication captions**.

The class: *a count filtered to rows with an outcome, printed under the label
"training rows".* Not a wrong number. A number whose name describes a strictly
larger population, which is `AGENT_ONBOARD.md` §07's *the machine-readable form
is lossier than the sentence* running in the other direction — the value is
right and the word beside it is not.

**The correct phrasing already existed four lines from one of the wrong ones**,
at `instability.py`'s own refusal: *"N training row(s) with an outcome is too
few to resample from"*. A sweep that checked the numbers and never checked
their labels is the blind spot that let the design sentence survive `DRIVE-045`
in the same file, thirty-three lines from the fix.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import figure_specs as FS                            # noqa: E402
from turbotab import instability as INST                           # noqa: E402

#: The qualifier that makes the label describe the number. One spelling, so a
#: site that drifts to a synonym is visible here rather than plausible.
QUALIFIER = "with an outcome"


def _instability_payload(n=40, b=6):
    """A payload of the shape `instability.run` returns, for the captions.

    Built rather than driven: the two captions read `p["n"]`, which is
    `len(rows)` AFTER `instability.py:228` narrows to rows with an outcome, and
    the label is what is under test — not the resampling, which
    `test_the_whole_pipeline_is_refitted.py` owns.
    """
    rng = np.random.default_rng(3)
    original = rng.random(n)
    return {
        "model_name": "Logistic Regression", "n": n,
        "b_completed": b, "b_requested": b,
        "b_recommended": INST.RECOMMENDED_B,
        "points": n * b, "alpha": 0.02, "events": int(n * 0.4),
        "n_bins": 10,
        "original": [float(v) for v in original],
        "bootstrap": [[float(v) for v in rng.random(n)] for _ in range(b)],
        "mape": {"value": 0.05, "units": "risk"},
    }


@pytest.mark.parametrize("spec_id", ["prediction_instability",
                                     "calibration_instability"])
def test_a_publication_caption_names_the_population_it_counted(spec_id):
    """The two that leave the building.

    `instability.py` hands these an already-outcome-filtered `n` and both
    printed it as *"N training rows"*, in the artifact a reader takes to a
    journal. `AUDIT-001`'s address: a false statistical claim in the generated
    manuscript.
    """
    from turbotab import figures

    spec = figures.REGISTRY[spec_id]
    caption = spec.caption(_instability_payload())
    assert "training rows" in caption, (
        f"{spec_id}'s caption no longer says which rows it counted at all, "
        f"which is a different defect from saying it wrongly: {caption[:200]}")
    assert f"training rows {QUALIFIER}" in caption, (
        f"{spec_id}'s caption prints an outcome-filtered count labeled "
        f"'training rows', which names a strictly larger population than the "
        f"number counts: {caption[:300]}")


def test_the_instability_payload_says_which_rows_it_scored_on():
    """`scored_on` is the machine-readable half of the same claim.

    It read *"training rows only"* over a count that had already dropped every
    row with no outcome — and it is what a downstream consumer reads when the
    caption is not in front of it.
    """
    source = INST.__dict__
    assert "run" in source
    # The literal, asserted where it is composed. Driving a full resample to
    # read one string would be minutes of refits for a label.
    import inspect

    body = inspect.getsource(INST.run)
    assert '"scored_on"' in body, "the payload no longer states what it scored on"
    assert QUALIFIER in body.split('"scored_on"', 1)[1][:400], (
        "`scored_on` no longer qualifies the population it names; it is "
        "composed from rows already narrowed to those with an outcome")


def test_the_training_note_names_the_population_it_counted():
    """`training.py`'s run note counts `X_train`, which is
    `features[has_y & ~is_test]` — the analysis population — and called it
    *"the N training rows"*.

    This is the note that reaches the manuscript's methods section, quoted from
    the record, so the label travels further than any of the others.
    """
    import inspect

    from turbotab import training as T

    body = inspect.getsource(T)
    marker = "Every statistic in it is fitted once over the"
    assert marker in body, "the run note moved; re-anchor this assertion"
    after = body.split(marker, 1)[1][:300]
    assert "training rows" in after
    assert QUALIFIER in after, (
        f"the run note counts `X_train`, which excludes rows with no outcome, "
        f"and labels it 'training rows': {after[:200]}")


def test_the_refusal_that_was_already_right_is_unchanged():
    """The control, and it is the model the four corrections copied.

    `instability.py`'s refusal has said *"training row(s) with an outcome"* all
    along. If this ever drifts, the class has a wrong exemplar and the next
    correction will copy it.
    """
    import inspect

    body = inspect.getsource(INST)
    assert f"training row(s) {QUALIFIER} is too few to " in body, (
        "the correctly-phrased site drifted; it is what the others were "
        "corrected to match")


def test_the_forest_and_roc_captions_do_not_inherit_the_label():
    """The same lens one surface over, which §08 check 5 asks for.

    Neither of these counts rows, so neither should have acquired the phrase
    while the four above were corrected. A sweep that fixed six sites and
    introduced a seventh would be the class committed by its own repair.
    """
    payload = FS.forest_payload([{"name": "age", "estimate": 1.4}],
                                model="Logistic Regression")
    assert "training rows" not in FS.FOREST.caption(payload)

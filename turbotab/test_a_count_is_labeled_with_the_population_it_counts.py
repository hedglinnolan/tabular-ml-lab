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
# The tests here that drove the retired legacy app (turbotab/api.py, the page, the
# figure and manuscript modules, the gates) were removed with it, BLUEPRINT §9.1; git keeps them.
from __future__ import annotations

import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


#: The qualifier that makes the label describe the number. One spelling, so a
#: site that drifts to a synonym is visible here rather than plausible.
QUALIFIER = "with an outcome"


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



"""`AUDIT-024` — the Classic Feature Selection page states §A5.5's objection.

`CLINICAL_SURVEY_PACK.md` §A5.5, **[SETTLED]**: *"Avoid univariable pre-screening of
predictors by p-value. It is one of PROBAST's explicit high-risk-of-bias signals: it
discards variables that matter only in combination, and it invalidates the p-values in
the final model."* And, separately **[SETTLED]**: *"Avoid stepwise selection. It produces
unstable variable sets, biased coefficients, and confidence intervals with wrong
coverage."*

`pages/04_Feature_Selection.py` shipped a BH-FDR univariable p-value screen **pre-ticked**,
whose survivors are written into `data_config.feature_cols` and drive every downstream
page, and the page's only methodological framing listed four *benefits* of selecting
features. The registry's objection appeared nowhere in shipped code.

**The shelf is not shortened** (`AGENT_ONBOARD.md` §08 check 6). The method is still
offered — it is the standard tool for high-dimensional discovery — so this file asserts
both halves: the control is *present*, and it is present *unticked and with the objection
beside it*. A future change that deletes the checkbox fails here just as one that
re-ticks it does.

Every assertion reads text `AppTest` actually rendered (§07 trap 6), and the objection is
required in an **always-visible** element rather than only inside the collapsed
"Why feature selection?" expander — a sentence a user must click to reveal is not the
same as one they are shown.

`GUIDED-097`: run against two fixtures of different target shape — a continuous float
outcome (`glucose`) and a binary 0/1 outcome (`condition`). **The shape not covered is a
non-numeric outcome** (string labels / multi-class); `tests/integration/conftest.py` has
no such fixture, and `TARGET_SHAPE_NOT_COVERED` records the gap.

`GUIDED-045`: the absence assertion (the checkbox is not pre-ticked) is preceded by a
positive control that the checkbox panel rendered at all.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from streamlit.testing.v1 import AppTest
from tests.integration.conftest import (
    build_classification_dataframe,
    build_test_dataframe,
    inject_data_state,
)

TARGET_SHAPE_NOT_COVERED = "non-numeric outcome (string labels / multi-class)"

UNIVARIATE_LABEL = "Univariate Screening (FDR-corrected)"

TARGET_SHAPES = [
    pytest.param(build_test_dataframe, "glucose", "regression", id="continuous-float"),
    pytest.param(
        build_classification_dataframe, "condition", "classification", id="binary-0-1"
    ),
]


def _run(builder, target, task):
    at = AppTest.from_file("pages/04_Feature_Selection.py", default_timeout=120)
    inject_data_state(at, builder(), target_col=target, task_type=task)
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    return at


def _always_visible_text(at):
    """Text a person sees without opening anything.

    `st.caption` and `st.info` on this page render at page level and inside columns;
    the "Why feature selection?" framing is `st.markdown` inside a collapsed
    `st.expander`. Excluding markdown here is what makes this an assertion about
    what is *shown* rather than what is merely present in the element tree.
    """
    parts = []
    for attr in ("caption", "info", "warning"):
        for el in getattr(at, attr, []):
            parts.append(str(getattr(el, "value", "")))
    return "\n".join(parts)


@pytest.mark.parametrize("builder,target,task", TARGET_SHAPES)
def test_univariable_screening_is_still_offered(builder, target, task):
    """§08 check 6: the correction is a disclosure, not a deletion."""
    at = _run(builder, target, task)
    labels = [c.label for c in at.checkbox]
    assert labels, "Feature Selection rendered no checkboxes — nothing was swept"
    assert UNIVARIATE_LABEL in labels, (
        f"the univariable screening control was removed rather than disclosed; "
        f"checkboxes present: {labels}"
    )


@pytest.mark.parametrize("builder,target,task", TARGET_SHAPES)
def test_univariable_screening_is_not_pre_ticked(builder, target, task):
    """The [SETTLED]-against method must not be what a user gets by pressing Run."""
    at = _run(builder, target, task)
    boxes = {c.label: c.value for c in at.checkbox}
    # GUIDED-045 positive control.
    assert boxes, "Feature Selection rendered no checkboxes — nothing was swept"
    assert UNIVARIATE_LABEL in boxes, f"control missing; present: {list(boxes)}"

    assert boxes[UNIVARIATE_LABEL] is False, (
        "pages/04_Feature_Selection.py pre-ticks univariable p-value screening. "
        "CLINICAL_SURVEY_PACK.md §A5.5 [SETTLED]: 'Avoid univariable pre-screening "
        "of predictors by p-value. It is one of PROBAST's explicit high-risk-of-bias "
        "signals.' A pre-ticked default is the app recommending it, and its "
        "survivors are written into data_config.feature_cols."
    )


@pytest.mark.parametrize("builder,target,task", TARGET_SHAPES)
def test_the_probast_objection_is_shown_without_opening_anything(builder, target, task):
    """§A5.5's objection is on the always-visible surface, not only in the expander."""
    at = _run(builder, target, task)
    visible = _always_visible_text(at)
    assert visible.strip(), "no caption/info/warning rendered — nothing was swept"
    lowered = visible.lower()

    assert "probast" in lowered, (
        "the rendered Feature Selection page states PROBAST's objection nowhere a "
        "user can see without opening the expander. §A5.5 [SETTLED]: univariable "
        "pre-screening by p-value 'is one of PROBAST's explicit high-risk-of-bias "
        "signals'."
    )
    assert "invalidates the p-values" in lowered, (
        "the page does not tell the user that screening by p-value invalidates the "
        "p-values of the model fitted on the survivors (§A5.5 [SETTLED])"
    )
    assert "matter only in combination" in lowered, (
        "the page does not tell the user that univariable screening 'discards "
        "variables that matter only in combination' (§A5.5 [SETTLED])"
    )


@pytest.mark.parametrize("builder,target,task", TARGET_SHAPES)
def test_the_panel_states_the_objection_to_selecting_from_these_rows_at_all(
    builder, target, task
):
    """Every method on the panel is data-driven selection; §A5.5's second [SETTLED]."""
    at = _run(builder, target, task)
    lowered = _always_visible_text(at).lower()

    assert "unstable" in lowered, (
        "the method panel does not state that selecting predictors from these rows "
        "produces unstable variable sets (§A5.5 [SETTLED], on stepwise selection)"
    )
    assert "wrong coverage" in lowered, (
        "the method panel does not state that confidence intervals from a model "
        "refitted on a data-driven selection have wrong coverage (§A5.5 [SETTLED])"
    )
    assert "pre-specif" in lowered, (
        "the page states the objection but names neither preferred route — "
        "§A5.5: 'Prefer pre-specification on clinical grounds, or penalized "
        "regression which shrinks rather than selects abruptly.'"
    )

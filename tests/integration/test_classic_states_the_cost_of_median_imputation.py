"""`AUDIT-007` — the Classic door says what median imputation costs.

`CLINICAL_SURVEY_PACK.md` §A2 anti-pattern 2 is **[SETTLED as bad]**: *"Mean/median
imputation. Understates variance, destroys the distribution, indefensible in a
manuscript."* Cross-cutting item 11 is blunter still — *"Mean-imputing anything, ever,
without a loud warning."* Anti-pattern 3, also **[SETTLED]**, adds the second half that
applies wherever the fill is computed from `X` alone: *"Imputing with the outcome
excluded from the imputation model. Biases associations toward the null."*

Two Classic surfaces are covered here, and both are **driven**, not composed — every
assertion below reads text that `AppTest` actually rendered, so a sentence the server
builds and the interface never shows cannot pass this file (`AGENT_ONBOARD.md` §07 trap 6):

* `pages/02_EDA.py` told the user that median/mode imputation *"should be sufficient"* —
  an assertion of the sufficiency of the method the registry settles against. That is the
  governing rule failing outright, not merely a silence.
* `pages/04_Feature_Selection.py` fits `SimpleImputer(strategy="median")` over the
  predictors before every selector runs, with the outcome excluded, and disclosed only
  that "results may be affected".

`GUIDED-097`: every claim here runs against two fixtures of **different target shape** —
a continuous float outcome (`glucose`) and a binary 0/1 outcome (`condition`).
**The shape not covered is a non-numeric outcome** — a string-labeled or multi-class
target, which `tests/integration/conftest.py` has no fixture for; `TARGET_SHAPE_NOT_COVERED`
below names it so the gap is a record rather than an omission.

`GUIDED-045`: each absence assertion is preceded by a positive control that the surface
it sweeps is non-empty and that the specific insight under test actually fired.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from streamlit.testing.v1 import AppTest
from tests.integration.conftest import (
    build_classification_dataframe,
    build_test_dataframe,
    inject_data_state,
)

TARGET_SHAPE_NOT_COVERED = "non-numeric outcome (string labels / multi-class)"

# The moderate-missing band on `pages/02_EDA.py` is the intersection of
# `signals.high_missing_cols` (rate > `ml.missingness_plan.HIGH_MISSING_SHARE`, 0.20)
# with the page's own `0.05 < rate <= 0.30`, so only a column in (0.20, 0.30] reaches
# `eda_missing_moderate`. Reproducing that arithmetic here rather than guessing is the
# difference between this file testing the page and testing a fixture.
_MODERATE_RATE = 0.25
_SEVERE_RATE = 0.40


def _with_missing(df, rates):
    df = df.copy()
    rng = np.random.default_rng(11)
    n = len(df)
    for col, frac in rates.items():
        idx = rng.choice(n, int(n * frac), replace=False)
        df.loc[df.index[idx], col] = np.nan
    return df


def _rendered_text(at):
    """Everything a person can actually read on the rendered page."""
    parts = []
    for attr in ("markdown", "caption", "info", "warning", "error", "success"):
        for el in getattr(at, attr, []):
            parts.append(str(getattr(el, "value", "")))
    return "\n".join(parts)


# Two target shapes, per GUIDED-097. `id` is the shape, so a failure names it.
TARGET_SHAPES = [
    pytest.param(build_test_dataframe, "glucose", "regression", id="continuous-float"),
    pytest.param(
        build_classification_dataframe, "condition", "classification", id="binary-0-1"
    ),
]


# ── pages/02_EDA.py ──────────────────────────────────────────────────


@pytest.mark.parametrize("builder,target,task", TARGET_SHAPES)
def test_eda_does_not_tell_the_user_median_imputation_is_sufficient(
    builder, target, task
):
    """The moderate-missing insight states §A2's cost instead of asserting sufficiency."""
    df = _with_missing(builder(), {"cholesterol": _MODERATE_RATE})
    at = AppTest.from_file("pages/02_EDA.py", default_timeout=120)
    inject_data_state(at, df, target_col=target, task_type=task)
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]

    text = _rendered_text(at)

    # GUIDED-045 positive control, in three steps: the page rendered something,
    # the insight under test actually fired, and its implication reached the surface.
    assert text.strip(), "EDA rendered no text at all — nothing was swept"
    ledger = at.session_state["insight_ledger"]
    ids = [i.id for i in ledger.insights]
    assert "eda_missing_moderate" in ids, (
        f"the insight under test did not fire, so the sweep below proves nothing; "
        f"insights present: {ids}"
    )
    implication = next(
        i.implication for i in ledger.insights if i.id == "eda_missing_moderate"
    )
    assert implication in text, (
        "eda_missing_moderate's implication never reached the rendered page — "
        "an absence assertion over text that does not carry the claim is vacuous"
    )

    # The claim itself.
    assert "should be sufficient" not in text, (
        "pages/02_EDA.py still tells the user median/mode imputation "
        "'should be sufficient', against CLINICAL_SURVEY_PACK.md §A2 anti-pattern 2 "
        "[SETTLED as bad]: 'Mean/median imputation. Understates variance, destroys "
        "the distribution, indefensible in a manuscript.'"
    )
    lowered = text.lower()
    assert "understates the variance" in lowered, (
        "the cost §A2 names — understated variance — is stated nowhere on the "
        "rendered EDA page beside the moderate-missing finding"
    )
    assert "distorts its distribution" in lowered, (
        "the second half of §A2 anti-pattern 2 — the distorted distribution — "
        "is not stated on the rendered EDA page"
    )


@pytest.mark.parametrize("builder,target,task", TARGET_SHAPES)
def test_eda_qualifies_the_missingness_indicator_by_what_the_model_is_for(
    builder, target, task
):
    """§A2 splits the indicator: fine for prediction, [SETTLED] biased for inference."""
    df = _with_missing(builder(), {"cholesterol": _MODERATE_RATE})
    at = AppTest.from_file("pages/02_EDA.py", default_timeout=120)
    inject_data_state(at, df, target_col=target, task_type=task)
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]

    text = _rendered_text(at)
    ids = [i.id for i in at.session_state["insight_ledger"].insights]
    assert "eda_missing_moderate" in ids, f"insight did not fire; present: {ids}"

    assert "missingness indicator" in text.lower(), (
        "the indicator recommendation was deleted rather than corrected — "
        "AUDIT-028's model is a weaker true claim on the same subject, not silence"
    )
    lowered = text.lower()
    assert "contraindicated for an unbiased association estimate" in lowered, (
        "pages/02_EDA.py recommends a missingness indicator without the condition "
        "§A2 attaches to it: legitimate for prediction, and 'known to give biased "
        "estimates and should not be used' for inference [SETTLED]"
    )


# ── pages/04_Feature_Selection.py ────────────────────────────────────


@pytest.mark.parametrize("builder,target,task", TARGET_SHAPES)
def test_feature_selection_states_what_the_median_fill_costs_the_ranking(
    builder, target, task
):
    """The page median-fills predictors before every selector; it now says the price."""
    df = _with_missing(builder(), {"cholesterol": _SEVERE_RATE})
    at = AppTest.from_file("pages/04_Feature_Selection.py", default_timeout=120)
    inject_data_state(at, df, target_col=target, task_type=task)
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]

    text = _rendered_text(at)
    lowered = text.lower()

    # GUIDED-045 positive control: the disclosure branch under test actually ran.
    assert text.strip(), "Feature Selection rendered no text at all"
    assert "temporarily filled with column medians" in lowered, (
        "the imputation-disclosure branch did not fire on this fixture, so the "
        "assertions below sweep a surface that was never rendered"
    )

    assert "shrinks that column's variance" in lowered, (
        "pages/04_Feature_Selection.py fits SimpleImputer(strategy='median') over "
        "the predictors and does not state §A2 anti-pattern 2's cost "
        "[SETTLED as bad]: 'Understates variance, destroys the distribution.'"
    )
    assert "biased toward the null" in lowered, (
        "the median fill on this page excludes the outcome, which §A2 anti-pattern 3 "
        "settles: 'Imputing with the outcome excluded from the imputation model. "
        "Biases associations toward the null.' The page does not say so."
    )


# ── pages/05_Preprocess.py ───────────────────────────────────────────
#
# The half the row was filed against and the half that stayed OPEN through the
# previous pass, which owned neither `pages/05_Preprocess.py` nor
# `ml/pipeline.py`. `ml/pipeline.py:213` is where `median` becomes a
# `SimpleImputer`, and `pages/05_Preprocess.py:580` skips the whole per-model
# configuration block while **Smart Defaults** is selected — so on the DEFAULT
# path a person is given median imputation, is never shown the imputation
# control, and cannot reach MICE without first switching mode.


@pytest.mark.parametrize("builder,target,task", TARGET_SHAPES)
def test_the_smart_defaults_path_states_what_its_median_fill_costs(
    builder, target, task
):
    """The path that hides the control is the path that must state the cost."""
    df = _with_missing(builder(), {"cholesterol": _MODERATE_RATE})
    at = AppTest.from_file("pages/05_Preprocess.py", default_timeout=120)
    inject_data_state(at, df, target_col=target, task_type=task)
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]

    # GUIDED-045 positive control, in two steps: the page rendered, and it
    # rendered on the default Smart Defaults path — which is the only path
    # these assertions are about.
    text = _rendered_text(at)
    assert text.strip(), "Preprocess rendered no text at all — nothing was swept"
    modes = [r for r in at.radio if r.label == "Configuration mode"]
    assert modes, "no configuration-mode radio rendered"
    assert "Smart" in str(modes[0].value), (
        f"the default mode is no longer Smart Defaults ({modes[0].value!r}), so "
        f"this test is sweeping a path the user does not land on"
    )
    assert "median imputation" in text, (
        "the Smart Defaults summary no longer says it applies median "
        "imputation, so the cost assertions below have nothing to attach to"
    )

    lowered = text.lower()
    assert "understates that column's variance" in lowered, (
        "pages/05_Preprocess.py applies median imputation on the default path "
        "without stating CLINICAL_SURVEY_PACK.md §A2 anti-pattern 2's cost "
        "[SETTLED as bad]: 'Understates variance, destroys the distribution, "
        "indefensible in a manuscript.'"
    )
    assert "distorts its distribution" in lowered, (
        "the second half of §A2 anti-pattern 2 — the distorted distribution — "
        "is not stated on the default Preprocess path"
    )
    assert "biased toward the null" in lowered, (
        "§A2 anti-pattern 3 [SETTLED] — 'Imputing with the outcome excluded "
        "from the imputation model. Biases associations toward the null.' — is "
        "not stated, and the outcome is excluded from this fill"
    )
    # AUDIT-028: a weaker TRUE claim, not silence and not a blur. The sentence
    # has to keep pointing at the alternative, and has to be honest that the
    # alternative is not on this path.
    assert "mice" in lowered, (
        "the cost is stated and the remedy is not — §A2's alternative must be "
        "named where the default is applied"
    )
    assert "advanced (full control)" in lowered, (
        "the page tells the user MICE exists without saying that the control "
        "is not on this path; pages/05_Preprocess.py:580 skips the per-model "
        "block entirely while Smart Defaults is selected, so an unqualified "
        "'MICE is available in Preprocessing' is a second false claim"
    )


@pytest.mark.parametrize("builder,target,task", TARGET_SHAPES)
def test_the_imputation_control_does_not_endorse_the_settled_anti_pattern(
    builder, target, task
):
    """Advanced mode: the help text under each option carries §A2's cost."""
    df = _with_missing(builder(), {"cholesterol": _MODERATE_RATE})
    at = AppTest.from_file("pages/05_Preprocess.py", default_timeout=120)
    inject_data_state(at, df, target_col=target, task_type=task)
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]

    modes = [r for r in at.radio if r.label == "Configuration mode"]
    assert modes, "no configuration-mode radio rendered"
    modes[0].set_value("🔧 Advanced (full control)").run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]

    # GUIDED-045 positive control: the control under test rendered, and the
    # option whose caption is being read is the one selected.
    imps = [sb for sb in at.selectbox if sb.label == "Numeric imputation"]
    assert imps, (
        f"no numeric-imputation selectbox in Advanced mode; selectboxes: "
        f"{[sb.label for sb in at.selectbox]}"
    )
    assert str(imps[0].value) == "median", (
        f"the pre-selected imputation is {imps[0].value!r}, so the caption "
        f"rendered beneath it is not median's"
    )

    lowered = _rendered_text(at).lower()
    assert "most common default." not in lowered, (
        "pages/05_Preprocess.py still describes median imputation as 'Robust "
        "to skewed distributions. Most common default.' — an endorsement of "
        "the method CLINICAL_SURVEY_PACK.md §A2 anti-pattern 2 settles as bad"
    )
    assert "robust to skew as a point estimate" in lowered, (
        "the true half of the old sentence was deleted rather than narrowed; "
        "AUDIT-028's model is a claim that says LESS but stays true"
    )
    assert "understates that column's variance" in lowered, (
        "the imputation control offers median with no statement of what §A2 "
        "settles it costs"
    )


def test_the_route_to_mice_is_the_route_the_corrected_sentences_name():
    """`AGENT_ONBOARD.md` §07 trap 1: a named capability must have a real consumer.

    Both corrected sentences send the user to multiple imputation *under Advanced
    (full control)*. This drives that exact route rather than grepping for the word,
    and it pins the qualifier: MICE is absent from the rendered page on the default
    Smart Defaults path, which is why the sentences name the mode. If the gate at
    `pages/05_Preprocess.py:580` is ever removed, the first assertion fails and the
    now-overcautious qualifier can be dropped from both sentences.
    """
    df = build_test_dataframe()

    def _options(at):
        opts = []
        for sb in at.selectbox:
            opts.extend(str(o) for o in (sb.options or []))
        return opts

    # Default path: Smart Defaults. MICE is not on it — this is the qualifier's warrant.
    at = AppTest.from_file("pages/05_Preprocess.py", default_timeout=120)
    inject_data_state(at, df)
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    modes = [r for r in at.radio if r.label == "Configuration mode"]
    assert modes, "Preprocess rendered no configuration-mode radio — nothing was swept"
    assert not any("MICE" in o for o in _options(at)), (
        "MICE now renders on the default Smart Defaults path, so the qualifier "
        "'under Advanced (full control)' in pages/02_EDA.py and "
        "pages/04_Feature_Selection.py understates what is reachable"
    )

    # The route the sentences name.
    modes[0].set_value("🔧 Advanced (full control)").run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    options = _options(at)
    assert options, "Advanced mode rendered no selectbox options — nothing was swept"
    assert any("MICE" in o for o in options), (
        "pages/02_EDA.py and pages/04_Feature_Selection.py tell the user multiple "
        "imputation (MICE) is available in Preprocessing under Advanced (full "
        "control); driving that route renders no such option, which would make the "
        "correction a second false claim"
    )

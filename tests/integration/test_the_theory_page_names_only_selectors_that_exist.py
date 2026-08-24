"""`AUDIT-025` — the Theory Reference does not credit Feature Selection with VIF.

The row: `pages/11_Theory_Reference.py`'s `app_connection(...)` callout in the
collinearity/regularization tab told the reader *"The **Feature Selection** page
offers VIF-based filtering as one of its selection methods."*
`pages/04_Feature_Selection.py` offers four methods — LASSO Path, RFE-CV, Univariate
Screening, Stability Selection — and contains zero occurrences of "vif";
`ml/feature_selection.py` has no VIF function. VIF exists only as a read-only
diagnostic table on page 02 (`ml/eda_actions.py`), which warns and **drops nothing**.
The registry's ruling was to correct the sentence, not to build a VIF filter, and
§A5.5 says why it matters directionally: a reader told the app can filter on VIF may
believe a collinearity screen ran when the actual selection was LASSO/RFE/univariate
p-values.

**Disposition honesty.** The sentence was already corrected at `HEAD` (commit
`cc93767`), before this loop. `pages/11_Theory_Reference.py` is not this chunk's to
edit, so what is added here is the guard the correction never had — and the guard is
proved non-vacuous against the historical sentence rather than against a revert
(`HISTORICAL_FALSE_CALLOUT` below, `test_the_matcher_fires_on_the_sentence_the_row_filed`).

**The matcher is anchored, not prose.** `AGENT_ONBOARD.md`: a matcher that fires on
prose has silence that means nothing — and the *corrected* sentence necessarily
mentions both "Feature Selection" and "VIF", so any co-occurrence rule would fire on
the fix. The rule here is the row's own: a capability an `app_connection` string
attributes to a page must be findable in that page's source, and the one escape is an
explicit disclaimer that the app does not have it. Both sides are read structurally —
the callouts by AST, the page-04 methods from the checkbox labels the page renders.

`GUIDED-045`: every absence assertion is preceded by a positive control that the set
being swept is non-empty.
`GUIDED-097`: the driven claim runs against two fixtures of different target shape —
a continuous float outcome (`glucose`) and a binary 0/1 outcome (`condition`).
**The shape not covered is a non-numeric outcome** (string-labeled / multi-class),
for which `tests/integration/conftest.py` has no fixture.
"""
from __future__ import annotations

import ast
import os
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

from tests.integration.conftest import (
    build_classification_dataframe,
    build_test_dataframe,
    inject_data_state,
)

TARGET_SHAPE_NOT_COVERED = "non-numeric outcome (string labels / multi-class)"

THEORY = ROOT / "pages" / "11_Theory_Reference.py"
SELECTION_PAGE = ROOT / "pages" / "04_Feature_Selection.py"
SELECTION_MODULE = ROOT / "ml" / "feature_selection.py"

# The sentence the row was filed against, kept verbatim so the matcher below has
# something real to be measured against. This is the ONLY place it survives.
HISTORICAL_FALSE_CALLOUT = (
    "The <strong>EDA</strong> page computes pairwise correlations and flags highly "
    "correlated pairs (|r| > 0.8). The coaching layer only raises collinearity as an "
    "issue for linear model families, since tree-based and other models are "
    "unaffected. The <strong>Feature Selection</strong> page offers VIF-based "
    "filtering as one of its selection methods."
)

# The one escape from the rule: the app is allowed to name a capability it does
# not have, in order to say that it does not have it.
DISCLAIMER = "no page in this app filters features by"


def _app_connection_strings(path: pathlib.Path) -> list[str]:
    """Every constant string handed to `app_connection(...)`, by AST.

    Adjacent string literals are folded into one `Constant` at parse time, so a
    callout split across ten source lines arrives here as one sentence — which
    is the failure mode `AGENT_ONBOARD.md` §07 trap 5 records for grep.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = getattr(fn, "id", None) or getattr(fn, "attr", None)
        if name != "app_connection":
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                out.append(arg.value)
    return out


def _page_04_offers_vif() -> bool:
    """Structural, on both files the row cites."""
    return (
        "vif" in SELECTION_PAGE.read_text(encoding="utf-8").lower()
        or "vif" in SELECTION_MODULE.read_text(encoding="utf-8").lower()
    )


def _violates(callout: str, page_offers_it: bool) -> bool:
    """The row's rule, as one predicate so it can be aimed at either text."""
    if "VIF" not in callout or "Feature Selection" not in callout:
        return False
    if page_offers_it:
        return False
    return DISCLAIMER not in callout.lower()


def test_the_matcher_fires_on_the_sentence_the_row_filed():
    """GUIDED-045's positive control, and the only proof this guard has teeth.

    The fix predates this loop, so reverting this loop's diff cannot turn the
    guard red. Aiming it at the sentence the row was filed against can.
    """
    assert not _page_04_offers_vif(), (
        "pages/04_Feature_Selection.py or ml/feature_selection.py now contains "
        "'vif' — if a VIF selector was built, this whole guard is measuring a "
        "world that no longer exists and the callout may say so freely"
    )
    assert _violates(HISTORICAL_FALSE_CALLOUT, _page_04_offers_vif()), (
        "the matcher does not flag the exact sentence AUDIT-025 was filed "
        "against, so its silence over the current file means nothing"
    )
    # And it must NOT fire merely because both words appear — otherwise it
    # would flag the correction itself.
    assert not _violates(
        "VIF lives on the EDA page. No page in this app filters features by "
        "VIF — the Feature Selection page's four methods are LASSO path, "
        "RFE-CV, univariate screening and stability selection.",
        _page_04_offers_vif(),
    ), "the matcher fires on a callout that correctly disclaims the capability"


def test_no_theory_callout_credits_feature_selection_with_vif():
    callouts = _app_connection_strings(THEORY)

    # GUIDED-045 positive control, in two steps: the AST found callouts at all,
    # and it found the specific one this row is about.
    assert len(callouts) >= 10, (
        f"only {len(callouts)} app_connection() strings were extracted from "
        f"{THEORY.name}; the AST walk has stopped matching and the sweep below "
        f"is over an almost-empty set"
    )
    collinearity = [c for c in callouts if "VIF" in c]
    assert collinearity, (
        "no app_connection callout mentions VIF at all, so the collinearity "
        "tab's callout is not being swept"
    )

    offers = _page_04_offers_vif()
    bad = [c for c in callouts if _violates(c, offers)]
    assert not bad, (
        "a Theory Reference callout attributes VIF-based filtering to the "
        "Feature Selection page. pages/04_Feature_Selection.py offers LASSO "
        "Path, RFE-CV, Univariate Screening and Stability Selection and "
        "contains no VIF; ml/feature_selection.py has no VIF function. VIF is "
        "a read-only diagnostic on the EDA page and it drops nothing. "
        f"offending callout(s): {bad}"
    )


TARGET_SHAPES = [
    pytest.param(build_test_dataframe, "glucose", "regression", id="continuous-float"),
    pytest.param(
        build_classification_dataframe, "condition", "classification", id="binary-0-1"
    ),
]


@pytest.mark.parametrize("builder,target,task", TARGET_SHAPES)
def test_the_feature_selection_page_offers_no_vif_filter(builder, target, task):
    """Driven, because the claim is about what a person is offered.

    `AGENT_ONBOARD.md` §07 trap 5: grep answers *does this text appear*. The
    question is what the panel renders.
    """
    at = AppTest.from_file("pages/04_Feature_Selection.py", default_timeout=120)
    inject_data_state(at, builder(), target_col=target, task_type=task)
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]

    labels = [str(c.label) for c in at.checkbox]
    # GUIDED-045 positive control: the method panel rendered its options.
    assert labels, "Feature Selection rendered no method checkboxes at all"
    assert any("LASSO" in l for l in labels), (
        f"the method panel did not render; checkboxes present: {labels}"
    )

    vif_offers = [l for l in labels if "vif" in l.lower()]
    assert not vif_offers, (
        f"the Feature Selection panel now offers {vif_offers}. If a VIF "
        f"selector was genuinely built, AUDIT-025's corrected Theory Reference "
        f"callout — which tells the reader no page filters features by VIF — "
        f"has become the false one and must be corrected in the other direction."
    )

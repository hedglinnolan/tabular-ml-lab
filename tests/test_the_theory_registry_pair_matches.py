"""`utils/theory_anchors.py` and `theory_demos.py` — the pair with no key test.

`FEATURE_PARITY.md`, "two specific things to watch":

> **The pedagogy layer** — `utils/theory_anchors.py` (532 loc) and
> `utils/theory_demos.py` (869 loc) are a 19-key registry pair with **no test
> asserting the keys match**, plus a substring-matching fallback that silently
> drops a theory link when a finding string is reworded. It is the most fragile
> intelligent feature in the app and the most likely to quietly not survive a
> rewrite.

Both halves are addressed here, and the order matters: the key test is written
**before** anything is built on the pair, because a registry that drifts under a
new consumer fails in the consumer and gets diagnosed there.

## The fifth substring registry

`infer_theory_anchor` matched three ways and the third scanned `insight.finding`,
which is **prose** — twenty keywords against a sentence. So `"missing"` matched
*"no missing values"* and linked a clean column to the missing-data theory, and a
reworded finding lost its link with nothing saying so.

That is the class `ml/name_registry.py` exists for, after `clinical_units`,
`physiology_reference`, `_KNOWN_UNITS` (`IMPORT-267`) and `ml/triage.py`'s three
local lists (`COACH-034`). The remedy is applied rather than repeated: exact key
or declared alias, and an unknown insight yields silence.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml import name_registry as N                                     # noqa: E402
from utils.theory_anchors import (                                    # noqa: E402
    INSIGHT_CATEGORY_TO_ANCHOR, THEORY_ANCHORS, get_theory_anchor,
    infer_theory_anchor,
)
from utils.theory_demos import DEMO_REGISTRY                          # noqa: E402


class _Insight:
    """The shape `infer_theory_anchor` reads. Only the fields it looks at."""

    def __init__(self, theory_anchor=None, category="", id="", finding=""):
        self.theory_anchor = theory_anchor
        self.category = category
        self.id = id
        self.finding = finding


# ── the key match, which is what FEATURE_PARITY asked for ────────────────────

def test_every_theory_anchor_has_a_demo():
    """A key in the anchors and not the demos is a theory link that opens onto
    nothing."""
    missing = sorted(set(THEORY_ANCHORS) - set(DEMO_REGISTRY))
    assert not missing, (
        f"these anchors have no demo: {missing}. The link renders and the "
        f"panel is empty.")


def test_every_demo_has_a_theory_anchor():
    """And the other direction: a demo nothing can reach is a demo nobody sees.

    Asserted separately rather than as a set equality, because the two failures
    read differently — one is a broken link and the other is dead content, and a
    single assertion would report whichever it found first.
    """
    orphans = sorted(set(DEMO_REGISTRY) - set(THEORY_ANCHORS))
    assert not orphans, (
        f"these demos have no anchor and cannot be reached: {orphans}")


def test_the_pair_is_the_size_the_documentation_claims():
    """19 keys, which is the number `FEATURE_PARITY.md` records.

    Pinned so a later loop can tell "the pair grew" from "the pair drifted" —
    the first is work and the second is the defect.
    """
    assert len(THEORY_ANCHORS) == len(DEMO_REGISTRY) == 19


@pytest.mark.parametrize("key", sorted(THEORY_ANCHORS))
def test_every_anchor_carries_the_fields_a_link_renders(key):
    """One test per key, so a failure names the anchor rather than the set."""
    anchor = get_theory_anchor(key)
    assert anchor is not None
    # The fields the Theory Reference actually renders, read off the registry
    # rather than assumed — `title` and `concept` were the assumption, and the
    # keys are `chapter`, `section`, `why_it_matters`, `what_to_look_for` and
    # `misconception`. A test written from a guess about the schema tests the
    # guess.
    for field in ("chapter", "section", "why_it_matters", "what_to_look_for"):
        assert anchor.get(field), f"{key} has no {field}"
    assert callable(DEMO_REGISTRY[key])


# ── the substring fallback, removed ──────────────────────────────────────────

def test_a_theory_link_is_never_inferred_from_prose():
    """The defect, in one assertion.

    *"There are no missing values here"* contains `"missing"`, and the removed
    scan linked it to the missing-data theory — teaching a concept about a
    problem the finding says the data does not have.
    """
    assert infer_theory_anchor(
        _Insight(finding="There are no missing values here.")) is None
    assert infer_theory_anchor(
        _Insight(finding="Seeds were not varied, so nothing was measured.")) is None
    assert infer_theory_anchor(
        _Insight(finding="Skewness is not a concern for this column.")) is None


def test_an_explicit_anchor_is_honored():
    assert infer_theory_anchor(_Insight(theory_anchor="skewness")) == "skewness"
    # And an explicit anchor that names nothing is refused rather than trusted.
    assert infer_theory_anchor(_Insight(theory_anchor="phlogiston")) is None


@pytest.mark.parametrize("field", ["category", "id"])
@pytest.mark.parametrize("value,expected", [
    ("outliers", "outliers"),
    ("non_normality", "skewness"),          # a declared alias
    ("class_imbalance", "class_imbalance"),
    ("low_sample_size", "sample_size"),     # a declared alias
    ("Feature Scale", "scaling"),           # case and separators only
])
def test_a_declared_spelling_resolves_from_a_structured_field(field, value, expected):
    """Matched on `category` and `id`, which are identifiers, never on prose."""
    assert infer_theory_anchor(_Insight(**{field: value})) == expected


@pytest.mark.parametrize("value", [
    "outliers_are_fine",      # contains `outliers`
    "no_missing_data",        # contains `missing_data`
    "ward_effects",           # nothing declared
    "seedling_counts",        # contains `seed`
])
def test_a_near_miss_yields_silence(value):
    """`None` is an answer: this insight has no theory link. A gap the user can
    see beats a wrong concept beside a right number."""
    assert infer_theory_anchor(_Insight(category=value)) is None


def test_every_alias_resolves_to_a_real_anchor():
    """A declared alias pointing at a key the anchors do not have would render a
    link onto nothing — the same failure as a missing demo, from the other end."""
    for alias, anchor in INSIGHT_CATEGORY_TO_ANCHOR.items():
        assert anchor in THEORY_ANCHORS, f"{alias!r} points at {anchor!r}"


def test_the_registry_uses_the_shared_matching_rule():
    """`ml/name_registry.py` states the rule once, and this registry is the fifth
    member of the class. It uses the helper rather than a sixth local fix."""
    import inspect

    from utils import theory_anchors
    source = inspect.getsource(theory_anchors)
    assert "name_registry" in source, (
        "the fifth registry has its own matching implementation again")
    assert "in finding_lower" not in source, "the prose scan is back"

    # And the rule behaves the same here as it does there.
    lookup = N.build({"skewness": ["skew"]})
    assert N.match("SKEW", lookup) == "skewness"
    assert N.match("skewed_badly", lookup) is None

"""`GUIDED-031` — the one moment a decision is due was the one place with
nothing to act on.

The product owner drove the app, clicked *"Show me what this means"* on a
finding with no proposed repair, and landed in a branch that printed
`suggested_actions` as em-dash bullets and closed with *"the engine reports this
without proposing a repair."* Honest about the engine. A dead end for the user.

And the list was already **options wearing prose**:

    — Consider winsorizing or capping
    — Tree models are robust to outliers
    — Investigate if outliers are errors or genuine

Three different decisions rendered as three paragraphs. `DESIGN_LANGUAGE.md`
§01.4 — *three attributes wearing a sentence costume* — the critique that
started this project, surviving the rewrite by moving branch.

Run:  ./venv/bin/python -m pytest \\
          turbotab/test_a_finding_with_no_repair_still_offers_something.py -q
"""
# The tests here that drove the retired legacy app (turbotab/api.py, the page, the
# figure and manuscript modules, the gates) were removed with it, BLUEPRINT §9.1; git keeps them.
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import actions as A                                       # noqa: E402
from turbotab.project import ProjectError                             # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"
ROOT = Path(__file__).resolve().parents[1]


# ── the classification is complete, and stays complete ───────────────────────

def test_every_suggested_action_the_engine_can_emit_is_classified():
    """The guard that stops a new suggestion arriving as a bullet again.

    Walks `ml/dataset_profile.py` for every literal it can put in
    `suggested_actions` and requires the table to know it. A phrase this does
    not know still renders — as the prose it always was — so the failure is
    a gap rather than a disappearance, and this is what stops the gap being
    permanent.
    """
    tree = ast.parse((ROOT / "ml" / "dataset_profile.py").read_text())
    phrases = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and getattr(node.func, "id", "") == "DataWarning"):
            continue
        for kw in node.keywords:
            if kw.arg != "suggested_actions" or not isinstance(kw.value, ast.List):
                continue
            for element in kw.value.elts:
                if isinstance(element, ast.Constant) and isinstance(element.value, str):
                    phrases.append(element.value)

    assert len(phrases) > 20, "the walk found nothing; the search is wrong"
    unknown = sorted({p for p in phrases if A.classify(p) is None})
    assert not unknown, (
        "the engine can emit these and the table does not classify them:\n  "
        + "\n  ".join(unknown)
        + "\n\nEach is either an OPERATION the app can do and preview, or an "
          "EARMARK that goes to a person or to a later step. Deciding which is "
          "the work; leaving it unclassified renders it as a bullet again.")


def test_the_two_kinds_are_never_confused():
    """An operation claims the app can do this. An earmark claims it cannot.

    Getting one wrong in either direction is the governing rule broken: an
    earmark rendered as an operation offers a control that does nothing, and an
    operation rendered as an earmark hides a thing the app could have done.
    """
    assert isinstance(A.classify("Consider winsorizing or capping"), A.Operation)
    assert isinstance(A.classify("Verify units and data entry"), A.Earmark)
    assert A.classify("Verify units and data entry").is_for_a_person
    assert not A.classify("Tree models are robust to outliers").is_for_a_person


def test_the_same_phrase_with_different_examples_classifies_once():
    """*"Use regularized linear models (Ridge, Lasso)"* and *"Use regularized
    models (Ridge, Lasso, ElasticNet)"* are one suggestion with two example
    lists. A table that distinguished them would classify one decision twice
    and let the two drift."""
    a = A.classify("Use regularized linear models (Ridge, Lasso)")
    b = A.classify("Use regularized models (Ridge, Lasso, ElasticNet)")
    assert a is not None and a.key == b.key == "prefer_regularized"


def test_every_operation_binds_to_a_catalogue_that_resolves_it():
    """A binding the catalogue does not carry is an option that cannot preview
    and cannot execute — a control that does nothing, which is worse than the
    bullet it replaced."""
    from turbotab import features as F, missingness as M, recipes as R, selection as S
    for phrase, found in A.known_phrases().items():
        if not isinstance(found, A.Operation):
            continue
        if found.catalogue == A.FEATURE:
            assert F.get(found.binding) is not None
        elif found.catalogue == A.MISSINGNESS:
            assert M.strategy(found.binding) is not None
        elif found.catalogue == A.RECIPE:
            op = R.operation(found.binding)
            assert found.variant in op.variants, (
                f"{found.key}: {found.variant!r} is not a variant of "
                f"{found.binding!r}")
        elif found.catalogue == A.SELECTION:
            assert found.binding in S.METHODS
        else:                                          # pragma: no cover
            pytest.fail(f"{found.key} names an unknown catalogue")


def test_every_earmark_names_a_step_that_exists_or_a_person():
    from ml.router import STEP_LABELS
    for found in A.known_phrases().values():
        if not isinstance(found, A.Earmark):
            continue
        assert found.target_step == A.YOU or found.target_step in STEP_LABELS, (
            f"{found.key} resurfaces at {found.target_step!r}, which is not a "
            f"step — an earmark with nowhere to go is a discard with manners")

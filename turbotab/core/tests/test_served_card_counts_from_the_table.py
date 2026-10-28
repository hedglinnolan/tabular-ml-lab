"""The proposals stage hands the estimand card the table's column summaries (``table_info``), so
the family card served counts the tests feature-wise regression would make from each member's
type, as the preview does (the verifier's V4a3-1; T4a's patch to ``proposals.py``).

On the base the stage handed the card no columns, so a member whose type no answer recorded (a
float column, a text one) was not known: an ordinary numeric family's multiplicity options lost
their count ("among the 2 tests" became "among the tests"), and a family with a text member was
served ``n_tests`` 3 where the preview and the fit make 2.

Truth from construction: {`fiber`, `supplement`}, both numbers, makes 2 tests, and its options are
the base's words before T4a, byte for byte; {`fiber`, `grp`, `supplement`}, `grp` drawn as text,
makes the 2 the preview divides by (``ctx.read["multiplicity"]["m"]``), naming `grp`.
"""
from __future__ import annotations

import numpy as np
import pytest

from turbotab.core import decisions as d

BH = "Benjamini & Hochberg 1995, J R Stat Soc B 57:289"
NUMERIC = {
    "fdr_bh": (f"controls the expected share of false discoveries among the 2 tests ({BH}), and "
               f"every member is still shown"),
    "count_stated": ("sound for a few hypotheses each declared on its own, read with all 2 tests "
                     "in view (STROBE item 20: \"multiplicity of analyses\")")}
NUMERIC_TENSION = ("For a few declared nutrient hypotheses the field reports unadjusted p-values "
                   "with the count stated (Rothman 1990, Epidemiology 1:43); Benjamini–Hochberg "
                   "controls the false-discovery rate across all 2 of them.")


def _project(folder, text: bool):
    from turbotab.core.tests.acceptance import preview_fixtures as F
    from turbotab.core.tests.acceptance.preview_harness import Project

    frame = F.estimand_table()
    base = F.family_state()
    roles = dict(base.roles)
    if text:
        frame["grp"] = np.random.default_rng(5).choice(["a", "b", "c"], len(frame))
        roles["grp"] = "exposure"
    state = base.model_copy(update={"roles": roles, "role_confirmations": dict(roles)})
    return Project(frame, folder, state, upto=["proposals", "cohort", "split"])


@pytest.fixture(scope="module", params=[False, True], ids=["numeric", "text"])
def served(request, tmp_path_factory):
    project = _project(tmp_path_factory.mktemp("served"), request.param)
    art = project.artifacts["proposals"]
    yield request.param, project, getattr(art, "data", art)["estimand"]["family"]
    project.close()


def _reasons(family: dict) -> dict:
    return {o["key"]: o["sound"]["reason"] for o in family["multiplicity"]["options"]}


def test_the_served_family_card_counts_what_feature_wise_regression_would_test(served):
    """Numeric members: the served options and tension count the 2 tests, byte for byte as before
    T4a, where they read "among the tests". A text member: the served count is the preview's, 2,
    and the options name `grp` as a category, where the card served 3."""
    text, project, family = served
    q = family["multiplicity"]
    if not text:
        assert (_reasons(family), q["tension"], q["n_tests"]) == (NUMERIC, NUMERIC_TENSION, 2)
        return
    _, ctx = project.preview(d.SetMultiplicity(method="bh"))
    assert q["n_tests"] == ctx.read["multiplicity"]["m"] == 2
    named = "; `grp` is a category, adjusted for, not tested"
    assert _reasons(family) == {
        "fdr_bh": (f"controls the expected share of false discoveries among the 2 tests ({BH}), "
                   f"and every member tested is still shown{named}"),
        "count_stated": ("sound for a few hypotheses each declared on its own, read with all 2 "
                         "tests in view (STROBE item 20: \"multiplicity of analyses\")" + named)}
    assert q["tension"] == (
        "For a few declared nutrient hypotheses the field reports unadjusted p-values with the "
        "count stated (Rothman 1990, Epidemiology 1:43); Benjamini–Hochberg controls the "
        f"false-discovery rate across all 2 of them{named}.")

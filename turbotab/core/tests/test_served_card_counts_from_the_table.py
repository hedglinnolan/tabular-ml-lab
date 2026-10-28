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


# ── a whole-number member whose codes-or-amounts answer is pending (the verifier's T5bV-2) ──
#
# `visits` holds the whole numbers 0 to 7, which may be codes for categories (indicators,
# adjusted for, not tested) or amounts (one column, tested): the fit waits for that answer
# (``readings.predictors_or_ask``), so before it is recorded the card does not know whether the
# family makes 2 tests or 3. It served 3, and the options counted them; now it keeps the
# uncounted words, as for any member whose type is not known, and counts 2 once `visits` is
# recorded as codes and 3 once as amounts. Stored as integers or as floats (a blank turns codes
# 1-5 into 1.0-5.0), as the fit reads the values either way.

UNCOUNTED = {
    "fdr_bh": (f"controls the expected share of false discoveries among the tests ({BH}), and "
               f"every member tested is still shown"),
    "count_stated": ("sound for a few hypotheses each declared on its own, read with every test "
                     "in view (STROBE item 20: \"multiplicity of analyses\")")}
UNCOUNTED_TENSION = ("For a few declared nutrient hypotheses the field reports unadjusted p-values "
                     "with the count stated (Rothman 1990, Epidemiology 1:43); Benjamini–Hochberg "
                     "controls the false-discovery rate across the tests made.")


def _visits(folder, stored: str, answer: str | None):
    """{`fiber`, `supplement`, `visits`}, `visits` the integers 0 to 7 (``stored`` as "integer"
    or "float"), its codes-or-amounts answer ``answer`` (None: pending)."""
    from turbotab.core.tests.acceptance import preview_fixtures as F
    from turbotab.core.tests.acceptance.preview_harness import Project

    frame = F.estimand_table()
    visits = np.random.default_rng(5).integers(0, 8, len(frame))
    frame["visits"] = visits.astype(float) if stored == "float" else visits
    base = F.family_state()
    roles = {**base.roles, "visits": "exposure"}
    shapes = dict(base.shape_confirmations)
    if answer is not None:
        shapes["code_or_count:visits"] = answer
    state = base.model_copy(update={"roles": roles, "role_confirmations": dict(roles),
                                    "shape_confirmations": shapes})
    return Project(frame, folder, state, upto=["proposals", "cohort", "split"])


@pytest.mark.parametrize("stored", ["integer", "float"])
def test_a_member_waiting_for_its_codes_or_amounts_answer_is_not_counted(tmp_path, stored):
    """Pending: the served options and tension are the uncounted words, where they counted 3
    tests. The premise: the column summary's type is ``stored``'s, and the values are whole."""
    project = _visits(tmp_path, stored, None)
    try:
        [info] = [c for c in project.store.info().columns if c.name == "visits"]
        assert info.dtype == ("integer" if stored == "integer" else "numeric")
        art = project.artifacts["proposals"]
        family = getattr(art, "data", art)["estimand"]["family"]
    finally:
        project.close()
    assert (_reasons(family), family["multiplicity"]["tension"]) == (UNCOUNTED, UNCOUNTED_TENSION)


@pytest.mark.parametrize("answer", ["code", "amount"])
def test_a_member_answered_codes_or_amounts_is_counted_as_the_answer_says(tmp_path, answer):
    """Silent once answered: as codes, `visits` enters as indicators, adjusted for, so the card
    counts 2 tests and names it; as amounts it is one column, tested, so 3, every member shown."""
    project = _visits(tmp_path, "integer", answer)
    try:
        art = project.artifacts["proposals"]
        family = getattr(art, "data", art)["estimand"]["family"]
    finally:
        project.close()
    q = family["multiplicity"]
    if answer == "amount":
        assert q["n_tests"] == 3
        assert _reasons(family) == {
            "fdr_bh": (f"controls the expected share of false discoveries among the 3 tests ({BH}), "
                       f"and every member is still shown"),
            "count_stated": ("sound for a few hypotheses each declared on its own, read with all 3 "
                             "tests in view (STROBE item 20: \"multiplicity of analyses\")")}
        return
    named = "; `visits` is recorded as a category, adjusted for, not tested"
    assert q["n_tests"] == 2
    assert _reasons(family) == {
        "fdr_bh": (f"controls the expected share of false discoveries among the 2 tests ({BH}), "
                   f"and every member tested is still shown{named}"),
        "count_stated": ("sound for a few hypotheses each declared on its own, read with all 2 "
                         "tests in view (STROBE item 20: \"multiplicity of analyses\")" + named)}

"""P1-FU on the server (WAVE_C6A_PLAN §3, the integrator's remaining issues on feat/wave-c6a1-int).

2. The triage's measured noticings reach a project under any lens: the K5 noticing (structure among
   the adjustment terms) is not dietary, so it is measured on a clinical project too, while the
   dietary proof's noticings stay with the dietary lens. Expected threads are the catalog's names,
   written here.
3. The quest route says what the log does after Fit: under Estimate and Describe, Results counts
   its exhibits and Write-up waits for Results to be placed, so a reached stage that asks nothing
   is complete except there.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from turbotab.core.tests.test_collinear_adjustment import ADJUST, _Store, state, table
from turbotab.server.service import ProjectService

COLLINEAR = "shared-collinear-predictors"
DIETARY = {"diet-energy-carries-the-nutrient", "diet-implausible-reporters",
           "diet-day-to-day-variance"}


def _service(frame):
    """The parts of the service ``_noticed`` reads: the table, its columns, no working table."""
    ingest = {"columns": [{"name": c} for c in frame.columns]}
    return SimpleNamespace(store=lambda pid: _Store(frame), table_source=lambda pid: None,
                           _artifact=lambda pid, *a: None, _shown=lambda pid, name: ingest)


@pytest.mark.parametrize("lens", [["clinical"], []])
def test_the_k5_noticing_reaches_the_triage_without_the_dietary_lens(lens):
    f = table()
    noticed = ProjectService._noticed(_service(f), "p", state(ADJUST, lens=lens))
    assert [n.thread for n in noticed] == [COLLINEAR]
    assert noticed[0].family == "K5"


def test_the_dietary_noticings_stay_dietary():
    f = table()
    clinical = {n.thread for n in ProjectService._noticed(_service(f), "p",
                                                          state(ADJUST, lens=["clinical"]))}
    assert not clinical & DIETARY
    dietary = {n.thread for n in ProjectService._noticed(_service(f), "p",
                                                         state(ADJUST, lens=["dietary"]))}
    assert COLLINEAR in dietary and dietary & DIETARY


def test_the_quest_route_says_results_and_write_up_are_the_exception():
    from turbotab.server.openapi import build

    said = " ".join(build()["paths"]["/api/projects/{pid}/quest"]["get"]["description"].split())
    assert "0 of 0, is complete" in said
    assert ("except Results under Estimate and Describe, which counts the exhibits the engine "
            "serves after Fit") in said, said
    assert "Write-up is not reached until Results is placed" in said, said

"""SMC-FCS's Cox refusal names the stratum with no death when it is the reference stratum, and its
ways forward are decisions that run (the verifier's T2-B-V4-1, T2-B-open-0 and T2-B-open-1; the
orchestrator's rulings (d) and (f) on the swarm's tail).

The survey design's strata enter the imputation's model as an indicator for each stratum but the
first, the reference (``missing._DesignTerms``). Where no row of the reference stratum has the
event, no indicator of it orders the risk sets: the fit fails along a combination of the other
strata's indicators, or along a PSU indicator that marks rows of that stratum only, and the words
named those, never the stratum with no death (14 strata that all have deaths, in the order their
labels sort as text; or "an indicator of the PSUs ... (as when no row of those PSUs has the
event)"). Now the words say which of the design's groups no row with the event is in, counted on
every row, the reference stratum among them, and list several indicators in the strata's own
order. A stratum other than the reference keeps its words (``test_mi_cox_design_refusal``).

The ways forward: the refusal names the sample answer, and the stage now offers it as an exit
read from the refusal, not words alone, which runs; complete cases only where the analysis model
has a maximum on them, so where no woman dies (on the complete cases too) they are not offered,
and the table they would give is refused.

**Truth.** By construction (``nhanes_mortality(seed=5, n_per_psu=35)``, kcal 15% blank): every
death of stratum 101, the lowest stratum (the reference), is set to no death, with its three PSUs
and with its third merged into its first (two); or every woman's. Which indicators the solver's
ordering runs along is the solver's, so it is not pinned; the counted groups are.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbotab.core.decisions import MissingSpec
from turbotab.core.methods.missing import WITHOUT_THE_DESIGN
from turbotab.core.models.survey import SAMPLE_EXIT, SurveyDesign
from turbotab.core.tests.test_mi_cox_design_refusal import (COMPLETE_CASES, _fitted, _mortality,
                                                            _table, four_copies)  # noqa: F401

LEAD = ("Multiple imputation cannot run on these data: The Cox model the imputations are drawn "
        "from has no maximum on a completed copy of the data, so its coefficients cannot be "
        "drawn: to carry the survey design into the copies, it holds ")
TAIL = (" The analysis model itself is not the one that fails: without the survey design's terms, "
        "its partial likelihood has a maximum on the same copy. " + WITHOUT_THE_DESIGN)
COUNTED = " (no row of stratum 101 of `SDMVSTRA` has the event)"
SAMPLE = {"label": SAMPLE_EXIT, "decision": {"kind": "set_survey", "estimand": "sample"}}


def _reference_quiet(psus: int) -> pd.DataFrame:
    """Stratum 101's deaths set to none; ``psus`` 2 merges its third PSU into its first."""
    frame = _mortality(None)
    at = frame["SDMVSTRA"] == 101
    assert sorted(frame.loc[at, "SDMVPSU"].unique()) == [1, 2, 3]
    assert frame["SDMVSTRA"].min() == 101  # the reference: the first of the strata's own values
    if psus == 2:
        frame.loc[at & (frame["SDMVPSU"] == 3), "SDMVPSU"] = 1
    frame.loc[at, "mortstat"] = 0
    eligible = frame["eligible"] == 1
    deaths = frame[eligible].groupby("SDMVSTRA")["mortstat"].sum()
    assert deaths[101] == 0 and (deaths.drop(101) > 0).all()
    return frame


@pytest.mark.parametrize("psus", [3, 2], ids=["three PSUs", "two PSUs"])
def test_the_reference_stratum_with_no_death_is_named(psus, tmp_path):
    inf = _table(_reference_quiet(psus), tmp_path)["inference"]
    words = inf["refused"]
    assert words.startswith(LEAD) and words.endswith(TAIL), words
    assert COUNTED in words, words
    held = words[len(LEAD):words.index(" beside the analysis model's terms")]
    if psus == 2:
        # several strata's indicators, in the strata's own order
        named = [int(w.split()[0]) for w in held.split("the indicator of stratum ")[1:]]
        assert named and named == sorted(named) and 101 not in named
    assert "as when no row of those PSUs has the event" not in words
    assert inf["exits"][-1] == SAMPLE


def test_the_sample_answer_the_refusal_offers_runs(tmp_path):
    frame = _reference_quiet(3)
    inf = _table(frame, tmp_path / "mi")["inference"]
    assert [e["label"] for e in inf["exits"]] == [COMPLETE_CASES, SAMPLE_EXIT]
    assert inf["exits"][1]["decision"] == SAMPLE["decision"]
    _fitted(_table(frame, tmp_path / "sample", estimand="sample"), "Cox proportional hazards",
            "multiple_imputation")
    # and complete cases, offered, give the survey-weighted table on the rows with kcal recorded
    _fitted(_table(frame, tmp_path / "cc", complete=True,
                   missing=MissingSpec(strategy="complete_case")),
            "survey-weighted Cox", "complete_case")


def test_complete_cases_are_not_offered_where_the_analysis_model_has_no_maximum_on_them(tmp_path):
    """No woman dies, so none dies on the complete cases either: the analysis table on them is
    refused (no interval), and the refusal offers only to leave `female` out (which runs:
    ``test_mi_cox_design_refusal``)."""
    frame = _mortality("women")
    inf = _table(frame, tmp_path / "mi")["inference"]
    assert [e["label"] for e in inf["exits"]] == ["Leave `female` out"]
    complete = _table(frame, tmp_path / "cc", complete=True,
                      missing=MissingSpec(strategy="complete_case"))
    assert complete["inference"]["refused"]
    assert all(c.get("ci_low") is None and c.get("p") is None
               for c in complete.get("coefficients") or [])


def _terms(stratum: list[int], psu: list[int], codes: tuple = (101, 103, 107)) -> object:
    """The design's levels as ``design_variables`` writes them for each row, named by its words."""
    from turbotab.core.methods.missing import DesignWords, _DesignTerms

    design = SurveyDesign(row_ids=np.arange(len(stratum)), weight=np.ones(len(stratum)),
                          stratum=np.asarray(stratum), psu=np.arange(len(stratum)),
                          stratum_labels=codes, weight_column="W", strata_column="S",
                          psu_column="P")
    data = pd.DataFrame({"__stratum": [f"stratum {s}" for s in stratum],
                         "__psu": [f"psu {p}" for p in psu]})
    return _DesignTerms(data, ["__stratum", "__psu"], DesignWords(design))


def test_the_groups_with_no_event_are_counted_on_every_row():
    """Codes 0, 1, 2 are strata 101, 103 and 107. Stratum 101 (the reference) has no event; in
    stratum 103 one of its two PSUs has none; stratum 107 has an event in each PSU. Counted by
    hand from the rows written here."""
    stratum = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    psu = [1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2]
    event = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0], dtype=bool)
    assert _terms(stratum, psu).quiet(event) == ["stratum 101 of `S`",
                                                 "one PSU of stratum 103 of `S`"]
    # silent: an event in every stratum and every PSU
    assert _terms(stratum, psu).quiet(np.array([1, 0, 1, 0] * 3, dtype=bool)) == []
    # the strata's own order, not their levels' order as text ("stratum 10" before "stratum 2")
    many = _terms(list(range(11)), [1] * 11, codes=tuple(range(100, 111)))
    assert many.quiet(np.zeros(11, dtype=bool)) == [f"stratum {v} of `S`"
                                                    for v in range(100, 111)]

"""Fit waits for the open noticings' triage under Estimate and Describe (calm/FOUNDATION §7): the
triage of the open noticings comes before the plan's lock, and pressing Fit records the lock, so a
press is refused while the triage is not confirmed (``fit_press.triage_refusal``).

Each triage here is built by hand, so what it holds is known by construction: which noticings
block, which carry a recorded disposition, whether the gate has passed. The refusal's words and
ways forward are written out here; the confirmation its way forward records is the triage's own
"Confirm all" (``sweep._confirm_the_triage``: each noticing as recorded while that stands, else as
recommended).

* It fires while a noticing blocks (each is decided on its own first) and while one has no recorded
  disposition (the triage in Models, or its "Confirm all" as recommended), one noticing or several.
* It is silent once every disposition is recorded, "act on it" among them (the gate is the
  confirmation, FOUNDATION §7), with nothing to triage, and once the gate has passed.
* It names each noticing where no summary can be misread: one after a colon, several each in a
  numbered sentence of its own, with its recommended disposition in a sentence after it, so a
  summary's own semicolon and a label's own parentheses stay inside their noticing.
"""
from __future__ import annotations

import pytest

from turbotab.core import fit_press
from turbotab.core.decisions import Refusal, parse_decision
from turbotab.core.sweep import Triage, TriageItem

LABELS = {"no_change": "Doesn't change your numbers here", "could_bias": "Could bias the estimate",
          "act_on_it": "Act on it"}
CONFIRM = parse_decision({"kind": "confirm_sweep", "stage": "models", "sweep": "noticings",
                          "lines": []}).model_dump(mode="json")
LEAD = ("Fit waits: pressing it locks the analysis plan, and the open noticings are triaged before "
        "the plan is locked.")


def item(fid: str, summary: str, recommended: str = "no_change", *, blocker: bool = False,
         recorded: str | None = None, label: str | None = None) -> TriageItem:
    return TriageItem(id=fid, line=f"finding:{fid}", stage="models", summary=summary,
                      severity="critical" if blocker else "info", recommended=recommended,
                      label=label or LABELS[recommended], reason="the reason", blocker=blocker,
                      recorded=recorded)


def triage(*items: TriageItem, passed: bool = False) -> Triage:
    blockers = sum(i.blocker for i in items)
    return Triage(stage="models", gate="gate:open-noticings-before-lock", items=list(items),
                  blockers=blockers, confirmable=bool(items) and not blockers and not passed,
                  answered=not blockers and all(i.recorded is not None for i in items),
                  passed=passed)


def test_one_noticing_with_no_recorded_disposition_holds_fit_with_the_triage_as_the_way_forward():
    refused = fit_press.triage_refusal(triage(
        item("binary_text__gender", "`gender` is two-level text (`female`, `male`)."),
        item("diet-day-to-day-variance", "Part of `sugar`'s spread is day-to-day noise",
             "could_bias", recorded="could_bias")))
    assert isinstance(refused, Refusal) and refused.code == "triage_open"
    assert refused.message == (
        f"{LEAD} 1 noticing is still open: `gender` is two-level text (`female`, `male`). "
        f"Recommended: doesn't change your numbers here. Triage it in Models first, so it is "
        f"decided before any estimate is seen.")
    assert refused.exits == [{"label": "Triage the open noticings in Models", "decision": None},
                             {"label": "Keep the recommended dispositions", "decision": CONFIRM}]


def test_several_open_noticings_are_each_named_with_their_recommended_disposition():
    refused = fit_press.triage_refusal(triage(
        item("sas_zeros", "Cells hold 5.4e-79: a zero in a SAS transport file, misread.",
             "act_on_it"),
        item("diet-implausible-reporters", "Some energy reports are implausible", "could_bias")))
    assert refused is not None and refused.code == "triage_open"
    assert refused.message == (
        f"{LEAD} 2 noticings are still open. (1) Cells hold 5.4e-79: a zero in a SAS transport "
        f"file, misread. Recommended: act on it. (2) Some energy reports are implausible. "
        f"Recommended: could bias the estimate. Triage them in Models first, so each is decided "
        f"before any estimate is seen.")
    assert refused.exits[1] == {"label": "Keep the recommended dispositions", "decision": CONFIRM}


def test_a_summary_s_own_semicolon_and_a_label_s_parentheses_stay_inside_their_noticing():
    """The verifier's NHANES words: a summary holding a semicolon read as two noticings once the
    noticings were joined by semicolons, and a label ending in parentheses nested inside the
    parentheses that held it. Each noticing is now its own numbered sentence, its recommended
    disposition a sentence of its own."""
    refused = fit_press.triage_refusal(triage(
        item("binary_text__gender", "`gender` is two-level text (`female`, `male`); which level "
             "counts as 1 is not asked yet"),
        item("diet-day-to-day-variance", "Part of `sugar`'s spread is day-to-day noise",
             "could_bias", label="Could bias the estimate (not measurable here)"),
        item("diet-implausible-reporters",
             "Some energy reports are implausible, and the screen removes them.", "could_bias")))
    assert refused is not None and refused.code == "triage_open"
    assert refused.message == (
        f"{LEAD} 3 noticings are still open. (1) `gender` is two-level text (`female`, `male`); "
        f"which level counts as 1 is not asked yet. Recommended: doesn't change your numbers "
        f"here. (2) Part of `sugar`'s spread is day-to-day noise. Recommended: could bias the "
        f"estimate (not measurable here). (3) Some energy reports are implausible, and the screen "
        f"removes them. Recommended: could bias the estimate. Triage them in Models first, so "
        f"each is decided before any estimate is seen.")
    assert "(could bias" not in refused.message and "))" not in refused.message


def test_a_blocking_noticing_is_decided_on_its_own_before_the_rest_are_triaged():
    refused = fit_press.triage_refusal(triage(
        item("outcome_constant", "`glucose` takes one value", blocker=True),
        item("binary_text__gender", "`gender` is two-level text")))
    assert refused is not None and refused.code == "triage_open"
    assert refused.message == f"{LEAD} 1 noticing must be resolved first: `glucose` takes one value."
    assert refused.exits == [{"label": "Decide: `glucose` takes one value", "decision": None}]
    two = fit_press.triage_refusal(triage(item("a", "A.", blocker=True),
                                          item("b", "B", blocker=True, recorded="no_change")))
    assert two.message == f"{LEAD} 2 noticings must be resolved first. (1) A. (2) B."
    assert [e["label"] for e in two.exits] == ["Decide: A.", "Decide: B"]


@pytest.mark.parametrize("recorded", ["no_change", "could_bias", "act_on_it"])
def test_a_confirmed_triage_holds_nothing_whatever_disposition_it_recorded(recorded):
    assert fit_press.triage_refusal(triage(
        item("binary_text__gender", "`gender` is two-level text", recorded="no_change"),
        item("sas_zeros", "Cells hold 5.4e-79", "act_on_it", recorded=recorded))) is None


def test_nothing_to_triage_holds_nothing():
    assert fit_press.triage_refusal(triage()) is None
    assert fit_press.triage_refusal(None) is None


def test_once_the_gate_has_passed_the_triage_holds_nothing():
    """The plan fixed (a press after the lock): the triage is read, no longer confirmed."""
    assert fit_press.triage_refusal(triage(item("a", "A"), item("b", "B", blocker=True),
                                           passed=True)) is None

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
* It fires while a noticing recorded "act on it" is still listed, not acted on (the owner's Q8,
  2026-11-04: Fit holds until each is acted on or its disposition changed): each is named with
  where it is acted on and a way forward that changes its disposition, recording it as a
  limitation. Built by the engine's own triage, a kept "act on it" holds the press, and taking
  that way forward, or acting on the noticing (its own decision answers it, so it drops out of the
  triage), releases it.
* It is silent once every disposition is recorded and none is "act on it" still listed, with
  nothing to triage, and once the gate has passed.
* It names each noticing where no summary can be misread: one after a colon, several each in a
  numbered sentence of its own, with its recommended disposition in a sentence after it, so a
  summary's own semicolon and a label's own parentheses stay inside their noticing.
* Each way forward names the card it opens (``target``, G12; the owner's Q3, 2026-11-04: no dead
  end): the triage in its stage; a blocking noticing, and one kept to act on, at the question it
  routes to, else at its own line where the quest log serves one (a finding's), else in its stage;
  and "Record as a limitation", a decision, the triage's card, which records it (the
  tapestry, where Fit sits, records nothing: FOUNDATION §1). "Keep the recommended dispositions"
  names none: the triage's way forward beside it opens the same card. Built by the engine's own
  triage, each target names a stage and a line the quest log serves.
* The reference journeys' export, pressing Fit again where it waits for the lock, takes its way
  forward "Keep the recommended dispositions" as a person pressing Confirm all does, recorded as
  that way forward's, and presses again, and so on through each refusal that offers a way forward
  recording a disposition ("Record as a limitation" for a noticing marked "act on it" and not
  acted on); a blocking noticing's refusal stands, nothing posted. The server there is one written
  here that serves the refusals built from triages written here.
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
# The card the triage is drawn on: its gate's stage (each triage here is Models', as the engine's
# is under Estimate), on its open line, as the quest log keys a stage (``exit_target.toward``).
AT_TRIAGE = {"stage": "models", "key": None}


def item(fid: str, summary: str, recommended: str = "no_change", *, blocker: bool = False,
         recorded: str | None = None, label: str | None = None, stage: str = "models",
         question: str | None = None) -> TriageItem:
    return TriageItem(id=fid, line=f"finding:{fid}", stage=stage, summary=summary,
                      severity="critical" if blocker else "info", recommended=recommended,
                      label=label or LABELS[recommended], reason="the reason", blocker=blocker,
                      recorded=recorded, question=question)


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
    assert refused.exits == [{"label": "Triage the open noticings in Models", "decision": None,
                              "target": AT_TRIAGE},
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
    # It is decided on its own line, in the stage that serves it (`item`'s stage, Models).
    assert refused.exits == [{"label": "Decide: `glucose` takes one value", "decision": None,
                              "target": {"stage": "models", "key": "outcome_constant"}}]
    two = fit_press.triage_refusal(triage(item("a", "A.", blocker=True),
                                          item("b", "B", blocker=True, recorded="no_change")))
    assert two.message == f"{LEAD} 2 noticings must be resolved first. (1) A. (2) B."
    assert [e["label"] for e in two.exits] == ["Decide: A.", "Decide: B"]


def test_built_by_the_engine_a_measured_blocker_opens_the_question_it_is_decided_at():
    """The other way for where a blocker is decided: a measured noticing (``sweep._noticing_item``:
    its line ``noticing:<thread>``) is no line of the quest log, so its way forward opens the
    question it routes to, a line the log serves, not a key the log does not hold. Built by the
    engine, on the exact Atwater identity (kcal the sum of its parts: a blocker, decided at the
    adjustment question; test_collinear_adjustment.py), the stage that serves that line is read
    off the engine's own quest log."""
    from turbotab.core import sweep
    from turbotab.core.tests.test_collinear_adjustment import ADJUST, THREAD, noticing, state, table
    from turbotab.core.tests.test_confirm_sweep import log_of

    f = table(exact=True)
    s = state(ADJUST)
    log = log_of(s, findings={"findings": []}, columns=list(f.columns))
    t = sweep.triage(s, log, {"findings": []}, noticed=[noticing(f)])
    blocker = next(i for i in t.items if i.id == THREAD)
    assert blocker.blocker and blocker.line == f"noticing:{THREAD}"
    assert blocker.question == "adjustment"
    served = {st.key: {line.key for line in st.lines} for st in log.stages}
    assert all(THREAD not in keys for keys in served.values())
    (at,) = [stage for stage, keys in served.items() if "adjustment" in keys]
    refused = fit_press.triage_refusal(t)
    assert refused is not None and refused.code == "triage_open"
    (decide,) = refused.exits
    assert decide["label"].startswith("Decide: ") and decide["decision"] is None
    assert decide["target"] == {"stage": at, "key": "adjustment"}
    assert decide["target"]["key"] in served[decide["target"]["stage"]]


@pytest.mark.parametrize("recorded", ["no_change", "could_bias"])
def test_a_confirmed_triage_holds_nothing_where_no_noticing_listed_is_marked_act_on_it(recorded):
    """Silent: every disposition recorded, none of them "act on it" (one recommended to act on,
    recorded otherwise by the person, included)."""
    assert fit_press.triage_refusal(triage(
        item("binary_text__gender", "`gender` is two-level text", recorded="no_change"),
        item("sas_zeros", "Cells hold 5.4e-79", "act_on_it", recorded=recorded))) is None


# ── a noticing marked "act on it" and not acted on (the owner's Q8) ──────────

ACT_LEAD = "Fit waits: pressing it locks the analysis plan, and"
SAS = "`152` cells in `9` columns hold `5.4e-79`: a zero in a SAS transport file, misread."


def keeping(*lines: dict) -> dict:
    """The triage's own confirmation of ``lines`` (each noticing's id, key and value), as the
    server stores a way forward's decision."""
    return parse_decision({"kind": "confirm_sweep", "stage": "models", "sweep": "noticings",
                           "lines": list(lines)}).model_dump(mode="json")


def test_a_noticing_kept_to_act_on_and_not_acted_on_holds_fit_with_where_and_a_change():
    """Fires: every disposition is recorded, one of them "act on it", and that noticing is still
    listed, so it is not acted on (a noticing acted on drops out of the triage: ``sweep.triage``).
    The refusal names it, says where it is acted on (its stage, Your data), and offers changing its
    disposition, recorded as a limitation (the triage's own confirmation of that one line, "could
    bias the estimate"); the other noticing's disposition is untouched. Acting on it opens its own
    line in Your data, a finding's, where its repair is chosen; recording it as a limitation opens
    the triage's card, which records that decision."""
    refused = fit_press.triage_refusal(triage(
        item("binary_text__gender", "`gender` is two-level text", recorded="no_change"),
        item("sas_zeros", SAS, "act_on_it", recorded="act_on_it", stage="data")))
    assert isinstance(refused, Refusal) and refused.code == "triage_open"
    assert refused.message == (
        f"{ACT_LEAD} 1 noticing marked \"act on it\" is not acted on yet: {SAS} Act on it where "
        f"it is decided, or change its disposition by recording it as a limitation, before any "
        f"estimate is seen.")
    assert refused.exits == [
        {"label": f"Act on it in Your data: {SAS}", "decision": None,
         "target": {"stage": "data", "key": "sas_zeros"}},
        {"label": f"Record as a limitation: {SAS}",
         "decision": keeping({"id": "finding:sas_zeros", "key": "sas_zeros",
                              "value": "could_bias"}),
         "target": AT_TRIAGE}]


def test_several_kept_to_act_on_are_each_named_with_the_question_or_stage_that_acts_on_it():
    """Fires, several: each in a numbered sentence of its own, in the triage's order, each with its
    two ways forward; one routed to a question is acted on at that question, in that question's
    stage (the eligibility question, in Who's in), one with none in its own stage."""
    refused = fit_press.triage_refusal(triage(
        item("sas_zeros", SAS, "act_on_it", recorded="act_on_it", stage="data"),
        item("pack::dietary::implausible_intake", "Some sugar reports are implausible",
             "act_on_it", recorded="act_on_it", stage="whos_in", question="exclusions"),
        item("diet-day-to-day-variance", "Part of `sugar`'s spread is day-to-day noise",
             "could_bias", recorded="could_bias")))
    assert refused is not None and refused.code == "triage_open"
    assert refused.message == (
        f"{ACT_LEAD} 2 noticings marked \"act on it\" are not acted on yet. (1) {SAS} (2) Some "
        f"sugar reports are implausible. Act on each where it is decided, or change its "
        f"disposition by recording it as a limitation, before any estimate is seen.")
    assert [e["label"] for e in refused.exits] == [
        f"Act on it in Your data: {SAS}", f"Record as a limitation: {SAS}",
        "Act on it at the eligibility question in Who's in: Some sugar reports are implausible",
        "Record as a limitation: Some sugar reports are implausible"]
    assert [e["decision"] for e in refused.exits] == [
        None, keeping({"id": "finding:sas_zeros", "key": "sas_zeros", "value": "could_bias"}),
        None, keeping({"id": "finding:pack::dietary::implausible_intake",
                       "key": "pack::dietary::implausible_intake", "value": "could_bias"})]
    # The one routed to a question opens that question's line in its stage, as its label says;
    # the other its own line; each "Record as a limitation" the triage's card.
    assert [e["target"] for e in refused.exits] == [
        {"stage": "data", "key": "sas_zeros"}, AT_TRIAGE,
        {"stage": "whos_in", "key": "exclusions"}, AT_TRIAGE]


def test_a_measured_noticing_kept_to_act_on_opens_its_stage_where_no_line_of_its_own_is_served():
    """Both ways for where acting on it opens: a measured noticing (``sweep._noticing_item``: its
    line ``noticing:<thread>``) is no line of the quest log, so with no question to route to its
    way forward opens its stage on its open line, not a line the log does not serve; one whose
    stage the log does not name opens nothing, and says so in words."""
    measured = TriageItem(id="diet-day-to-day-variance", line="noticing:diet-day-to-day-variance",
                          stage="models", summary="Part of `sugar`'s spread is day-to-day noise",
                          severity="noticing", recommended="act_on_it", label=LABELS["act_on_it"],
                          reason="the reason", blocker=False, recorded="act_on_it",
                          family="K3")
    act, change = fit_press.triage_refusal(triage(measured)).exits
    assert act["label"] == f"Act on it in Models: {measured.summary}"
    assert act["target"] == {"stage": "models", "key": None}
    assert change["target"] == AT_TRIAGE
    nowhere = item("sas_zeros", SAS, "act_on_it", recorded="act_on_it", stage="nowhere")
    act, change = fit_press.triage_refusal(triage(nowhere)).exits
    assert act == {"label": f"Act on it where it is decided: {SAS}", "decision": None}
    assert change["target"] == AT_TRIAGE


def test_a_noticing_still_open_speaks_before_one_kept_to_act_on():
    """The triage first: while a noticing has no recorded disposition, the press waits for the
    triage (its "Confirm all" as recommended), whatever else is marked to act on."""
    refused = fit_press.triage_refusal(triage(
        item("sas_zeros", SAS, "act_on_it", recorded="act_on_it", stage="data"),
        item("binary_text__gender", "`gender` is two-level text")))
    assert refused is not None and "1 noticing is still open" in refused.message
    assert refused.exits[1] == {"label": "Keep the recommended dispositions", "decision": CONFIRM}


# Two findings about `age` written here, under Estimate (``test_confirm_sweep.planned``): one with a
# repair this engine applies (``repairs.Family``: `binary_text`), so it is acted on in Your data where
# its repair is chosen ("act on it", ``sweep.recommend``'s finding with a repair); one whose family
# applies no repair (`heaping`), about a column the analysis reads, so it could bias the estimate
# (``sweep.recommend`` rule 4).
ACT_ON = {"id": "binary_text__age", "severity": "warning", "summary": "Age is written as text",
          "affected_columns": ["age"], "routes_to": None, "repairs": [{"key": "level"}],
          "answered_by": None}
HEAPS = {"id": "heaping__age", "severity": "warning", "summary": "Ages heap on round numbers",
         "affected_columns": ["age"], "routes_to": None, "repairs": [{"key": "map"}],
         "answered_by": None}


def estimating(findings: dict, records: list):
    """The engine's own triage at the gate before the lock, on ``findings`` after ``records``."""
    from turbotab.core import decisions, sweep
    from turbotab.core.tests.test_confirm_sweep import log_of, planned

    state = planned()
    if records:
        state = state.model_copy(update={"sweeps": decisions.fold(records).sweeps})
    log = log_of(state, records, findings=findings)
    return log, sweep.triage(state, log, findings, records)


def confirmed(log, t, body: dict, seq: int):
    """``body`` validated as the server validates a confirmation (the triage's own completion),
    as a record at ``seq``."""
    from turbotab.core import decisions
    from turbotab.core.tests.test_confirm_sweep import Ctx
    from turbotab.core.tests.test_stage_registry import record

    return record(seq, decisions.validate(body, Ctx(log, t)).model_dump(mode="json"))


def test_built_by_the_engine_a_kept_act_on_it_holds_fit_until_its_disposition_is_changed():
    """Construction, both ways, on the engine's own triage: "Keep the recommended dispositions"
    records `age`'s text as to act on and its heaping as could bias, and the press is held, naming
    the text alone; its way forward recorded, the press is released, the text recorded as could
    bias the estimate and the heaping as it was."""
    findings = {"findings": [ACT_ON, HEAPS]}
    log, t = estimating(findings, [])
    assert {i.id: (i.recommended, i.stage) for i in t.items} == {
        "binary_text__age": ("act_on_it", "data"), "heaping__age": ("could_bias", "data")}
    held = fit_press.triage_refusal(t)
    records = [confirmed(log, t, held.exits[1]["decision"], 1)]
    log, kept = estimating(findings, records)
    assert kept.answered and {i.id: i.recorded for i in kept.items} == {
        "binary_text__age": "act_on_it", "heaping__age": "could_bias"}
    refused = fit_press.triage_refusal(kept)
    assert refused is not None and refused.code == "triage_open"
    assert "1 noticing marked \"act on it\" is not acted on yet: Age is written as text." \
        in refused.message
    act, change = refused.exits
    assert act == {"label": "Act on it in Your data: Age is written as text", "decision": None,
                   "target": {"stage": "data", "key": "binary_text__age"}}
    # Each target names a card the engine's own quest log serves: the noticing's line in Your
    # data, and the stage the triage is drawn in (its gate, Models, before the lock).
    served = {s.key: {l.key for l in s.lines} for s in log.stages}
    assert "binary_text__age" in served[act["target"]["stage"]]
    assert change["target"] == {"stage": kept.stage, "key": None} and kept.stage in served
    records.append(confirmed(log, kept, change["decision"], 2))
    _log, changed = estimating(findings, records)
    assert {i.id: i.recorded for i in changed.items} == {
        "binary_text__age": "could_bias", "heaping__age": "could_bias"}
    assert fit_press.triage_refusal(changed) is None


def test_built_by_the_engine_a_kept_act_on_it_acted_on_drops_out_and_holds_fit_no_longer():
    """Silent, the other way through: the finding answered by a record of its own (its repair
    applied: ``answered_by``, as ``repairs.annotate`` serves it), its line is answered, so the
    triage no longer lists it, and nothing else holds the press."""
    findings = {"findings": [ACT_ON, HEAPS]}
    log, t = estimating(findings, [])
    records = [confirmed(log, t, CONFIRM, 1)]
    acted = {"findings": [{**ACT_ON, "answered_by": "r-repair"}, HEAPS]}
    _log, after = estimating(acted, records)
    assert [(i.id, i.recorded) for i in after.items] == [("heaping__age", "could_bias")]
    assert fit_press.triage_refusal(after) is None


def test_nothing_to_triage_holds_nothing():
    assert fit_press.triage_refusal(triage()) is None
    assert fit_press.triage_refusal(None) is None


def test_once_the_gate_has_passed_the_triage_holds_nothing():
    """The plan fixed (a press after the lock): the triage is read, no longer confirmed."""
    assert fit_press.triage_refusal(triage(item("a", "A"), item("b", "B", blocker=True),
                                           passed=True)) is None


# ── the reference journeys' export, pressing Fit again ───────────────────────

PLAN_OPEN = {"error": {"code": "plan_open", "message": "The analysis plan is still open.",
                       "exits": [{"label": "Press Fit on the analysis flowchart; it locks the plan",
                                  "decision": None}]}}


class Answer:
    def __init__(self, status: int, body: dict):
        self.status_code, self.body = status, body
        self.text = str(body)

    def json(self) -> dict:
        return self.body


class Served:
    """The server as the export's re-press meets it (``journeys.export``): the export refused while
    the plan is open; Fit refused with each of ``held`` in turn (the server's refusals while the
    triage is unconfirmed, then while a noticing marked "act on it" is not acted on), each until a
    decision is recorded, then taken, which locks the plan; each post kept."""

    def __init__(self, *held: Refusal):
        self.left, self.posts, self.locked = list(held), [], False

    @property
    def triaged(self) -> bool:
        return not self.left

    def get(self, url: str, **kw):
        assert url.endswith("/export"), url
        return Answer(200, {}) if self.locked else Answer(409, PLAN_OPEN)

    def post(self, url: str, json: dict | None = None, **kw):
        self.posts.append((url.rsplit("/", 1)[1], json))
        if url.endswith("/decisions"):
            if self.left:
                self.left.pop(0)
            return Answer(200, {"decisions": [{"seq": 30 + len(self.posts),
                                               "sentence": "Triaged.", "decision": json}]})
        if self.left:
            return Answer(409, self.left[0].to_dict())
        self.locked = True
        return Answer(200, {"locked": True, "pressed": True})


def exported(*held: Refusal):
    """``journeys.export`` on :class:`Served` with ``held``: what it returned, the server, and the
    run (its log of the decisions posted, each with where its answer came from, and its notes)."""
    from types import SimpleNamespace

    from turbotab.core.reference import journeys as J

    server = Served(*held)
    run = J.Run(SimpleNamespace(name="stub"))
    run.pid, run.source = "p1", "the export"
    run.client = J.Recording(server, run)
    run.drive = SimpleNamespace(c=run.client,
                                view=lambda: {"state": {"purpose": "inference"}, "stages": {}})
    return J.export(run), server, run


def test_the_journeys_export_presses_fit_again_through_the_triage_s_keep_the_recommended():
    """V2 (journeys.py's export re-press): Fit pressed again where the export waits for the lock is
    refused while the triage is unconfirmed; the refusal's way forward "Keep the recommended
    dispositions" is taken, as a person pressing Confirm all does, recorded as that way forward's,
    and Fit pressed again: the plan locks and the bundle is made."""
    held = fit_press.triage_refusal(triage(item("binary_text__gender", "`gender` is two-level text")))
    r, server, run = exported(held)
    assert r.status_code == 200
    assert server.posts == [("fit", None), ("decisions", CONFIRM), ("fit", None)]
    assert [(e["kind"], e["status"], e["source"]) for e in run.log] == [
        ("confirm_sweep", 200, "the way forward of Fit's refusal, “Keep the recommended "
                               "dispositions”")]


def test_the_journeys_export_leaves_a_blocking_noticing_s_refusal_as_it_is():
    """Both ways: a blocking noticing's refusal offers no way that records the triage, so nothing
    is posted; Fit is pressed again as before (twice at most) and the export's refusal stands."""
    held = fit_press.triage_refusal(triage(item("outcome_constant", "`glucose` takes one value",
                                                blocker=True)))
    r, server, run = exported(held)
    assert r.status_code == 409 and r.json() == PLAN_OPEN
    assert server.posts == [("fit", None), ("fit", None)] and run.log == []


# ── the reference journeys' FitPresser, once the Router reaches Fit ──────────

AT_FIT = {"interview": [{"key": "open_seal", "status": "waiting", "waiting_on": ["fit"]}],
          "decisions": [], "stages": {"fit": {"status": "fresh", "key": "k1"}},
          "state": {"purpose": "inference"}}
AT_END = {"interview": [{"key": "open_seal", "status": "answered", "waiting_on": []}],
          "decisions": [{}], "stages": {"fit": {"status": "fresh", "key": "k1"}},
          "state": {"purpose": "inference"}}
WAY = "the way forward of Fit's refusal, “Keep the recommended dispositions”"


def followed(*held: Refusal):
    """``journeys.follow`` on :class:`Served` with ``held`` (none: Fit taken at its first press):
    the Router at Fit, the seal waiting for it, until the plan locks, then at its end; the drivers'
    last answer, before Fit, the WP17 questions' (its source the run's own). The run (its log, each
    decision posted with where its answer came from) and the server."""
    from types import SimpleNamespace

    from turbotab.core.reference import journeys as J

    server = Served(*held)
    run = J.Run(SimpleNamespace(name="stub", timeout=30.0, before={}))
    run.pid, run.source = "p1", J.WP17_SOURCE
    run.client = J.Recording(server, run)
    run.drive = SimpleNamespace(c=run.client, view=lambda: AT_END if server.locked else AT_FIT)
    assert J.follow(run)
    return run, server


def test_the_journeys_fit_presser_credits_the_triage_it_confirms_to_fits_way_forward():
    """P3: where the Router reaches Fit and the open noticings' triage holds the press, the way
    forward "Keep the recommended dispositions" the drivers take (``journeys.FitPresser``) is the
    source of the confirmation it posts, in the run's log and in the review packet's table of
    answers (``packet.answers_rows``), not the answer posted before it (the WP17 questions')."""
    from turbotab.core.reference import journeys as J
    from turbotab.core.reference import packet

    held = fit_press.triage_refusal(triage(item("binary_text__gender", "`gender` is two-level text")))
    run, server = followed(held)
    assert server.posts == [("fit", None), ("decisions", CONFIRM), ("fit", None)]
    assert [(e["kind"], e["status"], e["source"]) for e in run.log] == [
        ("confirm_sweep", 200, WAY)]
    rows = packet.answers_rows({"answers": run.log})
    assert len(rows) == 3 and f"| {WAY} |" in rows[2] and J.WP17_SOURCE not in rows[2]
    assert "Fit waited for the open noticings' triage: took “Keep the recommended dispositions”" \
        in run.notes


def test_the_journeys_fit_presser_posts_nothing_where_no_triage_holds_fit():
    """Both ways: Fit taken at its first press, nothing is posted beside it and the run's source
    is the answer's before it, as it was."""
    from turbotab.core.reference import journeys as J

    run, server = followed()
    assert server.posts == [("fit", None)] and run.log == [] and run.source == J.WP17_SOURCE


# ── the drivers, through a noticing kept to act on ───────────────────────────


def kept_then_unacted() -> tuple[Refusal, Refusal]:
    """The server's two refusals of a press where the triage holds one open noticing recommended to
    act on: first while it has no recorded disposition, then, "Keep the recommended dispositions"
    taken, while it is recorded "act on it" and not acted on (each built from a triage written
    here)."""
    held = fit_press.triage_refusal(triage(item("sas_zeros", SAS, "act_on_it", stage="data")))
    unacted = fit_press.triage_refusal(triage(item("sas_zeros", SAS, "act_on_it",
                                                   recorded="act_on_it", stage="data")))
    return held, unacted


LIMIT = keeping({"id": "finding:sas_zeros", "key": "sas_zeros", "value": "could_bias"})
LIMIT_WAY = f"the way forward of Fit's refusal, “Record as a limitation: {SAS}”"


def test_the_journeys_export_takes_each_way_forward_a_kept_act_on_it_offers_and_presses_again():
    """The export's re-press through both refusals: "Keep the recommended dispositions", then, the
    noticing kept to act on and not acted on, the first way forward that records something, its
    "Record as a limitation", each recorded as its way forward's, and Fit pressed after each: the
    plan locks and the bundle is made."""
    r, server, run = exported(*kept_then_unacted())
    assert r.status_code == 200
    assert server.posts == [("fit", None), ("decisions", CONFIRM), ("fit", None),
                            ("decisions", LIMIT), ("fit", None)]
    assert [(e["kind"], e["status"], e["source"]) for e in run.log] == [
        ("confirm_sweep", 200, WAY), ("confirm_sweep", 200, LIMIT_WAY)]


def test_the_journeys_fit_presser_takes_each_way_forward_a_kept_act_on_it_offers():
    """The FitPresser likewise, once the Router reaches Fit: each decision credited to the way
    forward it was posted for, and a note of each."""
    run, server = followed(*kept_then_unacted())
    assert server.posts == [("fit", None), ("decisions", CONFIRM), ("fit", None),
                            ("decisions", LIMIT), ("fit", None)]
    assert [(e["kind"], e["status"], e["source"]) for e in run.log] == [
        ("confirm_sweep", 200, WAY), ("confirm_sweep", 200, LIMIT_WAY)]
    assert f"Fit waited for the open noticings' triage: took “Record as a limitation: {SAS}”" \
        in run.notes


def test_the_server_drive_press_stops_where_no_way_forward_records_anything():
    """Both ways: the press goes on only through ways forward that record something; a refusal of
    the triage with none (a blocking noticing, after "Keep the recommended dispositions") is the
    answer, as it came, and nothing more is posted."""
    from turbotab.core.tests.acceptance import server_drive

    held, _ = kept_then_unacted()
    blocking = fit_press.triage_refusal(triage(item("outcome_constant", "`glucose` takes one value",
                                                    blocker=True)))
    server = Served(held, blocking)
    r = server_drive.fit_response(server, "p1")
    assert r.status_code == 409 and r.json() == blocking.to_dict()
    assert server.posts == [("fit", None), ("decisions", CONFIRM), ("fit", None)]

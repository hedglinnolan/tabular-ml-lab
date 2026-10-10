"""TRUST on the server: the service's wiring of the quest-log critique's trust leaks.

* GET /stages/explore serves each of Explore's findings with the label the person sees, the
  collinear one from the K5 noticing measured on the table (``_serve`` → ``_collinear_noticed``).
* GET /record leaves out a Router step whose reason is a consumer's refusal measured on the table
  (``record`` → ``quest.record_steps``), and the quest log is given those refusals (``_quest``).

Each reads the service's own methods on stand-ins for what they call, as the phase-1 follow-ups'
server tests do; the expected labels and lines are written here.
"""
from __future__ import annotations

from types import SimpleNamespace

from turbotab.core import materiality as M
from turbotab.core import quest
from turbotab.core.tests.test_c6a_phase1_follow_ups import _sugar_share
from turbotab.core.tests.test_collinear_adjustment import ADJUST, state, table
from turbotab.core.tests.test_trust_leaks_and_sentences import CLASH, _clashing, _log
from turbotab.server import service as S
from turbotab.server.service import ProjectService

EXPLORE = {"findings": [
    {"id": "explore::collinear", "kind": "collinear", "summary": "", "columns": ["fat", "protein"],
     "view": "table_focus"},
    {"id": "explore::low_variance", "kind": "low_variance", "summary": "", "columns": ["x"],
     "view": "table_focus"}]}


def _explore_service(s, noticed):
    svc = SimpleNamespace(
        log=lambda pid: SimpleNamespace(state=lambda: s), _pressed=lambda pid, st: False,
        _outcome_served=lambda pid, stage, artifact: artifact,
        _noticed=lambda pid, st: noticed)
    svc._collinear_noticed = lambda pid, st: ProjectService._collinear_noticed(svc, pid, st)
    return svc


def test_explore_is_served_with_each_findings_label_from_the_measured_noticing():
    f = table()
    s = state(ADJUST)
    assert _sugar_share(f, ADJUST) < 0.5  # by hand: sugar outside the near dependency
    noticing = M.collinear_noticing(s, f)
    assert noticing is not None and noticing.thread == M.COLLINEAR_THREAD
    served = ProjectService._serve(_explore_service(s, [noticing]), "p", "explore", EXPLORE, None)
    assert [x["label"] for x in served["findings"]] == ["For the record", "Confirm"]


def test_explore_inside_the_dependency_is_served_as_a_decide():
    f = table(sugar_inside=True)
    columns = [*ADJUST, "starch"]
    assert _sugar_share(f, columns) >= 0.5  # by hand: sugar inside the near dependency
    s = state(columns)
    served = ProjectService._serve(_explore_service(s, [M.collinear_noticing(s, f)]), "p",
                                   "explore", EXPLORE, None)
    assert served["findings"][0]["label"] == "Decide"


def test_the_record_route_leaves_out_what_waits_on_an_answer():
    st, artifacts, records, steps = _clashing()
    asked = quest.fit_waits(st)
    q = S._QuestInputs(log=_log(st, artifacts, records, steps, [asked]), state=st,
                       records=records, steps=steps, findings=None, asked=(asked,))
    svc = SimpleNamespace(_quest=lambda pid: q, _shown=lambda pid, name: None)
    texts = [r.text for stage in ProjectService.record(svc, "p").stages for r in stage.lines]
    assert CLASH not in texts
    # the same steps with no refusal measured keep it, as the critique found it
    bare = SimpleNamespace(_quest=lambda pid: S._QuestInputs(
        log=q.log, state=st, records=records, steps=steps, findings=None),
        _shown=lambda pid, name: None)
    assert CLASH in [r.text for stage in ProjectService.record(bare, "p").stages
                     for r in stage.lines]


def test_the_refusal_is_measured_on_the_tables_own_summaries():
    st, _artifacts, _records, _steps = _clashing()

    class Store:
        def info(self):
            return SimpleNamespace(columns=[SimpleNamespace(name="sugar", dtype="float",
                                                            n_unique=900)])

    svc = SimpleNamespace(_store_or_none=lambda pid: Store())
    fresh = {"ingest": SimpleNamespace(status="fresh")}
    [asked] = ProjectService._asked(svc, "p", st, fresh)
    assert (asked.question, asked.message) == ("models", CLASH)
    # before the table is read nothing is measured
    assert ProjectService._asked(svc, "p", st, {"ingest": SimpleNamespace(status="running")}) == ()

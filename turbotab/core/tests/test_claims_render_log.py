"""The claims render log (SWARM_ENGINE_PLAN E4; R1_PLAN R1-M): each claim rendered for a project is
appended to a JSON-lines file beside it, in every mode, from the server's threads and from the job
workers alike.

* Inside ``render_log(path)`` each render is one line: the claim asked for, the register, the
  frozen status, the grade (its weakest part's), the fallback that rendered in its place, and the
  claim's version and content hash; a render that is refused is logged with why, and still raises.
* Outside one, nothing is written anywhere.
* The log is per thread of control (a context variable): two projects rendering at once on two
  threads each get their own lines, and a worker process writes to the same file the server reads.
* The engine's stage graph (``workspace.stage_graph``) runs every stage inside its project's log,
  and is otherwise the stage graph itself, so no stage key moves.

The expected grades come from the planted parts (construction); the expected hash is the frozen
status file's, the record a render is gated by.
"""
from __future__ import annotations

import dataclasses
import json
import multiprocessing
import os
import threading
from pathlib import Path

import pytest

from turbotab.core.claims import Basis, NotRenderable, Part
from turbotab.core.claims._registry import render_log
from turbotab.core.tests.test_claims_fixtures import registry, rule, teaching

ID = "planted.few_clusters"
FALLBACK = "planted.your_call"
INSIDE = {"outcome": "binary"}
OUTSIDE = {"outcome": "continuous"}


def planted(tmp_path: Path, *claims):
    """A planted registry of ``claims`` in ``tmp_path/root``, its status frozen."""
    (tmp_path / "root").mkdir(exist_ok=True)
    return registry(tmp_path / "root", *claims)


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()] if path.exists() else []


def frozen_hash(reg, id: str) -> str:
    return {c["id"]: c["hash"] for c in json.loads(reg.frozen_path.read_text("utf-8"))["claims"]}[id]


def test_a_render_with_no_log_open_writes_nothing(tmp_path):
    reg = planted(tmp_path, rule())
    before = sorted(p for p in tmp_path.rglob("*") if p.is_file())
    assert reg.card(ID, INSIDE, strict=True)
    assert reg.methods(ID, None, strict=True)
    assert sorted(p for p in tmp_path.rglob("*") if p.is_file()) == before


def test_each_render_in_a_log_is_one_line_with_its_status_grade_and_hash(tmp_path):
    graded = rule(parts=(Part("value", grade="judgment", basis=Basis(reason="a planted judgment")),
                         Part("scope", grade="unsupported", basis=Basis(reason="none yet"))))
    reg = planted(tmp_path, graded)
    log = tmp_path / "project" / "claims_rendered.jsonl"
    with render_log(log):
        reg.card(ID, INSIDE, strict=True)
        reg.methods(ID, None, strict=True)
    card, methods = lines(log)
    assert {k: card[k] for k in ("id", "register", "status", "grade", "fallback", "refused")} == {
        "id": ID, "register": "card", "status": "built",
        "grade": "unsupported",  # the weaker of the two planted parts
        "fallback": None, "refused": None}
    assert card["version"] == 1 and card["hash"] == frozen_hash(reg, ID)
    assert card["at"].endswith("+00:00")
    assert methods["register"] == "methods" and methods["id"] == ID
    reg.card(ID, INSIDE, strict=True)  # the log is closed: nothing more
    assert len(lines(log)) == 2


def test_a_fallback_rendered_in_a_claim_s_place_is_named_with_its_own_record(tmp_path):
    reg = planted(tmp_path, rule(fallback=FALLBACK), teaching(version=3))
    log = tmp_path / "claims.jsonl"
    with render_log(log):
        assert reg.card(ID, OUTSIDE, strict=False) == "Your call: here is what each choice does."
        reg.card(ID, INSIDE, strict=False)
    fell, held = lines(log)
    assert fell["id"] == ID and fell["fallback"] == FALLBACK
    assert (fell["status"], fell["grade"], fell["version"]) == ("built", "judgment", 3)
    assert fell["hash"] == frozen_hash(reg, FALLBACK) != frozen_hash(reg, ID)
    assert held["fallback"] is None and held["hash"] == frozen_hash(reg, ID)


def test_a_refused_render_is_logged_with_why_and_still_raises(tmp_path):
    reg = planted(tmp_path, rule(enforced_by=("planted.engine:not_written_yet",)))
    log = tmp_path / "claims.jsonl"
    with render_log(log), pytest.raises(NotRenderable, match="proposed"):
        reg.card(ID, INSIDE, strict=False)
    [line] = lines(log)
    assert line["id"] == ID and line["status"] == "proposed" and line["fallback"] is None
    assert "proposed" in line["refused"]


CURRENT: dict = {}  # the planted registry a planted module renders from as it is imported
STATIC_FILL = f"""
from turbotab.core.tests import test_claims_render_log as T

SAID = T.CURRENT["reg"].card({ID!r}, {INSIDE!r}, strict=True)


def say():
    return T.CURRENT["reg"].card({ID!r}, {INSIDE!r}, strict=True)
"""


def test_a_static_fill_made_while_a_module_is_imported_is_not_the_project_s(tmp_path, monkeypatch):
    """A module's import renders its static fills once per process, for no project, even when
    the import happens to run inside a project's log (a lazy import in a stage, as
    ``methods.trials``'s contracts are built): only renders made after the import are the
    project's."""
    import importlib
    import sys

    CURRENT["reg"] = planted(tmp_path, rule())
    (tmp_path / "planted_static_fill.py").write_text(STATIC_FILL, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "planted_static_fill", raising=False)
    log = tmp_path / "claims.jsonl"
    try:
        with render_log(log):
            module = importlib.import_module("planted_static_fill")
            assert module.SAID  # rendered, as the import ran
            assert lines(log) == []  # and not logged
            module.say()  # a render after the import is the project's
    finally:
        sys.modules.pop("planted_static_fill", None)
        CURRENT.clear()
    assert [x["id"] for x in lines(log)] == [ID]


def test_two_threads_rendering_for_two_projects_each_log_only_their_own(tmp_path):
    reg = planted(tmp_path, rule(), teaching())
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    barrier = threading.Barrier(2)

    def render(log: Path, id: str, n: int) -> None:
        with render_log(log):
            barrier.wait()
            for _ in range(n):
                reg.card(id, INSIDE if id == ID else None, strict=True)

    threads = [threading.Thread(target=render, args=(a, ID, 30)),
               threading.Thread(target=render, args=(b, FALLBACK, 20))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert [line["id"] for line in lines(a)] == [ID] * 30
    assert [line["id"] for line in lines(b)] == [FALLBACK] * 20


def _render_in_a_child(root: str, log: str) -> None:
    Path(root).mkdir(exist_ok=True)
    reg = registry(Path(root), rule())
    with render_log(Path(log)):
        reg.card(ID, INSIDE, strict=True)


def test_a_render_in_another_process_lands_in_the_same_file(tmp_path):
    log = tmp_path / "claims.jsonl"
    with render_log(log):
        planted(tmp_path, rule()).card(ID, INSIDE, strict=True)
    child = multiprocessing.get_context("spawn").Process(
        target=_render_in_a_child, args=(str(tmp_path / "child"), str(log)))
    child.start()
    child.join(60)
    assert child.exitcode == 0
    assert [line["id"] for line in lines(log)] == [ID, ID]


REAL_CLAIM = "designs.observational_default"  # a built claim of the shipped registry, no regime
LOGGED_TOY = "turbotab.core.tests.test_claims_render_log:logged_toy_graph"


def _renders_a_claim(ctx) -> dict:
    from turbotab.core import claims

    return {"said": claims.card(REAL_CLAIM), "process": os.getpid()}


def logged_toy_graph():
    """A heavy stage (a job worker runs it) and a light one (the engine's thread), each wrapped
    as the engine's stage graph wraps every stage (``workspace.logged_stage``)."""
    from turbotab.core.graph import Graph, Stage
    from turbotab.core.workspace import logged_stage

    return Graph([Stage("heavy", 1, (), (), logged_stage(_renders_a_claim), heavy=True),
                  Stage("light", 1, (), (), logged_stage(_renders_a_claim))])


def test_a_claim_rendered_in_a_job_worker_lands_in_its_project_s_log(runner, tmp_path):
    """The stage runs in a worker process (its artifact names another process id) and in the
    engine's thread; each render is a line of the project folder's log, which the paths the engine
    hands every stage name (``ProjectService._project_ctx``)."""
    import time

    from turbotab.core.events import EventBus
    from turbotab.core.graph import Engine, ProjectContext
    from turbotab.core.tests.toy_stages import ToyState
    from turbotab.core.workspace import CLAIMS_LOG

    project = tmp_path / "projects" / "p0123456789"
    project.mkdir(parents=True)

    def project_ctx(pid: str) -> ProjectContext:
        return ProjectContext(project_id=pid, state=ToyState(), cache_root=project / "cache",
                              paths={"project_dir": str(project)}, settings={},
                              fingerprint="fp")

    engine = Engine(graph_factory=LOGGED_TOY, runner=runner, bus=EventBus(),
                    project_ctx=project_ctx)
    try:
        engine.on_decision("p0123456789")
        end = time.monotonic() + 60
        while not all(s.status == "fresh" for s in engine.status("p0123456789").values()):
            assert time.monotonic() < end, engine.status("p0123456789")
            time.sleep(0.05)
        heavy = engine.get("p0123456789", "heavy").artifact
        light = engine.get("p0123456789", "light").artifact
    finally:
        engine.shutdown()
    assert heavy["process"] != os.getpid() and light["process"] == os.getpid()
    logged = lines(project / CLAIMS_LOG)
    assert [(x["id"], x["status"], x["refused"]) for x in logged] == [(REAL_CLAIM, "built", None)] * 2


def test_the_engine_s_graph_is_the_stage_graph_with_each_stage_run_in_its_project_s_log(tmp_path):
    from turbotab.core.graph import StageContext
    from turbotab.core.stages import build_graph
    from turbotab.core.workspace import CLAIMS_LOG, logged_stage, stage_graph

    plain, logged = build_graph(), stage_graph()
    assert [s.name for s in logged.order()] == [s.name for s in plain.order()]
    for mine, theirs in zip(logged.stages(), plain.stages()):
        assert dataclasses.replace(mine, fn=theirs.fn) == theirs  # every field the key reads
        assert mine.fn is not theirs.fn

    reg = planted(tmp_path, rule())
    project = tmp_path / "projects" / "p0123456789"
    project.mkdir(parents=True)

    def renders(ctx: StageContext) -> dict:
        return {"said": reg.card(ID, INSIDE, strict=True)}

    ctx = StageContext(project_id="p0123456789", state=None, inputs={},
                       paths={"project_dir": str(project)}, settings={})
    assert logged_stage(renders)(ctx) == renders(ctx)  # the stage's own result, untouched
    assert [line["id"] for line in lines(project / CLAIMS_LOG)] == [ID]

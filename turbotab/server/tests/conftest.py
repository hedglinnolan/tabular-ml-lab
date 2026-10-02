from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from turbotab.core.config import Settings
from turbotab.server.app import create_app

REPO = Path(__file__).resolve().parents[3]
SAMPLES = REPO / "turbotab" / "sample_data"
DIETARY = SAMPLES / "dietary_recalls.csv"


def make_client(home: Path, mode: str, workers: int, base_url: str) -> TestClient:
    settings = Settings(home=home, mode=mode, workers=workers, memory_budget_bytes=2 << 30)
    # A dist folder that does not exist: the server answers / with its build hint.
    app = create_app(settings, frontend_dist=home / "no-frontend-here")
    return TestClient(app, base_url=base_url)


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    """A local-mode server with two workers, shared by the whole run."""
    with make_client(tmp_path_factory.mktemp("local"), "local", 2, "http://127.0.0.1") as c:
        yield c


@pytest.fixture(scope="session")
def server_client(tmp_path_factory):
    with make_client(tmp_path_factory.mktemp("server"), "server", 1, "http://turbotab.example") as c:
        yield c


def wait_for(client: TestClient, pid: str, want: dict[str, str], timeout: float = 60.0) -> dict:
    """Poll the project until each named stage has the wanted status; returns the view."""
    end = time.monotonic() + timeout
    while True:
        view = client.get(f"/api/projects/{pid}").json()
        stages = view["stages"]
        if all(stages[name]["status"] == status for name, status in want.items()):
            return view
        for name, status in want.items():
            if status != "error" and stages[name]["status"] == "error":
                pytest.fail(f"{name} failed: {stages[name]['error']}")
        if time.monotonic() > end:
            pytest.fail(f"timed out waiting for {want}: {stages}")
        time.sleep(0.05)


def open_by_path(client: TestClient, path: Path = DIETARY) -> str:
    response = client.post("/api/projects", json={"path": str(path)})
    assert response.status_code == 200, response.text
    return response.json()["id"]


@pytest.fixture(scope="session")
def dietary(client) -> str:
    """dietary_recalls.csv, opened by path, with ingest and profile fresh."""
    pid = open_by_path(client)
    wait_for(client, pid, {"ingest": "fresh", "profile": "fresh"})
    return pid


# ── answering in the Router's order (M2_CONTRACT §12.2) ──────────────────────
# The server refuses an answer to a question still waiting behind an unanswered one. A test about
# a later question first gives each earlier one its usual answer, the way a user clicking through
# would: ``prepare`` walks the Router up to the question a decision answers.


def _artifact(client: TestClient, pid: str, stage: str) -> dict:
    return client.get(f"/api/projects/{pid}/stages/{stage}").json().get("artifact") or {}


def usual_answer(client: TestClient, pid: str, key: str, view: dict) -> dict:
    """The answer a user most often gives ``key`` on this project (never the lens or the target)."""
    state = view["state"]
    if key == "orientation":
        return {"kind": "set_orientation", "orientation": "sample_major"}
    if key in ("event", "task"):
        info = _artifact(client, pid, "target_info")
        if key == "task":
            return {"kind": "set_task", "column": state["target"], "task": info["task"]}
        levels = [str(c["value"]) for c in info.get("classes") or []]
        level = "1" if "1" in levels else ("1.0" if "1.0" in levels else levels[-1])
        return {"kind": "set_event", "column": state["target"], "level": level}
    if key == "purpose":
        return {"kind": "set_purpose", "purpose": "prediction"}
    if key == "grain":
        reading = _artifact(client, pid, "structure").get("grain") or {}
        if reading.get("suggested"):
            return {"kind": "set_grain", "grain": "repeated", "id_column": reading["suggested"][0]}
        return {"kind": "set_grain", "grain": "one_row_per_unit",
                "acknowledged": bool(reading.get("if_one_row"))}
    if key == "repeat_kind":
        reading = _artifact(client, pid, "structure").get("repeats") or {}
        return {"kind": "set_repeat_kind", "repeat_kind": reading.get("reading") or "repeats"}
    if key == "unit":
        return {"kind": "set_unit", "unit": "row"}
    if key == "aggregation":
        outcome = _artifact(client, pid, "structure").get("outcome") or {}
        return {"kind": "set_aggregation", "method": "mean",
                "outcome": "first" if outcome.get("varies") else None}
    if key == "temporal":
        return {"kind": "set_temporal", "temporal": False}
    if key == "roles":
        columns = _artifact(client, pid, "roles").get("columns") or []
        return {"kind": "set_roles", "roles": {c["column"]: c["proposed"] for c in columns}}
    if key == "survey":  # the first weight the proposals offer (audit §5 WP10)
        offered = (_artifact(client, pid, "proposals").get("survey") or {}).get("options") or []
        return offered[0]["decision"] if offered else {"kind": "set_survey", "estimand": "sample"}
    if key == "exclusions":
        return {"kind": "set_exclusions", "rules": []}
    if key == "missing":
        return {"kind": "set_missing", "strategy": "complete_case"}
    if key == "split":
        return {"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5}
    if key == "energy_adjustment":
        return {"kind": "set_energy_adjustment", "method": "none"}
    if key == "models":
        return {"kind": "select_models", "models": ["linear"]}
    if key == "substitution":
        from turbotab.core.stages.rows import energy_bearing

        swaps = [c for c, r in (state["roles"] or {}).items() if r == "exposure" and energy_bearing(c)]
        return {"kind": "set_substitution", "donor": swaps[0], "recipient": swaps[1]}
    pytest.fail(f"the {key} question has no usual answer; answer it in the test")


def prepare(client: TestClient, pid: str, decision: dict, timeout: float = 120.0) -> None:
    """Answer, with its usual answer, every question the Router asks before the one ``decision``
    answers, waiting on the stages each needs. A decision that answers no question needs nothing."""
    from turbotab.core.interview import NEEDS
    from turbotab.core.sequence import question_of

    question = question_of(str(decision.get("kind")))
    if question is None:
        return
    end = time.monotonic() + timeout
    while True:
        view = client.get(f"/api/projects/{pid}").json()
        steps = view["interview"]
        mine = next(s for s in steps if s["key"] == question)
        first = next((s for s in steps if s["status"] in ("open", "waiting")), None)
        if mine["status"] not in ("open", "waiting") or first is None or first["key"] == question:
            return
        if time.monotonic() > end:
            pytest.fail(f"timed out answering what comes before {question}: {first}")
        # the first question waits on its own stages, or its options are not read yet
        unread = [n for n in (*first["waiting_on"], *NEEDS.get(first["key"], ()))
                  if view["stages"].get(n, {}).get("status") not in (None, "fresh")]
        if first["status"] == "waiting" or unread:
            failed = [n for n in unread if view["stages"][n]["status"] == "error"]
            if failed:
                pytest.fail(f"{failed} failed: {view['stages'][failed[0]].get('error')}")
            time.sleep(0.05)
            continue
        answer = usual_answer(client, pid, first["key"], view)
        response = client.post(f"/api/projects/{pid}/decisions", json=answer)
        assert response.status_code == 200, (first["key"], response.text)

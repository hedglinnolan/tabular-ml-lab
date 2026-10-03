"""Driving a project through the real server (FastAPI app over its job runner), as the audit did.

Shared by the WP12 acceptance tests: :class:`Drive` answers the opening sequence in the Router's
order and waits for a stage's artifact; :func:`local_server` is a local-mode server with two
workers over a fresh home. A reading the server asks about is answered from the fixture's declared
:class:`Truth`, never a constant (BLUEPRINT §14.3).
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from turbotab.core.graph import artifact_dir
from turbotab.core.tests.truths import Truth, answer_refusal, asked  # noqa: F401 - re-exported


@contextmanager
def local_server(home: Path) -> Iterator[Any]:
    from fastapi.testclient import TestClient

    from turbotab.core.config import Settings
    from turbotab.server.app import create_app

    settings = Settings(home=home, mode="local", workers=2, memory_budget_bytes=1 << 30)
    with TestClient(create_app(settings, frontend_dist=home / "none"),
                    base_url="http://127.0.0.1") as client:
        yield client


def open_project(client: Any, path: Path, truth: "Truth | None" = None) -> "Drive":
    r = client.post("/api/projects", json={"path": str(path)})
    assert r.status_code == 200, r.text
    drive = Drive(client, r.json()["id"], truth)
    drive.artifact("ingest")
    return drive


def settle_post(client: Any, pid: str, body: dict[str, Any], truth: Truth | None = None) -> Any:
    """Post ``body``; while it is refused only for readings to settle, answer every reading it
    asks about from the fixture's ``truth`` (one block confirmation, ``confirm_readings``, which
    settles exactly the readings it lists; an energy column's unit and days as declared), then
    post it again."""
    url = f"/api/projects/{pid}/decisions"
    truth = truth if truth is not None else Truth(fixture="the test (no truth given)")
    response = _post_when_reached(client, url, body)
    return answer_refusal(lambda d: client.post(url, json=d), response, truth,
                          lambda: _post_when_reached(client, url, body))


def _post_when_reached(client: Any, url: str, body: dict[str, Any], timeout: float = 240.0) -> Any:
    """Post ``body``; while the Router holds its question behind an earlier one that a reshaped
    table is recomputing (a code-or-amount answer rebuilds the working table), wait and post
    again."""
    end = time.monotonic() + timeout
    while True:
        response = client.post(url, json=body)
        if response.status_code != 409 or time.monotonic() > end:
            return response
        if (response.json().get("error") or {}).get("code") != "not_yet":
            return response
        time.sleep(0.1)


class Drive:
    """Answers the opening sequence through the real HTTP API, in the Router's order."""

    def __init__(self, client: Any, pid: str, truth: Truth | None = None):
        self.c, self.pid = client, pid
        self.truth = truth if truth is not None else Truth()

    def view(self) -> dict[str, Any]:
        return self.c.get(f"/api/projects/{self.pid}").json()

    def post(self, body: dict[str, Any]) -> Any:
        return self.c.post(f"/api/projects/{self.pid}/decisions", json=body)

    def decide(self, body: dict[str, Any]) -> None:
        r = settle_post(self.c, self.pid, body, self.truth)
        assert r.status_code == 200, (body["kind"], r.text[:900])

    def decide_roles(self, roles: dict[str, str]) -> None:
        """Record the roles as their author answers them (BLUEPRINT §14, recognition's leash):
        the ``set_roles`` answer, then every role the server recorded unconfirmed (proposed below
        high confidence) confirmed in one block, each with the role the author gave it, as a user
        who knows the table does. A bulk ``set_roles`` alone confirms none of them. The author's
        roles are the fixture's truth for its role readings."""
        for column, role in roles.items():
            self.truth.setdefault(f"role:{column}", role)
        self.decide({"kind": "set_roles", "roles": roles})
        record = self.view()["decisions"][-1]
        waiting = record["decision"].get("unconfirmed") or []
        if waiting:
            self.decide({"kind": "confirm_readings", "items": [
                {"reading": "role", "column": c, "value": roles[c]} for c in waiting]})

    def reach(self, key: str, timeout: float = 120.0) -> dict[str, Any]:
        end = time.monotonic() + timeout
        while True:
            steps = self.view()["interview"]
            step = next(s for s in steps if s["key"] == key)
            first = next((s for s in steps if s["status"] in ("open", "waiting")), None)
            ready = step["status"] not in ("open", "waiting") or first is None or first["key"] == key
            if ready and not (step["status"] == "waiting" and step.get("waiting_on")):
                return step
            if (key != "task" and first is not None and first["key"] == "task"
                    and first["status"] == "open" and first.get("followup")):
                # WP18: a driver passing the task question answers what it still asks, as usual.
                self.task_followups()
                continue
            assert time.monotonic() < end, f"{key} held behind {first}"
            time.sleep(0.05)

    def answer(self, key: str, body: dict[str, Any]) -> None:
        if self.reach(key)["status"] in ("open", "waiting"):
            self.decide(body)
        if key == "task":
            self.task_followups()

    def task_followups(self) -> None:
        """WP18 (audit RO-10): what the task question still asks after its task, answered as the
        fixture declares (``outcome_scale:<column>``), else as usual: the original scale, and an
        ordinal text outcome's levels in the order their words propose."""
        for _ in range(3):
            step = self.reach("task")
            follow = step.get("followup")
            if step["status"] not in ("open", "waiting") or follow is None:
                return
            target = self.view()["state"]["target"]
            if follow == "scale":
                scale = self.truth.get(f"outcome_scale:{target}", "original")
                self.decide({"kind": "set_outcome_scale", "column": target, "scale": scale})
            else:
                question = self.artifact("target_info").get("order_question") or {}
                self.decide({"kind": "set_outcome_order", "column": target,
                             "levels": question.get("proposed_order") or question.get("levels")})

    def artifact(self, stage: str, timeout: float = 240.0) -> dict[str, Any]:
        end = time.monotonic() + timeout
        while True:
            status = self.view()["stages"][stage]
            if status["status"] == "fresh":
                return self.c.get(f"/api/projects/{self.pid}/stages/{stage}").json()["artifact"]
            assert status["status"] != "error", status
            assert time.monotonic() < end, f"{stage} never fresh: {status}"
            time.sleep(0.05)

    def sealed(self) -> set[int]:
        self.artifact("split")
        key = self.view()["stages"]["split"]["key"]
        cache = self.c.app.state.service.workspace.cache_dir(self.pid)
        frame = pd.read_parquet(artifact_dir(cache, "split", key) / "frames" / "sealed.parquet")
        return set(frame["row_id"].astype(int))

"""Driving a project through the real server (FastAPI app over its job runner), as the audit did.

Shared by the WP12 acceptance tests: :class:`Drive` answers the opening sequence in the Router's
order and waits for a stage's artifact; :func:`local_server` is a local-mode server with two
workers over a fresh home.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from turbotab.core.graph import artifact_dir


@contextmanager
def local_server(home: Path) -> Iterator[Any]:
    from fastapi.testclient import TestClient

    from turbotab.core.config import Settings
    from turbotab.server.app import create_app

    settings = Settings(home=home, mode="local", workers=2, memory_budget_bytes=1 << 30)
    with TestClient(create_app(settings, frontend_dist=home / "none"),
                    base_url="http://127.0.0.1") as client:
        yield client


def open_project(client: Any, path: Path) -> "Drive":
    r = client.post("/api/projects", json={"path": str(path)})
    assert r.status_code == 200, r.text
    drive = Drive(client, r.json()["id"])
    drive.artifact("ingest")
    return drive


# The readings ledger (BLUEPRINT §14.1): an answer refused only because a reading it rests on is
# not settled (a predictor's whole numbers, a unit's repeating counts, a body measure's unit, the
# records' order) is answered as its author would, one ``confirm_reading`` per reading, keeping the
# reading the engine applied before the ledger, then posted again. Role confirmations the leash
# tests drive themselves (``role_unconfirmed``) are never taken here.
_KEEPS = {"select_models": "amount", "set_aggregation": "code"}


def settle_post(client: Any, pid: str, body: dict[str, Any]) -> Any:
    url = f"/api/projects/{pid}/decisions"
    response = client.post(url, json=body)
    for _ in range(4):
        if response.status_code != 409:
            return response
        error = response.json().get("error") or {}
        if error.get("code") != "reading_unsettled":
            return response
        seen: set[tuple[str, str]] = set()
        for item in error.get("exits") or []:
            d = item.get("decision") or {}
            if d.get("kind") != "confirm_reading" or (d["reading"], d["column"]) in seen:
                continue
            if d["reading"] == "code_or_count" and d["value"] != _KEEPS.get(body["kind"], "amount"):
                continue
            if d["reading"] == "cluster" and d["value"] != "yes":
                continue
            seen.add((d["reading"], d["column"]))
            r = client.post(url, json=d)
            assert r.status_code == 200, (d, r.text[:600])
        if not seen:
            return response
        response = client.post(url, json=body)
    return response


class Drive:
    """Answers the opening sequence through the real HTTP API, in the Router's order."""

    def __init__(self, client: Any, pid: str):
        self.c, self.pid = client, pid

    def view(self) -> dict[str, Any]:
        return self.c.get(f"/api/projects/{self.pid}").json()

    def post(self, body: dict[str, Any]) -> Any:
        return self.c.post(f"/api/projects/{self.pid}/decisions", json=body)

    def decide(self, body: dict[str, Any]) -> None:
        r = settle_post(self.c, self.pid, body)
        assert r.status_code == 200, (body["kind"], r.text[:900])

    def decide_roles(self, roles: dict[str, str]) -> None:
        """Record the roles as their author answers them, column by column (BLUEPRINT §14,
        recognition's leash): the ``set_roles`` answer, then each role the server recorded
        unconfirmed (proposed below high confidence) confirmed on its own, as a user who knows the
        table does. A bulk ``set_roles`` alone confirms none of them."""
        self.decide({"kind": "set_roles", "roles": roles})
        record = self.view()["decisions"][-1]
        for column in record["decision"].get("unconfirmed") or []:
            self.decide({"kind": "confirm_reading", "reading": "role", "column": column,
                         "value": roles[column]})

    def reach(self, key: str, timeout: float = 120.0) -> dict[str, Any]:
        end = time.monotonic() + timeout
        while True:
            steps = self.view()["interview"]
            step = next(s for s in steps if s["key"] == key)
            first = next((s for s in steps if s["status"] in ("open", "waiting")), None)
            ready = step["status"] not in ("open", "waiting") or first is None or first["key"] == key
            if ready and not (step["status"] == "waiting" and step.get("waiting_on")):
                return step
            assert time.monotonic() < end, f"{key} held behind {first}"
            time.sleep(0.05)

    def answer(self, key: str, body: dict[str, Any]) -> None:
        if self.reach(key)["status"] in ("open", "waiting"):
            self.decide(body)

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

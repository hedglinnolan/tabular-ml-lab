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


class Drive:
    """Answers the opening sequence through the real HTTP API, in the Router's order."""

    def __init__(self, client: Any, pid: str):
        self.c, self.pid = client, pid

    def view(self) -> dict[str, Any]:
        return self.c.get(f"/api/projects/{self.pid}").json()

    def post(self, body: dict[str, Any]) -> Any:
        return self.c.post(f"/api/projects/{self.pid}/decisions", json=body)

    def decide(self, body: dict[str, Any]) -> None:
        r = self.post(body)
        assert r.status_code == 200, (body["kind"], r.text[:900])

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

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

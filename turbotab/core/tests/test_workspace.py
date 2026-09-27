"""Settings from the environment, and the workspace's projects on disk."""
from __future__ import annotations

import json
import os
import time
from datetime import timezone
from pathlib import Path

import pytest

from turbotab.core.config import (
    WORKER_CAP, Settings, default_workers, parse_bytes, total_memory_bytes,
)
from turbotab.core.workspace import PROJECT_ID_RE, ProjectMeta, ProjectNotFound, Workspace

SAMPLE = Path(__file__).resolve().parents[2] / "sample_data" / "clinical_risk.csv"


# ── settings ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text, expected", [
    ("1048576", 1048576), ("8G", 8 * 1024**3), ("512M", 512 * 1024**2),
    ("1.5g", int(1.5 * 1024**3)), ("2GB", 2 * 1024**3), ("64k", 64 * 1024),
    ("1T", 1024**4), (" 3 GiB ", 3 * 1024**3),
])
def test_memory_sizes_parse(text, expected):
    assert parse_bytes(text) == expected


@pytest.mark.parametrize("text", ["", "lots", "8X", "-1G", "0"])
def test_nonsense_memory_sizes_are_refused(text):
    with pytest.raises(ValueError):
        parse_bytes(text)


def test_settings_read_the_environment(tmp_path):
    s = Settings.from_env({"TURBOTAB_HOME": str(tmp_path / "home"), "TURBOTAB_MODE": "server",
                           "TURBOTAB_WORKERS": "3", "TURBOTAB_MEMORY_BUDGET": "8G"})
    assert s == Settings(home=tmp_path / "home", mode="server", workers=3,
                         memory_budget_bytes=8 * 1024**3)


def test_settings_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    s = Settings.from_env({})
    assert s.home == Path(os.path.expanduser("~/.turbotab")).absolute()
    assert s.mode == "local"
    # capped by memory as well as cores: an idle worker holds ~200 MB
    assert s.workers == default_workers()
    assert 1 <= s.workers <= min(WORKER_CAP, max(1, (os.cpu_count() or 2) - 1))
    total = total_memory_bytes()
    assert 0 < s.memory_budget_bytes and (total is None or s.memory_budget_bytes <= total)


@pytest.mark.parametrize("env, word", [
    ({"TURBOTAB_MODE": "cloud"}, "TURBOTAB_MODE"),
    ({"TURBOTAB_WORKERS": "many"}, "TURBOTAB_WORKERS"),
    ({"TURBOTAB_WORKERS": "0"}, "workers"),
    ({"TURBOTAB_MEMORY_BUDGET": "lots"}, "TURBOTAB_MEMORY_BUDGET"),
])
def test_bad_settings_say_which_variable(env, word):
    with pytest.raises(ValueError, match=word):
        Settings.from_env(env)


# ── workspace ────────────────────────────────────────────────────────────────

@pytest.fixture
def workspace(tmp_path) -> Workspace:
    return Workspace(Settings(home=tmp_path / "home", workers=1,
                              memory_budget_bytes=1024**3))


def test_a_project_is_created_with_its_layout(workspace, tmp_path):
    meta = workspace.create_project("Risk study", SAMPLE, "path")
    assert PROJECT_ID_RE.fullmatch(meta.id)
    assert meta.name == "Risk study" and meta.source_kind == "path"
    assert meta.source_path == str(SAMPLE) and meta.source_name == "clinical_risk.csv"
    assert meta.created_at.tzinfo is not None and meta.created_at.utcoffset().total_seconds() == 0

    pdir = workspace.project_dir(meta.id)
    assert pdir == workspace.home / "projects" / meta.id
    assert workspace.data_path(meta.id) == pdir / "data" / "raw.parquet"
    assert workspace.decisions_path(meta.id) == pdir / "decisions.jsonl"
    assert workspace.cache_dir(meta.id) == pdir / "cache" and (pdir / "cache").is_dir()
    assert workspace.uploads_dir().is_dir()
    assert workspace.uploads_dir() != pdir

    on_disk = json.loads((pdir / "project.json").read_text())
    assert on_disk == meta.to_dict()
    assert set(on_disk) == {"id", "name", "created_at", "source_kind", "source_path",
                            "source_name"}
    assert workspace.get(meta.id) == meta
    assert ProjectMeta.from_dict(meta.to_dict()) == meta


def test_uploads_keep_the_clients_file_name(workspace):
    staged = workspace.uploads_dir() / "u_1f2e3d.csv"
    staged.write_text("a\n1\n")
    meta = workspace.create_project("", staged, "upload", source_name="my data.csv")
    assert meta.source_kind == "upload" and meta.source_name == "my data.csv"
    assert meta.name == "my data"   # a blank name falls back to the file's stem


def test_projects_list_newest_first_and_ids_are_unique(workspace):
    made = []
    for i in range(5):
        made.append(workspace.create_project(f"p{i}", SAMPLE, "path"))
        time.sleep(0.002)
    listed = workspace.list()
    assert [m.id for m in listed] == [m.id for m in reversed(made)]
    assert len({m.id for m in made}) == 5
    assert all(m.created_at.tzinfo == timezone.utc for m in listed)


def test_unknown_or_malformed_ids_are_not_found(workspace):
    workspace.create_project("real", SAMPLE, "path")
    for bad in ("p0000000000", "../../etc", "p12345", "P0123456789", "", "p0123456789/.."):
        with pytest.raises(ProjectNotFound):
            workspace.get(bad)
        with pytest.raises(KeyError):
            workspace.data_path(bad)
        assert not workspace.exists(bad)


def test_a_damaged_project_is_not_listed(workspace):
    good = workspace.create_project("good", SAMPLE, "path")
    broken = workspace.create_project("broken", SAMPLE, "path")
    (workspace.project_dir(broken.id) / "project.json").write_text("{not json")
    (workspace.projects_root / "stray-directory").mkdir()
    assert [m.id for m in workspace.list()] == [good.id]


def test_a_project_answers_through_its_datastore(workspace):
    from turbotab.core.datastore import ingest

    meta = workspace.create_project("risk", SAMPLE, "path")
    info = ingest(SAMPLE, workspace.data_path(meta.id))
    store = workspace.datastore(meta.id)
    assert store.info().n_rows == info.n_rows > 0
    assert store.memory_budget_bytes == 1024**3

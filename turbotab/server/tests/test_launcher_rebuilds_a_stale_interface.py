"""The launcher rebuilds an interface that is out of date (W1-B; the walkable plan, blocker B1).

``dist`` was served as it was whenever it existed, so a checkout whose source had moved on served
the interface it was last built from: the app on screen was not the code on disk. The rule, tried
here on temporary directories with file times set by hand (``os.utime``), not read back from the
launcher: the interface is out of date when any source under ``frontend/src`` (or the build's own
inputs) is newer than the build, or when the version stamp the launcher left beside the build
differs from the version it would build now. A machine without Node.js keeps working with what is
built, and says so.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

LAUNCH = Path(__file__).resolve().parents[2] / "deploy" / "launch.py"
BUILT = 1_700_000_000  # seconds since the epoch: when the build in each case was made


def load():
    spec = importlib.util.spec_from_file_location("turbotab_launch_under_test", LAUNCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def launch():
    return load()


def touch(path: Path, when: int, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, "utf-8")
    os.utime(path, (when, when))
    return path


@pytest.fixture
def frontend(tmp_path: Path) -> Path:
    """A frontend folder whose build is newer than every source file."""
    root = tmp_path / "frontend"
    touch(root / "src" / "main.tsx", BUILT - 100)
    touch(root / "src" / "components" / "Stage.tsx", BUILT - 50)
    touch(root / "index.html", BUILT - 100)
    touch(root / "package.json", BUILT - 100)
    touch(root / "dist" / "index.html", BUILT, '<div id="root"></div>')
    return root


def test_a_build_newer_than_every_source_is_current(launch, frontend):
    assert launch.stale_reason(frontend, "2.0.0") is None


def test_a_source_newer_than_the_build_makes_it_stale_and_names_the_file(launch, frontend):
    touch(frontend / "src" / "components" / "Stage.tsx", BUILT + 1)
    reason = launch.stale_reason(frontend, "2.0.0")
    assert reason is not None
    assert "Stage.tsx" in reason


def test_a_new_file_in_a_deep_folder_counts(launch, frontend):
    touch(frontend / "src" / "components" / "stage" / "results" / "New.tsx", BUILT + 5)
    assert launch.stale_reason(frontend, "2.0.0") is not None


def test_a_source_the_same_second_as_the_build_is_not_newer(launch, frontend):
    touch(frontend / "src" / "main.tsx", BUILT)
    assert launch.stale_reason(frontend, "2.0.0") is None


def test_the_builds_own_inputs_count_as_source(launch, frontend):
    for name in ("index.html", "package.json"):
        touch(frontend / name, BUILT + 1)
        assert launch.stale_reason(frontend, "2.0.0") is not None, name
        touch(frontend / name, BUILT - 1)
    assert launch.stale_reason(frontend, "2.0.0") is None


def test_packages_and_the_build_output_are_not_source(launch, frontend):
    touch(frontend / "node_modules" / "react" / "index.js", BUILT + 500)
    touch(frontend / "dist" / "assets" / "app.js", BUILT + 500)
    assert launch.stale_reason(frontend, "2.0.0") is None


def test_a_missing_build_is_reported_as_missing(launch, frontend):
    (frontend / "dist" / "index.html").unlink()
    assert launch.stale_reason(frontend, "2.0.0") == "not built"


def test_a_version_stamp_that_differs_makes_the_build_stale(launch, frontend):
    touch(frontend / "dist" / launch.BUILD_STAMP, BUILT, "1.9.0")
    reason = launch.stale_reason(frontend, "2.0.0")
    assert reason is not None
    assert "1.9.0" in reason and "2.0.0" in reason


def test_a_version_stamp_that_matches_leaves_it_current(launch, frontend):
    touch(frontend / "dist" / launch.BUILD_STAMP, BUILT, "2.0.0\n")
    assert launch.stale_reason(frontend, "2.0.0") is None


def test_a_build_with_no_stamp_is_judged_by_file_times_alone(launch, frontend):
    # A developer's own `npm run build` leaves no stamp; it is not rebuilt for lacking one.
    assert not (frontend / "dist" / launch.BUILD_STAMP).exists()
    assert launch.stale_reason(frontend, "2.0.0") is None


def test_the_version_comes_from_the_server_package_without_importing_it(launch):
    wanted = None
    text = (Path(__file__).resolve().parents[1] / "__init__.py").read_text("utf-8")
    for line in text.splitlines():
        if line.startswith("__version__"):
            wanted = line.split("=", 1)[1].strip().strip("\"'")
    assert wanted and launch.current_version() == wanted


# ── what the launcher does about it ──────────────────────────────────────────


class FakeNode:
    """Stands in for `shutil.which` and the build: records what was run, builds a fresh dist."""

    def __init__(self, frontend: Path, *, present: bool, builds: bool = True):
        self.frontend = frontend
        self.present = present
        self.builds = builds
        self.ran: list[list[str]] = []

    def which(self, name: str):
        return f"/fake/{name}" if self.present else None

    def run(self, cmd, **kwargs):
        self.ran.append(list(cmd))
        if cmd[1:3] == ["run", "build"] and self.builds:
            (self.frontend / "dist").mkdir(exist_ok=True)
            (self.frontend / "dist" / "index.html").write_text('<div id="root"></div>', "utf-8")
            return 0
        return 0 if self.builds else 1


@pytest.fixture
def stale(frontend):
    touch(frontend / "src" / "main.tsx", BUILT + 10)
    return frontend


def use(launch, monkeypatch, frontend: Path, node: FakeNode, version=(22, 1)):
    monkeypatch.setattr(launch, "FRONTEND", frontend)
    monkeypatch.setattr(launch, "DIST", frontend / "dist")
    monkeypatch.setattr(launch.shutil, "which", node.which)
    monkeypatch.setattr(launch, "node_version", lambda _node: version)
    monkeypatch.setattr(launch, "run", node.run)
    (frontend / "node_modules").mkdir(exist_ok=True)


def test_a_current_build_is_served_as_it_is(launch, monkeypatch, frontend, capsys):
    node = FakeNode(frontend, present=True)
    use(launch, monkeypatch, frontend, node)
    assert launch.ensure_interface() == "built"
    assert node.ran == []


def test_a_stale_build_is_rebuilt_and_stamped_with_the_version(launch, monkeypatch, stale, capsys):
    node = FakeNode(stale, present=True)
    use(launch, monkeypatch, stale, node)
    assert launch.ensure_interface() == "built"
    assert ["/fake/npm", "run", "build"] in node.ran
    assert (stale / "dist" / launch.BUILD_STAMP).read_text("utf-8").strip() == launch.current_version()
    assert "out of date" in capsys.readouterr().out
    # and the next start finds it current: the build is stamped after the sources it was built from
    os.utime(stale / "dist" / "index.html", (BUILT + 20, BUILT + 20))
    assert launch.stale_reason(stale, launch.current_version()) is None


def test_without_node_a_stale_build_is_served_and_the_launcher_says_so(launch, monkeypatch, stale, capsys):
    node = FakeNode(stale, present=False)
    use(launch, monkeypatch, stale, node)
    assert launch.ensure_interface() == "stale"
    assert node.ran == []
    out = capsys.readouterr().out
    assert "out of date" in out
    assert "Node.js" in out
    assert (stale / "dist" / "index.html").is_file()  # what was built is left alone


def test_with_too_old_a_node_it_also_keeps_what_is_built(launch, monkeypatch, stale, capsys):
    node = FakeNode(stale, present=True)
    use(launch, monkeypatch, stale, node, version=(18, 0))
    assert launch.ensure_interface() == "stale"
    assert node.ran == []
    assert "18.0" in capsys.readouterr().out


def test_without_node_and_no_build_nothing_is_built_and_it_says_how_to_get_it(launch, monkeypatch, frontend, capsys):
    (frontend / "dist" / "index.html").unlink()
    node = FakeNode(frontend, present=False)
    use(launch, monkeypatch, frontend, node)
    assert launch.ensure_interface() == "missing"
    assert "nodejs.org" in capsys.readouterr().out


def test_a_rebuild_that_fails_keeps_the_launcher_going(launch, monkeypatch, stale, capsys):
    node = FakeNode(stale, present=True, builds=False)
    use(launch, monkeypatch, stale, node)
    assert launch.ensure_interface() in ("stale", "missing")
    assert "did not build" in capsys.readouterr().out

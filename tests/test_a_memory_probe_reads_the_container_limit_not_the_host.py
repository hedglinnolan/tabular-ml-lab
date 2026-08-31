"""`utils/host_resources.py` — the probe reads the cgroup, and fails to `None`.

Two claims, and the first one is the reason the module exists rather than a
one-line `psutil.virtual_memory().available` at the call site.

**A container's ceiling is not the host's RAM.** `/proc/meminfo` is not
namespaced, so psutil inside a container reports the machine. This app's
documented enterprise path runs under `memory: ${APP_MEMORY_LIMIT:-4g}` in
`docker-compose.yml`, and `UNIVERSITY_DEPLOYMENT.md` tells omics deployers to
raise it — so the cgroup limit is a knob real deployers are instructed to turn,
and a probe that could not see it would contradict the app's own deployment
guide on the deployments that matter most.

**And "cannot estimate" is not "unlimited".** The lean-install case, where
psutil is absent, is the one where being wrong is cheapest to cause and hardest
to notice: a probe that raised would take down an upload page, and a probe that
returned a large number would wave through the file it was built to stop. It
returns `None`, and the docstring on `available_memory_bytes()` says in as many
words that `None` is not a ceiling.

**This runs on Windows, which is why the parsing is separated from the
reading.** `/sys/fs/cgroup` does not exist on the machines this repository is
developed on, and neither do cgroups. Every sentinel and every malformed-file
case is therefore exercised against `_parse_limit`, which takes a string — a
parser that needs a Linux kernel to test is a parser that is tested nowhere.
"""
from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import host_resources                                 # noqa: E402


# ── the fixture, and the note it owes the write guard ────────────────────────

def _cgroup(tmp_path: Path, relative: str, text: str) -> Path:
    """A fake cgroup root holding one limit file, and its path.

    **The only two write-shaped calls in this file, kept in one helper on
    purpose.** `tests/test_no_test_writes_a_path_git_tracks.py` counts write
    destinations its resolver cannot compute, and a destination built from the
    `tmp_path` fixture never resolves. These two cannot be avoided: the code
    under test calls `Path.read_text`, so exercising it needs a real file at a
    real path, and the v1 name is nested one directory down exactly as the
    kernel nests it. Everything that does not need a file is tested against
    `_parse_limit` instead, which is why there are two here and not eight.
    """
    target = tmp_path.joinpath(*relative.split("/"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return tmp_path


@contextlib.contextmanager
def _psutil_absent():
    """psutil genuinely ABSENT in this interpreter, not merely broken.

    A `psutil.py` on the path that raises would still be FOUND, and *found* and
    *importable* are different predicates — the trap
    `tests/test_the_launch_command_refuses_a_stack_it_cannot_import.py` spends
    a paragraph on. A `meta_path` finder that raises makes both `find_spec` and
    a real import fail exactly as an uninstalled package does.

    `sys.modules` is cleared and restored too, because the import under test is
    inside the function and a cached module would satisfy it without ever
    reaching the finder.
    """
    class _Absent:
        def find_spec(self, name, path=None, target=None):
            if name == "psutil" or name.startswith("psutil."):
                raise ModuleNotFoundError(f"No module named {name!r}")
            return None

    finder = _Absent()
    cached = {k: v for k, v in sys.modules.items()
              if k == "psutil" or k.startswith("psutil.")}
    for name in cached:
        del sys.modules[name]
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
        sys.modules.update(cached)


# ── the parser, where every sentinel lives ───────────────────────────────────

def test_a_v2_numeric_limit_is_read_as_bytes():
    """The ordinary case: `memory.max` under a 4 GiB container."""
    assert host_resources._parse_limit("4294967296\n") == 4 * 1024 ** 3


def test_the_v2_word_max_is_not_a_limit():
    """cgroup v2 spells "no limit" as a word, not a number."""
    assert host_resources._parse_limit("max\n") is None


@pytest.mark.parametrize("sentinel", [
    "9223372036854771712",    # PAGE_COUNTER_MAX, the canonical v1 value
    "9223372036854775807",    # INT64_MAX, reported by other kernels
    "18446744073709551615",   # the unsigned saturation, seen on some configs
])
def test_a_v1_saturated_sentinel_is_not_a_limit(sentinel):
    """Three spellings, because an equality test would fail OPEN on two of them.

    Matching only the canonical literal would read the other two as a real
    ceiling of eight million terabytes — indistinguishable from having no gate,
    on exactly the machines the equality test does not cover. The threshold in
    `_NO_LIMIT_AT_OR_ABOVE` is what makes all three the same answer.
    """
    assert host_resources._parse_limit(sentinel) is None


def test_the_threshold_still_admits_a_real_machine():
    """The other half of the threshold's polarity.

    A cutoff that rejected sentinels by rejecting large numbers would also
    reject the large-memory hosts this app is deployed on, and the failure
    would look like the gate working. 1 TiB is a plausible departmental server
    and must survive.
    """
    assert host_resources._parse_limit(str(1024 ** 4)) == 1024 ** 4


@pytest.mark.parametrize("junk", ["", "   \n", "unlimited", "4 GB", "0", "-1"])
def test_an_unreadable_value_reports_no_reading_rather_than_a_guess(junk):
    """Including `0`, which is arithmetically a limit and practically a bug.

    Zero would forbid every upload. If a kernel ever really reports it the
    container is unusable anyway, so "cannot estimate" costs nothing and stops
    the probe rejecting everything over a value it did not expect.
    """
    assert host_resources._parse_limit(junk) is None


# ── the reader, against real files ───────────────────────────────────────────

def test_no_cgroup_files_at_all_means_no_container_ceiling(tmp_path):
    """Windows, macOS, and any un-containerized Linux host.

    Zero writes: `tmp_path` is already an empty directory, which is precisely
    the shape being tested.
    """
    assert host_resources._cgroup_limit_bytes(root=tmp_path) is None


def test_the_v2_limit_file_is_read_from_the_path_the_kernel_uses(tmp_path):
    root = _cgroup(tmp_path, "memory.max", "4294967296\n")
    assert host_resources._cgroup_limit_bytes(root=root) == 4 * 1024 ** 3


def test_the_v1_limit_file_is_read_from_the_path_the_kernel_uses(tmp_path):
    """The nested v1 name, end to end.

    Worth its own file rather than folding into the parser tests: the one thing
    a parser test cannot catch is a typo in `memory/memory.limit_in_bytes`, and
    nothing on a developer's Windows box or in a non-containerized CI job would
    ever notice one.
    """
    root = _cgroup(tmp_path, "memory/memory.limit_in_bytes", "2147483648\n")
    assert host_resources._cgroup_limit_bytes(root=root) == 2 * 1024 ** 3


def test_a_v2_file_saying_max_falls_through_instead_of_reporting_a_limit(
        tmp_path):
    """A v2 host with no memory cap set. The file exists and says nothing."""
    root = _cgroup(tmp_path, "memory.max", "max\n")
    assert host_resources._cgroup_limit_bytes(root=root) is None


def test_a_directory_where_the_limit_file_should_be_does_not_raise(tmp_path):
    """The read is guarded by `OSError`, not `FileNotFoundError`.

    A cgroup path can be present and unreadable — permissions under a
    restricted runtime, `EISDIR` on a mount that is not shaped as expected. A
    probe that raised there would take down an upload page over a file it only
    wanted to consult. A directory at the v2 name reproduces that class
    portably; `read_text` raises `IsADirectoryError` on Linux and
    `PermissionError` on Windows, and both are `OSError`.
    """
    root = _cgroup(tmp_path, "memory.max/placeholder", "ignored")
    assert host_resources._cgroup_limit_bytes(root=root) is None


# ── a ceiling is not headroom ────────────────────────────────────────────────
#
# The container case fails OPEN if only the limit is read. Nothing else can see
# what the container has spent — psutil's `available` reads the un-namespaced
# host, so on an idle host it is larger than the whole container and the `min`
# returns the container's full size with the interpreter, pandas, torch and
# Streamlit already resident and counted as free.


def _cgroup_pair(tmp_path: Path, limit: str, usage: str) -> Path:
    """A v2 cgroup root reporting both a ceiling and what it has spent."""
    _cgroup(tmp_path, "memory.max", limit)
    (tmp_path / "memory.current").write_text(usage, encoding="utf-8")
    return tmp_path


def test_what_the_container_already_spent_is_subtracted_from_its_ceiling(
        tmp_path):
    """4 GiB granted, 1 GiB resting: 3 GiB is what remains, not 4."""
    root = _cgroup_pair(tmp_path, str(4 * 1024 ** 3), str(1024 ** 3))
    assert host_resources._cgroup_headroom_bytes(root) == 3 * 1024 ** 3


def test_a_cgroup_at_its_limit_reports_no_headroom_rather_than_a_negative(
        tmp_path):
    """`memory.current` can exceed `memory.max` transiently under reclaim. A
    negative figure would flow into the `min` and make every later comparison
    read backwards, so it is floored."""
    root = _cgroup_pair(tmp_path, str(1024 ** 3), str(2 * 1024 ** 3))
    assert host_resources._cgroup_headroom_bytes(root) == 0


def test_an_unreadable_usage_file_falls_back_to_the_bare_ceiling(tmp_path):
    """The looser answer, chosen deliberately: it is what this returned before
    it could read usage at all, so a kernel exposing one file and not the other
    is no worse off than it was."""
    root = _cgroup(tmp_path, "memory.max", str(4 * 1024 ** 3))
    assert host_resources._cgroup_headroom_bytes(root) == 4 * 1024 ** 3


def test_the_v1_usage_file_is_read_from_the_path_the_kernel_uses(tmp_path):
    """The nested v1 name, end to end. A parser test cannot catch a typo in
    `memory/memory.usage_in_bytes`; only a real file at the real path can."""
    _cgroup(tmp_path, "memory/memory.limit_in_bytes", str(8 * 1024 ** 3))
    (tmp_path / "memory" / "memory.usage_in_bytes").write_text(
        str(2 * 1024 ** 3), encoding="utf-8")
    assert host_resources._cgroup_headroom_bytes(tmp_path) == 6 * 1024 ** 3


def test_no_cgroup_means_no_headroom_reading_rather_than_zero(tmp_path):
    """`None` and `0` are opposite instructions to the gate: one means the
    un-containerized path where psutil's figure stands alone, the other would
    refuse every upload on the planet."""
    assert host_resources._cgroup_headroom_bytes(tmp_path) is None


@pytest.mark.parametrize("junk", ["", "   ", "max", "not-a-number", "-1"])
def test_an_unreadable_usage_value_is_no_reading_rather_than_a_guess(
        tmp_path, junk):
    """A usage file that says something unexpected falls back to the ceiling —
    the same fail-loose-not-wrong rule as a missing file."""
    root = _cgroup_pair(tmp_path, str(4 * 1024 ** 3), junk)
    assert host_resources._cgroup_headroom_bytes(root) == 4 * 1024 ** 3


# ── the public surface ───────────────────────────────────────────────────────

def test_a_missing_psutil_reports_nothing_rather_than_raising():
    """The lean install. `None` is the contract, and it is not a ceiling."""
    with _psutil_absent():
        assert host_resources.available_memory_bytes() is None


def test_the_container_ceiling_wins_over_the_hosts_free_memory(
        tmp_path, monkeypatch):
    """THE CLAIM THIS MODULE EXISTS FOR, through the public function.

    64 MiB is below any machine's free memory, so if the returned figure is the
    cgroup limit rather than psutil's reading, the minimum is being taken and
    the container is being seen. Skipped rather than failed where psutil is not
    installed: it is in `requirements.txt` and therefore present in CI, and a
    red local suite that is green in CI is the misleading baseline this
    repository's `requirements-dev.txt` header warns about.
    """
    pytest.importorskip("psutil")
    root = _cgroup(tmp_path, "memory.max", str(64 * 1024 ** 2))
    monkeypatch.setattr(host_resources, "_CGROUP_ROOT", root)
    assert host_resources.available_memory_bytes() == 64 * 1024 ** 2


def test_the_public_figure_is_the_containers_headroom_not_its_size(
        tmp_path, monkeypatch):
    """The same claim one level up, with usage in play. Both numbers here are
    far below any host's free memory, so the cgroup pair is what the `min`
    returns and the arithmetic is visible in the result."""
    pytest.importorskip("psutil")
    root = _cgroup_pair(tmp_path, str(256 * 1024 ** 2), str(64 * 1024 ** 2))
    monkeypatch.setattr(host_resources, "_CGROUP_ROOT", root)
    assert host_resources.available_memory_bytes() == 192 * 1024 ** 2


def test_with_no_cgroup_the_figure_is_psutils_own_reading(tmp_path,
                                                          monkeypatch):
    """The un-containerized path, and the control on the test above.

    If the minimum were being taken against something wrong, this would drift
    from psutil's own number and the test above would still pass.
    """
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(host_resources, "_CGROUP_ROOT", tmp_path)
    reading = host_resources.available_memory_bytes()
    assert reading is not None and reading > 0
    # Not equality: free memory moves between the two calls. Same order of
    # magnitude is the claim — that this is psutil's figure, unmodified.
    assert reading == pytest.approx(psutil.virtual_memory().available,
                                    rel=0.5)


def test_the_cpu_count_is_a_positive_number_or_nothing():
    """A plain inventory count. It is NOT cgroup-quota-aware and says so.

    Asserted here so the asymmetry with the memory probe is visible in the
    tests too, rather than only in a docstring somebody has to open.
    """
    count = host_resources.cpu_count()
    assert count is None or (isinstance(count, int) and count > 0)


def test_the_probe_imports_without_streamlit():
    """Both front doors have to be able to reach it.

    Most of `utils/` is host code — `utils/perf_cache.py` imports Streamlit at
    module scope — and the admission gate is wanted in TurboTab's FastAPI
    process as well as in the Streamlit pages. Cheap to assert, and it is the
    property that would be lost silently the first time a convenience import
    was added at the top of the module.
    """
    source = Path(host_resources.__file__).read_text(encoding="utf-8")
    assert "import streamlit" not in source

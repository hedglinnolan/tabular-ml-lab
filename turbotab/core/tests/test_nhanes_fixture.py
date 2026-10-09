"""The NHANES export is tracked, gzipped, as ``fixtures/nhanes.csv.gz`` (public-domain CDC data).

Every NHANES test reads it through ``stage_harness.NHANES``, so none skips in a fresh clone, and
what they read is the export byte for byte: the decompressed fixture's SHA-256 is the export's,
and equals the untracked original's wherever that file is still on the machine.

The cache it is decompressed to is checked as well (the verifier's hygiene notes): a corrupt
fixture leaves no partial file behind, a cached copy is compared with the fixture's own bytes
every time it is reused and written again when it differs, and the copy is readable by every user
of a shared host (mode 0644, not the 0600 a temporary file is created with).
"""
from __future__ import annotations

import gzip
import hashlib
import os
import stat
import tempfile
from pathlib import Path

import pytest

from turbotab.core.tests import stage_harness as H

TOY = b"SEQN,RIDAGEYR\n1,34\n2,61\n"  # a stand-in export: the cache does not care what it holds


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def toy_fixture(tmp_path, monkeypatch) -> Path:
    """A gzipped stand-in for the fixture, cached under its own temporary folder."""
    gz = tmp_path / "nhanes.csv.gz"
    gz.write_bytes(gzip.compress(TOY))
    monkeypatch.setattr(H, "NHANES_GZ", gz)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    return gz


def test_the_tests_read_the_tracked_fixture_before_any_untracked_copy(monkeypatch):
    monkeypatch.delenv("TURBOTAB_NHANES_CSV", raising=False)
    assert H.NHANES_GZ.is_file()
    assert H._nhanes() == H.nhanes_fixture()


def test_the_decompressed_fixture_is_the_export_byte_for_byte():
    path = H.nhanes_fixture()
    assert path.name == "_tt_tmp_nhanes.csv"  # the name truths.FIXTURE_TRUTHS keys on
    assert _sha256(path) == H.NHANES_SHA256
    assert H.nhanes_fixture() == path  # decompressed once, then reused


def test_the_untracked_original_where_present_is_the_same_bytes():
    originals = {H.REPO / "_tt_tmp_nhanes.csv", H._main_checkout(H.REPO) / "_tt_tmp_nhanes.csv"}
    present = sorted(p for p in originals if p.is_file())
    if not present:
        pytest.skip("no untracked original of the export on this machine to compare")
    for original in present:
        assert _sha256(original) == H.NHANES_SHA256, original


def test_a_corrupt_fixture_leaves_no_partial_file_in_the_cache(toy_fixture):
    """A truncated fixture fails loudly, and the cache holds no stray ``.part`` file after it."""
    toy_fixture.write_bytes(gzip.compress(TOY * 50)[:40])
    with pytest.raises((EOFError, gzip.BadGzipFile, OSError)):
        H.nhanes_fixture()
    left = [p.name for p in Path(tempfile.gettempdir()).rglob("*") if p.is_file()]
    assert left == [], left


def test_a_cached_copy_is_checked_against_the_fixture_each_time_it_is_reused(toy_fixture):
    """A cached copy that differs from the fixture's bytes (a partial write, an edit) is written
    again from the fixture, never handed to the tests as it is."""
    path = H.nhanes_fixture()
    assert path.read_bytes() == TOY
    path.write_bytes(TOY.replace(b"61", b"16"))
    again = H.nhanes_fixture()
    assert again == path and again.read_bytes() == TOY
    path.write_bytes(TOY[:10])
    assert H.nhanes_fixture().read_bytes() == TOY


def test_the_cached_copy_is_readable_by_every_user_of_the_host(toy_fixture):
    """The cache sits in the shared temporary folder: another user of a Linux host finds the file
    there, so it must be able to read it (0644, not mkstemp's 0600)."""
    path = H.nhanes_fixture()
    assert stat.S_IMODE(path.stat().st_mode) == 0o644


def test_a_folder_another_user_holds_sends_the_copy_to_this_user_s_own(toy_fixture):
    """Another user's cache folder that this user can neither read in nor write to (a copy left
    0600 by an earlier harness): the copy goes to a folder of this user's own beside it."""
    if not hasattr(os, "geteuid") or os.geteuid() == 0:
        pytest.skip("permissions do not bind root, and Windows has no POSIX modes")
    shared = H.nhanes_fixture().parent
    cached = shared / H.NHANES_NAME
    cached.chmod(0o000)
    shared.chmod(0o555)
    try:
        path = H.nhanes_fixture()
        assert path.parent != shared and path.parent.name.startswith(shared.name)
        assert path.read_bytes() == TOY
    finally:
        shared.chmod(0o755)
        cached.chmod(0o644)

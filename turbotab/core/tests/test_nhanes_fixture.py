"""The NHANES export is tracked, gzipped, as ``fixtures/nhanes.csv.gz`` (public-domain CDC data).

Every NHANES test reads it through ``stage_harness.NHANES``, so none skips in a fresh clone, and
what they read is the export byte for byte: the decompressed fixture's SHA-256 is the export's,
and equals the untracked original's wherever that file is still on the machine.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from turbotab.core.tests import stage_harness as H


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    for original in (p for p in originals if p.is_file()):
        assert _sha256(original) == H.NHANES_SHA256, original

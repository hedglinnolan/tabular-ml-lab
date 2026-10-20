"""A file moved into place over one another process holds open lands on Windows too.

On a Windows runner (the windows probe, 2026-10-20) stage jobs failed at ``graph.write_artifact``'s
``os.replace`` of the ``latest`` pointer: Windows refuses to replace a file another process has
open (PermissionError), where POSIX replaces it under the reader. The server and its workers read
and write that pointer all the time. The rule, tried here with ``os.replace`` made to refuse as
Windows does, not on Windows itself: on Windows a refused replace is tried again until it lands or
the wait runs out, and then it raises as before; elsewhere it raises at once, as ``os.replace``
does.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from turbotab.core import atomic, graph


def refusing(times: int, only: str | None = None):
    """``os.replace`` that refuses its first ``times`` calls (onto a file named ``only``, if given),
    as Windows does while another process holds the target open."""
    real = os.replace
    calls = {"refused": 0, "all": 0}

    def replace(src, dst):
        calls["all"] += 1
        if (only is None or Path(dst).name == only) and calls["refused"] < times:
            calls["refused"] += 1
            raise PermissionError(5, "Access is denied", str(dst))
        return real(src, dst)

    return replace, calls


def test_on_windows_a_refused_replace_is_tried_again_until_it_lands(tmp_path, monkeypatch):
    monkeypatch.setattr(atomic, "WINDOWS", True)
    replace, calls = refusing(2)
    monkeypatch.setattr(os, "replace", replace)
    src, dst = tmp_path / "new", tmp_path / "old"
    src.write_text("new", "utf-8")
    dst.write_text("old", "utf-8")
    atomic.replace(src, dst)
    assert dst.read_text("utf-8") == "new" and not src.exists()
    assert calls == {"refused": 2, "all": 3}


def test_on_windows_a_replace_refused_past_the_wait_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(atomic, "WINDOWS", True)
    monkeypatch.setattr(atomic, "WAIT_SECONDS", 0.05)
    replace, calls = refusing(10**6)
    monkeypatch.setattr(os, "replace", replace)
    src = tmp_path / "new"
    src.write_text("new", "utf-8")
    with pytest.raises(PermissionError):
        atomic.replace(src, tmp_path / "old")
    assert calls["all"] > 1 and src.exists()


def test_elsewhere_a_refused_replace_raises_at_once(tmp_path, monkeypatch):
    monkeypatch.setattr(atomic, "WINDOWS", False)
    replace, calls = refusing(1)
    monkeypatch.setattr(os, "replace", replace)
    src = tmp_path / "new"
    src.write_text("new", "utf-8")
    with pytest.raises(PermissionError):
        atomic.replace(src, tmp_path / "old")
    assert calls["all"] == 1


def test_a_stage_s_latest_pointer_lands_though_a_reader_held_it(tmp_path, monkeypatch):
    monkeypatch.setattr(atomic, "WINDOWS", True)
    root = tmp_path / "cache"
    graph.write_artifact(root, "S", "k1", 1, {"a": 1})
    replace, calls = refusing(1, only=graph.LATEST)
    monkeypatch.setattr(os, "replace", replace)
    graph.write_artifact(root, "S", "k2", 1, {"a": 2})
    assert calls["refused"] == 1
    assert (root / "S" / graph.LATEST).read_text("utf-8") == "k2"
    assert graph.latest_key(root, "S") == "k2"

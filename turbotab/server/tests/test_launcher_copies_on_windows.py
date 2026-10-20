"""The launcher installs by copying on Windows, where a synced user folder refuses hard links.

A first start on a Windows laptop whose user folder OneDrive manages stopped with uv's "failed to
hardlink file ... The cloud operation cannot be performed on a file with incompatible hardlinks
(os error 396)", and the launcher blamed the internet connection. uv hard-links from its cache by
default on Windows; its own ``--link-mode copy`` (``uv pip install --help``) installs the same
files without a link. The rule, tried here on the command the launcher would run, not on a real
install: on Windows, a uv install copies unless the researcher chose a mode in ``UV_LINK_MODE``;
elsewhere, and through pip, the command is as it was.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

LAUNCH = Path(__file__).resolve().parents[2] / "deploy" / "launch.py"
PYTHON = Path("env") / "Scripts" / "python.exe"


def load():
    spec = importlib.util.spec_from_file_location("turbotab_launch_under_test", LAUNCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def launch():
    return load()


def link_mode(cmd: list[str]) -> str | None:
    return cmd[cmd.index("--link-mode") + 1] if "--link-mode" in cmd else None


def test_a_uv_install_on_windows_copies(launch, monkeypatch):
    monkeypatch.setattr(launch, "WINDOWS", True)
    cmd = launch.install_command("uv.exe", PYTHON, {})
    assert cmd[:3] == ["uv.exe", "pip", "install"]
    assert link_mode(cmd) == "copy"


def test_a_mode_the_researcher_chose_is_left_to_uv(launch, monkeypatch):
    monkeypatch.setattr(launch, "WINDOWS", True)
    cmd = launch.install_command("uv.exe", PYTHON, {"UV_LINK_MODE": "hardlink"})
    assert link_mode(cmd) is None


def test_a_uv_install_elsewhere_keeps_uv_s_default(launch, monkeypatch):
    monkeypatch.setattr(launch, "WINDOWS", False)
    cmd = launch.install_command("/usr/local/bin/uv", Path("env/bin/python"), {})
    assert cmd[:3] == ["/usr/local/bin/uv", "pip", "install"]
    assert link_mode(cmd) is None


def test_a_pip_install_has_no_link_mode(launch, monkeypatch):
    monkeypatch.setattr(launch, "WINDOWS", True)
    cmd = launch.install_command(None, PYTHON, {})
    assert cmd[:4] == [str(PYTHON), "-m", "pip", "install"]
    assert link_mode(cmd) is None


@pytest.mark.parametrize("windows", [True, False])
def test_both_installers_take_the_tested_versions(launch, monkeypatch, windows):
    monkeypatch.setattr(launch, "WINDOWS", windows)
    for uv in ("uv", None):
        cmd = launch.install_command(uv, PYTHON, {})
        assert cmd[cmd.index("-c") + 1] == str(launch.CONSTRAINTS)
        assert cmd[cmd.index("-r") + 1] == str(launch.REQUIREMENTS)

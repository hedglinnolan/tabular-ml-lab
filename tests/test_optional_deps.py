"""Optional add-ons must be installable from inside the app.

The app ships without torch (~1.1 GB for one of 22 models). That is only
acceptable if enabling it costs one click: the audience for this app has never
opened a terminal, so "run uv pip install torch" is not a recourse.
"""
from __future__ import annotations

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.optional_deps import (  # noqa: E402
    ADDONS, InstallResult, can_install, install, installer_command, is_available,
)


def test_every_addon_declares_what_the_user_is_agreeing_to():
    for name, spec in ADDONS.items():
        assert spec["package"] and spec["label"] and spec["size_hint"] and spec["why"]
        # The button must be able to state the cost up front.
        assert any(ch.isdigit() for ch in spec["size_hint"])


def test_install_targets_the_running_interpreter_not_system_python():
    """Installing into the wrong environment would either fail silently or
    modify software outside the app."""
    cmd = installer_command("torch")
    assert sys.executable in cmd


def test_installer_prefers_uv_but_always_has_a_fallback():
    cmd = installer_command("torch")
    assert cmd[0].endswith(("uv", "uv.exe")) or cmd[:3] == [sys.executable, "-m", "pip"]


def test_unknown_addon_is_refused_not_crashed():
    res = install("not_a_real_addon")
    assert isinstance(res, InstallResult) and not res.ok
    assert "optional add-on" in res.message


def test_already_installed_is_reported_not_reinstalled():
    """pandas is certainly present; the check must short-circuit."""
    ADDONS["pandas"] = {"package": "pandas", "label": "Pandas",
                        "size_hint": "0 MB", "why": "test"}
    try:
        res = install("pandas")
        assert res.ok and "already installed" in res.message
    finally:
        ADDONS.pop("pandas", None)


def test_refuses_to_touch_a_system_wide_python():
    """Outside a virtualenv the app must not install anything — that would
    modify the user's other software."""
    ok, why = can_install()
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    assert ok == in_venv
    if not ok:
        assert "system-wide" in why or "read-only" in why


def test_availability_check_is_honest():
    assert is_available("pandas") is True
    assert is_available("a_module_that_does_not_exist_xyz") is False


def test_failure_messages_are_written_for_a_non_programmer():
    res = install("not_a_real_addon")
    for jargon in ("Traceback", "stderr", "subprocess", "ImportError"):
        assert jargon not in res.message


# ── enabled add-ons must survive relaunches AND app updates ──────────────

def test_enabling_is_recorded_and_idempotent(tmp_path, monkeypatch):
    """The launcher rebuilds the environment whenever requirements.txt changes.
    Add-ons are not in requirements.txt, so without a record the user would
    silently lose the neural network after updating the app."""
    import utils.optional_deps as od
    monkeypatch.setattr(od, "_app_root", lambda: tmp_path)

    assert od.remembered_addons() == []
    od._remember_addon("torch")
    assert od.remembered_addons() == ["torch"]
    od._remember_addon("torch")
    assert od.remembered_addons() == ["torch"]          # no duplicate
    od._remember_addon("something_else")
    assert od.remembered_addons() == ["torch", "something_else"]


def test_recording_failure_never_breaks_a_successful_install(monkeypatch):
    """Losing the record costs one extra click later; it must not turn a
    completed install into a reported failure."""
    import utils.optional_deps as od
    monkeypatch.setattr(od, "_app_root", lambda: (_ for _ in ()).throw(OSError("nope")))
    od._remember_addon("torch")        # must not raise
    assert od.remembered_addons() == []


def test_addons_file_is_not_committed():
    """It is per-machine state, like .venv."""
    gitignore = os.path.join(PROJECT_ROOT, ".gitignore")
    assert ".addons" in open(gitignore).read()


@pytest.mark.parametrize("launcher,needle", [
    ("launcher/posix_launch.sh", ".addons"),
    ("launcher/windows_setup.ps1", ".addons"),
])
def test_launchers_restore_enabled_addons_after_a_rebuild(launcher, needle):
    text = open(os.path.join(PROJECT_ROOT, launcher)).read()
    assert needle in text
    # The restore must run AFTER requirements are installed, or the rebuild
    # would overwrite it.
    assert text.index("requirements.txt") < text.index(needle) or \
        text.index("$Reqs") < text.index(needle)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

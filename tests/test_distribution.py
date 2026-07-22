"""Guards for the one-click download distribution.

The download IS the repo (GitHub's archive zip of main), so these tests pin
the things that would silently break a non-technical user's double-click:
launcher files present, executable bits recorded in git (GitHub's zip
preserves them), icons valid, and the README download path intact.
"""
import os
import subprocess

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(rel):
    with open(os.path.join(PROJECT_ROOT, rel), encoding="utf-8") as f:
        return f.read()


class TestLauncherFiles:
    def test_entry_points_exist(self):
        for rel in ("Start Tabular ML Lab.bat", "Start Tabular ML Lab.command",
                    "launcher/posix_launch.sh", "launcher/windows_setup.ps1"):
            assert os.path.exists(os.path.join(PROJECT_ROOT, rel)), rel

    def test_shell_launchers_executable_in_git(self):
        """GitHub's archive zip preserves the git mode bit — without 100755
        the Mac .command opens in a text editor instead of running."""
        out = subprocess.run(
            ["git", "ls-files", "-s", "Start Tabular ML Lab.command",
             "launcher/posix_launch.sh"],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        ).stdout
        lines = [ln for ln in out.strip().splitlines() if ln]
        assert len(lines) == 2, f"launchers not tracked: {out!r}"
        for ln in lines:
            assert ln.startswith("100755"), f"missing executable bit: {ln}"

    def test_python_pin_matches_ci(self):
        assert _read(".python-version").strip() == "3.12"

    def test_launcher_artifacts_gitignored(self):
        gi = _read(".gitignore")
        for pat in (".tools/", "Tabular ML Lab.app/", ".venv-stamp"):
            assert pat in gi, f".gitignore missing {pat}"

    def test_bat_delegates_to_ps1_with_bypass(self):
        bat = _read("Start Tabular ML Lab.bat")
        assert "-ExecutionPolicy Bypass" in bat
        assert "windows_setup.ps1" in bat

    def test_posix_launcher_covers_both_platforms(self):
        sh = _read("launcher/posix_launch.sh")
        assert "apple-darwin" in sh and "unknown-linux-gnu" in sh
        assert "astral.sh/uv/install.sh" in sh  # fallback path
        assert "Tabular ML Lab.app" in sh       # on-machine .app generation
        assert "gatherUsageStats false" in sh

    def test_windows_launcher_creates_shortcuts(self):
        ps1 = _read("launcher/windows_setup.ps1")
        assert "CreateShortcut" in ps1
        assert "icon.ico" in ps1
        assert "aarch64" in ps1  # ARM64 Windows covered


class TestIconAssets:
    def test_icons_exist_and_valid(self):
        from PIL import Image

        png = Image.open(os.path.join(PROJECT_ROOT, "launcher/icon.png"))
        assert png.size == (1024, 1024)
        ico = Image.open(os.path.join(PROJECT_ROOT, "launcher/icon.ico"))
        assert max(ico.size) >= 256
        icns = Image.open(os.path.join(PROJECT_ROOT, "launcher/icon.icns"))
        assert max(icns.size) >= 512


class TestReadmeDownloadPath:
    def test_download_link_present(self):
        readme = _read("README.md")
        assert "archive/refs/heads/main.zip" in readme
        assert "Start Tabular ML Lab.bat" in readme
        assert "Start Tabular ML Lab.command" in readme
        # the two trust prompts users WILL hit must be documented
        assert "Run anyway" in readme
        assert "Right-click" in readme or "right-click" in readme

"""Install optional add-ons into the app's own environment, from inside the app.

The app ships lean: torch is ~1.1 GB, roughly two-thirds of the whole install,
and it is needed for exactly one of the 22 models. Downloading it for every
researcher — most of whom will never train a neural network on a few thousand
rows of nutrition data — makes the first launch several times slower on the
kind of connection people actually have at a conference.

But "some models don't work" is not an acceptable trade for an audience that
has never opened a terminal, and "run `uv pip install torch`" is exactly the
barrier the one-click launcher exists to remove. So the add-on is installed
the same way everything else in this app happens: one button, plain language,
visible progress, and an honest failure message.

The install targets the interpreter that is running the app (sys.executable),
which is the launcher's private .venv — never the system Python.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

# Roughly what the user is committing to, so the button can say so.
ADDONS = {
    "torch": {
        "package": "torch",
        "label": "Neural network model",
        "size_hint": "about 1.1 GB",
        "why": ("Adds the neural-network model. Every other model, and the rest "
                "of the workflow, already works without it."),
    },
}


@dataclass
class InstallResult:
    ok: bool
    message: str
    log: str = ""


def is_available(module: str) -> bool:
    """Is this optional module importable right now?"""
    import importlib.util
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _venv_root() -> Optional[Path]:
    """The environment the app is running in, if it is a virtualenv."""
    base = getattr(sys, "base_prefix", sys.prefix)
    return Path(sys.prefix) if sys.prefix != base else None


def _find_uv() -> Optional[str]:
    """uv, preferring the copy the launcher installed next to the app."""
    here = Path(__file__).resolve().parent.parent
    for candidate in (here / ".tools" / "uv", here / ".tools" / "uv.exe"):
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return shutil.which("uv")


def installer_command(package: str) -> List[str]:
    """The command that installs `package` into THIS interpreter's environment.

    uv is preferred because it is what the launcher already used and it is
    dramatically faster on a large wheel; pip is the universal fallback.
    """
    uv = _find_uv()
    if uv:
        return [uv, "pip", "install", "--python", sys.executable, package]
    return [sys.executable, "-m", "pip", "install", package]


def can_install() -> Tuple[bool, str]:
    """Whether an in-app install is possible, and why not if it isn't."""
    if _venv_root() is None:
        return False, (
            "The app is running on a system-wide Python rather than its own "
            "private environment, so it will not install packages for you — that "
            "could affect your other software."
        )
    if not os.access(sys.prefix, os.W_OK):
        return False, "The app's environment folder is read-only."
    return True, ""


def install(module: str, on_log: Optional[Callable[[str], None]] = None,
            timeout: int = 1800) -> InstallResult:
    """Install an optional add-on, streaming progress to `on_log`.

    Returns an InstallResult; never raises for ordinary failures, because this
    is called from a button and the user needs a sentence, not a traceback.
    """
    spec = ADDONS.get(module)
    if spec is None:
        return InstallResult(False, f"'{module}' is not an optional add-on of this app.")

    if is_available(module):
        return InstallResult(True, f"{spec['label']} is already installed.")

    ok, why = can_install()
    if not ok:
        return InstallResult(False, why)

    cmd = installer_command(spec["package"])
    lines: List[str] = []
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    except FileNotFoundError:
        return InstallResult(
            False,
            "Could not find an installer (uv or pip) in the app's environment. "
            "Reinstalling the app usually fixes this.",
        )
    except Exception as exc:                       # pragma: no cover - defensive
        return InstallResult(False, f"Could not start the installer: {exc}")

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                lines.append(line)
                if on_log:
                    on_log(line)
        code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        return InstallResult(
            False,
            "The download took too long and was stopped. This usually means a slow "
            "or interrupted connection — it is safe to try again.",
            "\n".join(lines[-40:]),
        )
    except Exception as exc:                       # pragma: no cover - defensive
        proc.kill()
        return InstallResult(False, f"The install stopped unexpectedly: {exc}",
                             "\n".join(lines[-40:]))

    tail = "\n".join(lines[-40:])
    if code != 0:
        hint = ""
        low = tail.lower()
        if "no space" in low or "disk" in low:
            hint = " Your disk appears to be full — free up about 2 GB and try again."
        elif any(w in low for w in ("network", "resolve", "connection", "timed out", "ssl")):
            hint = " This looks like a connection problem — check your internet and try again."
        return InstallResult(False, f"The install did not finish.{hint}", tail)

    # Newly written packages are not visible to an interpreter that already
    # cached a failed import, so invalidate before re-checking.
    import importlib
    importlib.invalidate_caches()
    if not is_available(module):
        return InstallResult(
            True,
            f"{spec['label']} was installed. Close the app window and start it "
            f"again to use it.",
            tail,
        )
    return InstallResult(True, f"{spec['label']} is ready to use.", tail)

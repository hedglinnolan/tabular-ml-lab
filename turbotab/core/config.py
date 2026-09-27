"""Runtime settings, read once from the environment (BLUEPRINT §2, §5).

    TURBOTAB_HOME            workspace root                 default ~/.turbotab
    TURBOTAB_MODE            "local" | "server"             default local
    TURBOTAB_WORKERS         job worker processes           default max(1, cpu_count - 1)
    TURBOTAB_MEMORY_BUDGET   bytes, or "8G" / "512M" / …    default 50% of currently available RAM
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Mode = Literal["local", "server"]
MODES: tuple[str, ...] = ("local", "server")

_UNITS = {"": 1, "B": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([KMGT]?)(?:I?B)?\s*$", re.IGNORECASE)


def parse_bytes(text: str) -> int:
    """``"8G"`` → 8 GiB, ``"512M"``, ``"1.5g"``, ``"2GB"``, ``"1048576"`` → bytes."""
    match = _SIZE_RE.match(str(text))
    if not match:
        raise ValueError(
            f"cannot read {text!r} as a memory size; write bytes or a number with "
            "K, M, G or T (for example 8G or 512M)")
    number, unit = match.groups()
    value = int(float(number) * _UNITS[unit.upper()])
    if value <= 0:
        raise ValueError(f"a memory size must be positive, not {text!r}")
    return value


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def total_memory_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return None


def available_memory_bytes() -> int:
    """RAM the OS could hand out right now, without swapping.

    psutil when installed; otherwise /proc/meminfo (Linux), ``vm_stat``
    (macOS) or GlobalMemoryStatusEx (Windows); total RAM as a last resort.
    """
    try:  # optional: not a declared dependency
        import psutil  # type: ignore[import-not-found]
        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo", encoding="ascii") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        elif sys.platform == "darwin":
            out = subprocess.run(["vm_stat"], capture_output=True, text=True,
                                 timeout=5, check=True).stdout
            page = int(re.search(r"page size of (\d+) bytes", out).group(1))
            pages = {m.group(1): int(m.group(2))
                     for m in re.finditer(r"^Pages ([a-z ]+):\s+(\d+)\.", out, re.M)}
            free = sum(pages.get(k, 0) for k in ("free", "inactive", "speculative"))
            if free > 0:
                return free * page
        elif sys.platform == "win32":
            import ctypes

            class _MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            status = _MemoryStatus()
            status.dwLength = ctypes.sizeof(_MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
    except Exception:
        pass
    return total_memory_bytes() or 8 * 1024**3


def default_memory_budget() -> int:
    return max(256 * 1024**2, available_memory_bytes() // 2)


@dataclass(frozen=True)
class Settings:
    home: Path
    mode: Mode = "local"
    workers: int = field(default_factory=default_workers)
    memory_budget_bytes: int = field(default_factory=default_memory_budget)

    def __post_init__(self) -> None:
        object.__setattr__(self, "home", Path(self.home).expanduser())
        if self.mode not in MODES:
            raise ValueError(f"mode must be 'local' or 'server', not {self.mode!r}")
        if int(self.workers) < 1:
            raise ValueError(f"workers must be at least 1, not {self.workers!r}")
        if int(self.memory_budget_bytes) <= 0:
            raise ValueError("memory_budget_bytes must be positive")

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "Settings":
        env = os.environ if environ is None else environ
        home = Path(env.get("TURBOTAB_HOME") or "~/.turbotab").expanduser().absolute()
        mode = (env.get("TURBOTAB_MODE") or "local").strip().lower()
        if mode not in MODES:
            raise ValueError(f"TURBOTAB_MODE must be 'local' or 'server', not {mode!r}")
        raw_workers = (env.get("TURBOTAB_WORKERS") or "").strip()
        try:
            workers = int(raw_workers) if raw_workers else default_workers()
        except ValueError:
            raise ValueError(
                f"TURBOTAB_WORKERS must be a whole number, not {raw_workers!r}") from None
        raw_budget = (env.get("TURBOTAB_MEMORY_BUDGET") or "").strip()
        try:
            budget = parse_bytes(raw_budget) if raw_budget else default_memory_budget()
        except ValueError as exc:
            raise ValueError(f"TURBOTAB_MEMORY_BUDGET: {exc}") from None
        return cls(home=home, mode=mode, workers=workers,  # type: ignore[arg-type]
                   memory_budget_bytes=budget)

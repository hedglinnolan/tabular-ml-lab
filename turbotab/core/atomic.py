"""Moving a finished file into place, on Windows too.

``os.replace`` is atomic on every platform, but Windows refuses it (PermissionError) while another
process holds the target open, where POSIX replaces the file under its reader. The server and its
job workers read the files the engine replaces (a stage's ``latest`` pointer, the project file) all
the time, so on Windows a replace is tried again for a short while (each reader holds the file only
to read it), and raises as before once the wait runs out. Elsewhere it is ``os.replace`` itself.
"""
from __future__ import annotations

import os
import time

WINDOWS = os.name == "nt"
WAIT_SECONDS = 2.0  # how long a refused replace on Windows is tried again before it raises


def replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
    """``os.replace(src, dst)``; on Windows, tried again while another process holds a file open."""
    if not WINDOWS:
        os.replace(src, dst)
        return
    deadline = time.monotonic() + WAIT_SECONDS
    pause = 0.005
    while True:
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(pause)
            pause = min(pause * 2, 0.1)

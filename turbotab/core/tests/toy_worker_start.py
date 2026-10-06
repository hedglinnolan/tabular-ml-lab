"""A job worker's preload that misbehaves on request, for the start tests in ``test_jobs.py``.

Importing it does nothing unless ``TURBOTAB_TOY_WORKER_START`` is set (a spawned worker inherits
the environment): ``hang`` makes the import never finish, ``fail`` makes it raise ``SystemExit``,
which a worker's preload does not swallow (only ``Exception``, for an optional package missing).
"""
from __future__ import annotations

import os
import time

_HOW = os.environ.get("TURBOTAB_TOY_WORKER_START", "")
if _HOW == "hang":
    time.sleep(600)
elif _HOW == "fail":
    raise SystemExit("toy: this worker cannot start")

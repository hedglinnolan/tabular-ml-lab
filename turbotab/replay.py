"""Replay a TurboTab export bundle: ``python -m turbotab.replay <bundle.zip> --data <input file>``.

The input files are checked against the SHA-256 the bundle's provenance record holds (a file that
differs is refused, both hashes named); the analysis is then rebuilt from the decision log alone in a
fresh TurboTab home, and the model matrix and every reported estimate are compared with the
bundle's. See ``turbotab/core/export/replay.py``. Exit status: 0 reproduced, 1 not reproduced,
2 refused.
"""
from __future__ import annotations

from turbotab.core.export.replay import main

if __name__ == "__main__":
    raise SystemExit(main())

"""The provenance record of the bundle (North star 4; BLUEPRINT §2: "an exported project carries
decisions + inputs, never fitted objects").

``provenance.json`` holds what a reader needs to reconstruct, and a replay needs to recompute, the
analysis: the decision log (every record, as stored), each input file's SHA-256 and size, the
analysis plan's hashes, the engine (TurboTab's version, its source revision when it runs from a
checkout, and every stage's version: the stage versions are the engine's computational identity),
the environment it ran in (Python and the scientific packages), the stage keys of the results it
reports, the model matrix's two hashes (``matrix.record``), and every number of the results tables
(``tables.estimates``). It holds no fitted object and no row of data.
"""
from __future__ import annotations

import hashlib
import platform
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

FORMAT = "turbotab-provenance/1"
REPO = Path(__file__).resolve().parents[3]
PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "pyarrow", "duckdb",
            "joblib", "lightgbm", "xgboost", "shap", "pydantic")
TOLERANCE = 1e-12
REPLAY = "python -m turbotab.replay <bundle.zip> --data <the input file>"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@lru_cache(maxsize=1)
def source_revision() -> dict[str, Any]:
    """The git revision TurboTab runs from, and whether tracked files differ from it; nulls when it
    does not run from a checkout (an installed copy names only its version)."""
    try:
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain",
                                "--untracked-files=no"], capture_output=True, text=True,
                               timeout=20, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "modified": None}
    return {"commit": head or None, "modified": bool(dirty) if head else None}


@lru_cache(maxsize=1)
def environment() -> dict[str, Any]:
    """Python and the scientific packages' versions, read from their metadata (nothing imported)."""
    from importlib import metadata

    packages: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "implementation": platform.python_implementation(),
            "system": platform.system(), "machine": platform.machine(), "packages": packages}


def provenance(source: Any, *, plan: Any, plan_bytes: bytes, matrix: Mapping[str, Any] | None,
               estimates: Mapping[str, float | None], stage_keys: Mapping[str, str | None]
               ) -> dict[str, Any]:
    """The provenance record (module docstring). ``plan``: ``plan_lock.plan_document``'s export of
    the same records, whose bytes are ``plan_bytes``."""
    state = source.state
    return {
        "format": FORMAT,
        "project": {"name": source.name, "purpose": getattr(state, "purpose", None),
                    "target": getattr(state, "target", None)},
        "inputs": [f.to_json() for f in source.inputs],
        "decisions": {"file": "decisions.jsonl", "sha256": sha256(source.decisions_jsonl),
                      "n": len(source.records),
                      "records": [r.model_dump(mode="json") for r in
                                  sorted(source.records, key=lambda r: r.seq)]},
        "analysis_plan": {"file": "analysis_plan.json", "file_sha256": sha256(plan_bytes),
                          "sha256": plan.sha256, "plan_sha256": plan.plan_sha256,
                          "status": plan.status, "declared_at": plan.declared_at},
        "engine": {"turbotab": source.engine_version, **source_revision(),
                   "stages": dict(sorted(source.stage_versions.items()))},
        "environment": environment(),
        "stages": dict(stage_keys),
        "model_matrix": dict(matrix) if matrix is not None else None,
        "estimates": dict(estimates),
        "replay": {"command": REPLAY, "tolerance": TOLERANCE},
    }


__all__ = ["FORMAT", "REPLAY", "TOLERANCE", "environment", "provenance", "sha256",
           "source_revision"]

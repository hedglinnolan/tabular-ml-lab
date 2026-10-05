"""R as an independent reference inside the acceptance tests, never inside the app.

A test writes its data to CSV (every float at full precision), runs an R script on it with
``Rscript``, and reads back what the script prints as JSON (``jsonlite::toJSON(…, digits = NA,
auto_unbox = TRUE)``). The packages are those ``turbotab/server/requirements-dev-R.txt`` lists. A
machine without R skips these tests (:data:`needs_r`); nothing in ``turbotab/`` outside the tests
imports or calls R.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import pytest

RSCRIPT = shutil.which("Rscript")
needs_r = pytest.mark.skipif(RSCRIPT is None, reason="R (Rscript) is not installed")


def run_r(script: str, frames: Mapping[str, pd.DataFrame], folder: Path, *,
          timeout: float = 300.0) -> dict[str, Any]:
    """Run ``script`` with each frame written to ``folder/<name>.csv`` and its path bound to the R
    variable ``<name>_csv``; the script prints one JSON object (``out(list(...))`` does it), which
    is returned."""
    folder.mkdir(parents=True, exist_ok=True)
    lines = ["suppressPackageStartupMessages(library(jsonlite))",
             "out <- function(x) cat(toJSON(x, digits = NA, auto_unbox = TRUE, na = 'null'))"]
    for name, frame in frames.items():
        path = folder / f"{name}.csv"
        frame.to_csv(path, index=False, float_format="%.17g")
        lines.append(f'{name}_csv <- "{path.as_posix()}"')
    source = folder / "reference.R"
    source.write_text("\n".join(lines) + "\n" + script, encoding="utf-8")
    assert RSCRIPT is not None
    done = subprocess.run([RSCRIPT, "--vanilla", str(source)], capture_output=True, text=True,
                          timeout=timeout, env=_env())
    if done.returncode != 0:
        raise RuntimeError(f"R failed ({done.returncode}):\n{done.stderr[-3000:]}")
    text = done.stdout.strip().splitlines()
    if not text:
        raise RuntimeError(f"R printed nothing:\n{done.stderr[-3000:]}")
    return json.loads(text[-1])


def _env() -> dict[str, str]:
    import os

    return dict(os.environ)


__all__ = ["RSCRIPT", "needs_r", "run_r"]

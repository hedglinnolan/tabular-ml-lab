"""Independent references for the MS7 acceptance tests (QC drift, QC filters, PQN, values below
detection, batch, multiplicity): R run as a subprocess on CSV files, pandas computations written
from the published definitions, and simulated runs whose truth is known.

R is a reference only: nothing here is imported by the app, and every test that calls R is marked
:data:`needs_r`, so a machine without R skips it.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import pytest

RSCRIPT = shutil.which("Rscript")
needs_r = pytest.mark.skipif(shutil.which("Rscript") is None, reason="R is not installed")


def run_r(code: str, inputs: Mapping[str, pd.DataFrame], folder: Path,
          outputs: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """Write ``inputs`` as ``<name>.csv`` in ``folder``, run ``code`` with the working directory
    there, and read back each ``<output>.csv`` the code wrote."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, frame in inputs.items():
        frame.to_csv(folder / f"{name}.csv", index=False, na_rep="NA")  # a blank line would be skipped
    script = folder / "reference.R"
    script.write_text(code, encoding="utf-8")
    done = subprocess.run([RSCRIPT, "--vanilla", str(script)], cwd=folder, capture_output=True,
                          text=True, timeout=600)
    if done.returncode != 0:
        raise RuntimeError(f"R failed:\n{done.stderr[-4000:]}")
    return {name: pd.read_csv(folder / f"{name}.csv") for name in outputs}


# ── a drifting LC-MS run with pooled QCs, its truth known ─────────────────────


def drifting_run(seed: int = 0, p: int = 200, batches: int = 3, per_batch: int = 40,
                 qc_every: int = 5, noise: float = 0.05, bio_sd: float = 0.4,
                 amplitude: tuple[float, float] = (0.1, 0.4), batch_sd: float = 0.3
                 ) -> tuple[pd.DataFrame, np.ndarray]:
    """``(frame, truth)``: ``batches`` batches of ``per_batch`` injections, a pooled QC every
    ``qc_every``-th injection and at each batch's last; each feature's intensity is its level ×
    the sample's biology (1 for the pooled QCs) × the batch's offset × a drift that falls by up
    to ``amplitude`` over the batch (a power curve of injection position) × multiplicative noise
    of SD ``noise`` on the log scale. ``truth`` is level × biology, what a perfect correction
    recovers up to one constant per feature."""
    rng = np.random.default_rng(seed)
    rows, order = [], 0
    for b in range(batches):
        for k in range(per_batch):
            order += 1
            qc = (k % qc_every == 0) or k == per_batch - 1
            rows.append({"injection_order": order, "batch": f"B{b + 1}",
                         "sample_type": "QC" if qc else "Sample", "_k": k})
    meta = pd.DataFrame(rows)
    n = len(meta)
    level = np.exp(rng.normal(8, 1.5, p))
    biology = np.exp(rng.normal(0, bio_sd, (n, p)))
    biology[meta["sample_type"].eq("QC").to_numpy()] = 1.0
    position = meta["_k"].to_numpy() / (per_batch - 1)
    amp = rng.uniform(*amplitude, p)
    shape = rng.uniform(0.5, 2.0, p)
    offset = {f"B{b + 1}": np.exp(rng.normal(0, batch_sd, p)) for b in range(batches)}
    drift = np.vstack([offset[bb] * (1 - amp * pos ** shape)
                       for bb, pos in zip(meta["batch"], position)])
    truth = level * biology
    observed = truth * drift * np.exp(rng.normal(0, noise, (n, p)))
    frame = pd.DataFrame(observed, columns=[f"m{j:03d}" for j in range(p)])
    frame.insert(0, "sample_type", meta["sample_type"])
    frame.insert(0, "batch", meta["batch"])
    frame.insert(0, "injection_order", meta["injection_order"])
    return frame, truth


def uncorrectable_by_hand(frame: pd.DataFrame, features: list[str], qc_column: str = "sample_type",
                          qc_level: str = "QC", order: str = "injection_order",
                          batch: str | None = "batch", min_qc: int = 5) -> set[str]:
    """The features QC-RLSC cannot correct without guessing its curve, by the rule's definition
    (Dunn et al. 2011; Broadhurst et al. 2018), written with pandas: in some batch, fewer than
    ``min_qc`` QC injections detected the feature (a positive value), or a detected value of a
    study injection lies before the first or after the last QC injection that detected it (the
    curve, fit to the detected QCs only, would be extrapolated there)."""
    out: set[str] = set()
    groups = [frame] if batch is None else [g for _, g in frame.groupby(batch, sort=True)]
    for g in groups:
        is_qc = g[qc_column].astype(str).eq(qc_level)
        for c in features:
            seen = g[c].gt(0) & g[c].notna()
            qc_orders = g.loc[is_qc & seen, order]
            if len(qc_orders) < min_qc:
                out.add(c)
                continue
            study = g.loc[~is_qc & seen, order]
            if ((study < qc_orders.min()) | (study > qc_orders.max())).any():
                out.add(c)
    return out


# ── PQN by its published steps ───────────────────────────────────────────────


def pqn_by_hand(reference_rows: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Probabilistic quotient normalization (Dieterle et al. 2006), in Kohl et al.'s words:
    "PQN starts, with an integral normalization of each spectrum, followed by the calculation of a
    reference spectrum such as a median spectrum. Next, for each variable of interest the quotient
    of a given test spectrum and reference spectrum is calculated and the median of all quotients
    is estimated. Finally, all variables of the test spectrum are divided by the median quotient."
    The reference is learned from ``reference_rows`` alone (the pooled QCs, or a training fold);
    positive values only. Written with pandas, not the app's code."""
    ref = reference_rows.where(reference_rows > 0)
    totals = ref.sum(axis=1)
    constant = float(totals.median())
    spectrum = ref.mul(constant / totals, axis=0).median(axis=0)
    test = rows.where(rows > 0)
    quotient = test.div(spectrum, axis=1).median(axis=1)
    return rows.div(quotient, axis=0)


__all__ = ["RSCRIPT", "drifting_run", "needs_r", "pqn_by_hand", "run_r", "uncorrectable_by_hand"]

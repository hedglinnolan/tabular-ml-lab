"""The views lab's fixtures that no engine artifact supplies yet (embedding, matrix), computed from
the repository's sample data so every drawn number is real. Run from the repository root:

    .venv/bin/python turbotab/frontend/src/components/views/fixtures/make_fixtures.py

The overlap view needs no file here: its fixture is the engine's own OverlapView, captured in
src/mocks/fixtures/m3-causal.json.

Each output carries `source`, saying how it was computed, because it stands in for an engine
artifact (the engine contract items in the doc comments of each view's types.ts).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[6]
DATA = ROOT / "turbotab" / "sample_data"
OUT = Path(__file__).resolve().parent


def r6(v: float | None) -> float | None:
    if v is None or not np.isfinite(v):
        return None
    return float(f"{v:.6g}")


def embedding_pca() -> dict:
    """PCA of the untargeted metabolomics run: log2 intensities, each blank set to half its
    feature's smallest value, each feature centered and scaled to unit variance."""
    d = pd.read_csv(DATA / "metabolomics_untargeted.csv")
    feats = [c for c in d.columns if c.startswith("mz_")]
    x = d[feats].apply(pd.to_numeric, errors="coerce")
    x = x.where(x > 0)
    x = x.fillna(x.min() / 2)
    x = np.log2(x)
    x = (x - x.mean()) / x.std(ddof=1)
    x = x.loc[:, x.std(ddof=1) > 0]
    u, s, _vt = np.linalg.svd(x.to_numpy(), full_matrices=False)
    scores = u[:, :2] * s[:2]
    share = (s**2) / (s**2).sum()
    levels = ["B1", "B2"]
    return {
        "method": "pca",
        "axes": [
            {"label": "Component 1", "share": r6(share[0])},
            {"label": "Component 2", "share": r6(share[1])},
        ],
        "xs": [r6(v) for v in scores[:, 0]],
        "ys": [r6(v) for v in scores[:, 1]],
        "ids": d["sample_id"].astype(str).tolist(),
        "groups": [levels.index(b) for b in d["batch"]],
        "grouping": {"name": "batch", "levels": levels},
        "basis": f"{x.shape[1]} features, log-scaled and standardized, on {len(d)} samples",
        "source": "metabolomics_untargeted.csv via make_fixtures.py (no engine artifact yet)",
    }


CORR_COLUMNS = ["age", "bmi", "energy_kcal", "protein_g", "fat_g", "carbohydrate_g", "fiber_g",
                "sodium_mg", "protein_pct_kcal", "fat_pct_kcal", "carbohydrate_pct_kcal",
                "alcohol_pct_kcal"]


def matrix_correlation() -> dict:
    """Pairwise-complete Pearson correlations among the dietary recalls' columns (hba1c, the
    outcome, is left out: the outcome beside a column waits for the lock)."""
    d = pd.read_csv(DATA / "dietary_recalls.csv")
    v = d[CORR_COLUMNS].apply(pd.to_numeric, errors="coerce")
    c = v.corr()
    ok = v.notna().astype(int)
    n = ok.T @ ok
    return {
        "kind": "correlation",
        "method": "Pearson",
        "rows": CORR_COLUMNS,
        "cols": CORR_COLUMNS,
        "order": "as in your file",
        "symmetric": True,
        "values": [[r6(c.loc[a, b]) for b in CORR_COLUMNS] for a in CORR_COLUMNS],
        "n": [[int(n.loc[a, b]) for b in CORR_COLUMNS] for a in CORR_COLUMNS],
        "source": "dietary_recalls.csv via make_fixtures.py (no engine artifact yet)",
    }


def matrix_missingness() -> dict:
    """The share of blanks in the twelve features blank most often, in each batch and sample
    type of the untargeted metabolomics run."""
    d = pd.read_csv(DATA / "metabolomics_untargeted.csv")
    feats = [c for c in d.columns if c.startswith("mz_")]
    blank = d[feats].apply(pd.to_numeric, errors="coerce").isna()
    top = blank.mean().sort_values(ascending=False, kind="stable").index[:12]
    cols = [f for f in feats if f in set(top)]  # the declared order: the file's
    names = {"participant": "participants", "pooled_qc": "pooled QCs"}
    rows, values, ns = [], [], []
    for (batch, kind), g in d.groupby(["batch", "sample_type"], sort=True):
        rows.append(f"{batch} · {names.get(kind, kind)}")
        values.append([r6(blank.loc[g.index, f].mean()) for f in cols])
        ns.append([int(len(g))] * len(cols))
    return {
        "kind": "missingness",
        "rows": rows,
        "cols": cols,
        "order": "columns as in your file; groups by batch",
        "symmetric": False,
        "values": values,
        "n": ns,
        "source": "metabolomics_untargeted.csv via make_fixtures.py (no engine artifact yet)",
    }


def main() -> None:
    for name, fn in [("embedding-pca", embedding_pca), ("matrix-correlation", matrix_correlation),
                     ("matrix-missingness", matrix_missingness)]:
        (OUT / f"{name}.json").write_text(json.dumps(fn(), indent=1) + "\n")
        print("wrote", name)


if __name__ == "__main__":
    main()

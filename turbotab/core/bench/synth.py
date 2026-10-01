"""The two synthetic benchmark tables (M2_CONTRACT §5), written as CSV.

* ``wide``: 500 samples × 20,000 gene counts (negative binomial, a library-size factor per
  sample), plus ``sample_id``, ``age``, ``sex``, ``batch`` and a continuous outcome ``bmi`` that
  depends on 20 of the genes' log counts. 20,005 columns.
* ``tall``: 1,000,000 rows × 30 columns of a dietary-style table: ``participant_id``, ``age``,
  ``sex``, ``race``, nutrients in grams, ``kcal``, a few covariates, and an outcome ``glucose``.

Seeded, so a rerun writes the same bytes::

    venv/bin/python -m turbotab.core.bench.synth --out /path/to/dir [--which wide|tall|both]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

WIDE_ROWS, WIDE_GENES = 500, 20_000
TALL_ROWS = 1_000_000
WIDE_NAME = "wide_500x20000.csv"
TALL_NAME = "tall_1000000x30.csv"


def wide_table(n: int = WIDE_ROWS, genes: int = WIDE_GENES, seed: int = 2026) -> Any:
    import pyarrow as pa

    rng = np.random.default_rng(seed)
    library = rng.lognormal(0.0, 0.35, size=n)          # per-sample depth
    base = rng.lognormal(2.0, 1.6, size=genes)          # per-gene mean expression
    shape = 1.0 / 0.4                                   # negative binomial, dispersion 0.4
    counts = rng.poisson(rng.gamma(shape, np.outer(library, base) / shape)).astype(np.int32)
    signal = rng.choice(genes, size=20, replace=False)
    effect = rng.normal(0.0, 0.6, size=20)
    logc = np.log1p(counts[:, signal])
    logc = (logc - logc.mean(axis=0)) / (logc.std(axis=0) + 1e-9)
    bmi = 27.0 + logc @ effect + rng.normal(0.0, 1.5, size=n)
    columns: dict[str, Any] = {
        "sample_id": [f"S{i:05d}" for i in range(n)],
        "age": rng.integers(20, 80, size=n).astype(np.int64),
        "sex": rng.choice(["F", "M"], size=n).tolist(),
        "batch": rng.choice(["B1", "B2", "B3"], size=n).tolist(),
    }
    width = len(str(genes))
    for j in range(genes):
        columns[f"gene_{j + 1:0{width}d}"] = counts[:, j]
    columns["bmi"] = np.round(bmi, 2)
    return pa.table(columns)


def tall_table(n: int = TALL_ROWS, seed: int = 2027) -> Any:
    import pyarrow as pa

    rng = np.random.default_rng(seed)
    kcal = np.clip(rng.normal(2100, 650, size=n), 300, 7000)
    share = rng.dirichlet([5, 9, 6, 0.4], size=n)       # protein, carbohydrate, fat, alcohol
    carb = share[:, 1] * kcal / 4
    fat = share[:, 2] * kcal / 9
    sugar = carb * rng.uniform(0.2, 0.6, size=n)
    bmi = np.clip(rng.normal(28, 6, size=n), 14, 70)
    glucose = 85 + (bmi - 28) + 0.004 * sugar + rng.normal(0, 12, size=n)

    def clipped(mean: float, sd: float, low: float, digits: int) -> Any:
        return np.round(np.clip(rng.normal(mean, sd, size=n), low, None), digits)

    cols: dict[str, Any] = {
        "participant_id": np.arange(1, n + 1, dtype=np.int64),
        "age": rng.integers(18, 85, size=n).astype(np.int64),
        "sex": rng.choice(np.array(["female", "male"]), size=n),
        "race": rng.choice(np.array(["A", "B", "C", "D", "E"]), size=n,
                           p=[0.4, 0.2, 0.2, 0.1, 0.1]),
        "income_ratio": np.round(np.clip(rng.normal(2.5, 1.5, size=n), 0, 5), 2),
        "education": rng.integers(1, 6, size=n).astype(np.int64),
        "smoker": rng.choice(np.array(["yes", "no"]), size=n, p=[0.2, 0.8]),
        "activity_met": np.round(rng.gamma(2.0, 10.0, size=n), 1),
        "bmi": np.round(bmi, 1),
        "waist_cm": np.round(np.clip(bmi * 3.3 + rng.normal(0, 6, size=n), 50, 180), 1),
        "kcal": np.round(kcal, 0),
        "protein_g": np.round(share[:, 0] * kcal / 4, 1),
        "carb_g": np.round(carb, 1),
        "fat_g": np.round(fat, 1),
        "alcohol_g": np.round(share[:, 3] * kcal / 7, 1),
        "sugar_g": np.round(sugar, 1),
        "fiber_g": clipped(17, 7, 0, 1),
        "sat_fat_g": np.round(fat * rng.uniform(0.25, 0.45, size=n), 1),
        "sodium_mg": clipped(3400, 1100, 200, 0),
        "potassium_mg": clipped(2600, 800, 100, 0),
        "calcium_mg": clipped(950, 350, 50, 0),
        "iron_mg": clipped(14, 5, 1, 1),
        "vitamin_c_mg": np.round(rng.gamma(2.0, 40.0, size=n), 1),
        "caffeine_mg": np.round(rng.gamma(1.2, 120.0, size=n), 0),
        "water_g": clipped(2800, 900, 200, 0),
        "sbp": np.round(np.clip(rng.normal(122, 17, size=n), 80, 220), 0),
        "dbp": np.round(np.clip(rng.normal(74, 11, size=n), 40, 130), 0),
        "meds_hbp": rng.choice(np.array([1.0, 2.0, np.nan]), size=n, p=[0.25, 0.35, 0.4]),
        "recall_day": rng.integers(1, 3, size=n).astype(np.int64),
        "glucose": np.round(glucose, 1),
    }
    return pa.table(cols)


def write(table: Any, path: Path) -> Path:
    import pyarrow.csv as pacsv

    path.parent.mkdir(parents=True, exist_ok=True)
    pacsv.write_csv(table, path, write_options=pacsv.WriteOptions(quoting_style="needed"))
    return path


def ensure(folder: Path, which: str) -> Path:
    """The table's CSV in ``folder``, written first when it is not there yet."""
    name, make = (WIDE_NAME, wide_table) if which == "wide" else (TALL_NAME, tall_table)
    path = Path(folder) / name
    if not path.is_file():
        tmp = path.with_name(f".{name}.tmp")
        write(make(), tmp)
        tmp.replace(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the synthetic benchmark tables.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--which", choices=("wide", "tall", "both"), default="both")
    args = parser.parse_args()
    for which in ("wide", "tall"):
        if args.which in (which, "both"):
            t = time.perf_counter()
            p = ensure(args.out, which)
            print(f"{p}  {p.stat().st_size / 1e6:.1f} MB  {time.perf_counter() - t:.1f} s")


if __name__ == "__main__":
    main()

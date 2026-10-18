"""A check of each run's analytic-rows CSV against the engine's own estimate.

From the CSV alone (the engine's model matrix and its design columns, ``run.py``'s
``save_rows``): survey-weighted least squares on the rows with a positive weight, and the
Taylor-linearized SE written from the textbook sandwich (Binder 1983): the bread
``(XᵀWX)⁻¹``, the meat ``Σ_h n_h/(n_h − 1) Σ_j (z_hj − z̄_h)(z_hj − z̄_h)ᵀ`` over each stratum's
PSU totals of the weighted scores, no lonely-PSU rule (no stratum here has one PSU). numpy only;
no TurboTab code. Agreement says the CSV holds the rows and columns the engine fit; it is not the
independent reference (SU-R's R tables are).

    scripts/py research/replication/sullivan2021/engine/check_rows.py [names]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ENGINE = REPO / "research" / "data" / "derived" / "engine"
CYCLES = 4  # the weights' division (the plan file's survey answer names the cycle column)


def refit(name: str) -> dict[str, Any]:
    run = json.loads((ENGINE / "runs" / f"{name}.json").read_text())
    s = run["run"]["settings"]
    frame = pd.read_csv(ENGINE / f"{name}.csv")
    divide = CYCLES if s["survey"].get("cycle") else 1
    w = frame[s["survey"]["weight"]].to_numpy(float) / divide
    frame, w = frame[w > 0], w[w > 0]
    xcols = [c for c in frame.columns if c.startswith("x:")]
    X = np.column_stack([np.ones(len(frame)), frame[xcols].to_numpy(float)])
    y = frame[(run.get("rows") or {}).get("outcome") or s["target"]].to_numpy(float)
    bread = np.linalg.inv(X.T @ (X * w[:, None]))
    beta = bread @ (X.T @ (w * y))
    u = X * (w * (y - X @ beta))[:, None]
    strata, psu = frame[s["survey"]["strata"]].to_numpy(), frame[s["survey"]["psu"]].to_numpy()
    meat = np.zeros((X.shape[1], X.shape[1]))
    for h in np.unique(strata):
        at = strata == h
        totals = np.array([u[at & (psu == j)].sum(axis=0) for j in np.unique(psu[at])])
        if len(totals) > 1:
            dev = totals - totals.mean(axis=0)
            meat += len(totals) / (len(totals) - 1) * dev.T @ dev
    cov = bread @ meat @ bread
    k = 1 + xcols.index("x:DBQ700")
    row = next(r for r in run["table2"] if r["primary"])
    b, se = float(beta[k]), float(np.sqrt(cov[k, k]))
    return {"beta": b, "se": se, "n": int(len(frame)),
            "beta_rel_diff": abs(b / row["estimate"] - 1), "se_rel_diff": abs(se / row["se"] - 1),
            "n_matches": int(len(frame)) == int(row["n"])}


def main(argv: list[str]) -> int:
    names = argv or sorted(p.stem for p in (ENGINE / "runs").glob("*.json"))
    worst = 0.0
    for name in names:
        r = refit(name)
        worst = max(worst, r["beta_rel_diff"], r["se_rel_diff"])
        print(f"{name:28} beta {r['beta']:+.8f} se {r['se']:.8f} n {r['n']} rel "
              f"{r['beta_rel_diff']:.1e} {r['se_rel_diff']:.1e}")
    print(f"largest relative difference: {worst:.2e}")
    return 0 if worst < 1e-6 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

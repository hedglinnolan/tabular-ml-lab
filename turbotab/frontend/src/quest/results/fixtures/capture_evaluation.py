"""Captures the evaluation stage's exhibits for Predict Results (G09 of the front-end audit): what the
real stage graph serves under Predict for a yes/no outcome with decision support, named subgroups,
a site grouping and shrinkage as model updating.

The table is constructed here (a seeded draw, :func:`_table`): a yes/no outcome from three numeric
predictors and sex, an age, a smoking column with a level too rare to score, a count of visits
whose code-or-amount reading no answer settles, and four sites. The real stage graph runs it in
process the way the workers do (``turbotab/core/tests/graph_runner.GraphRun``, as
``test_explore_chain.py``'s chain A), up to the evaluation stage: the linear model and ridge, 20% held
out, five folds, decision support with ``sex``, ``age``, ``smoker`` and ``visits`` as subgroups, the
sites as the clusters, shrinkage answered as model updating.

    OMP_NUM_THREADS=2 TURBOTAB_WORKERS=2 scripts/py turbotab/frontend/src/quest/results/fixtures/capture_evaluation.py

writes ``evaluation-binary.json`` beside this file: the fit and evaluation stages' artifacts exactly
as the stages wrote them (the fit sealed, as the server serves it before the opening: no held-out
score is in the stage's own artifact). Nothing in it is written by hand.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))  # the repository's root: `turbotab`

from turbotab.core import decisions as d  # noqa: E402
from turbotab.core.tests.graph_runner import GraphRun  # noqa: E402

OUT = Path(__file__).with_name("evaluation-binary.json")
SEED = 41
SUBGROUPS = ["sex", "age", "smoker", "visits"]


def _table(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(600):
        x1, x2, x3 = rng.normal(size=3)
        sex = int(rng.integers(1, 3))
        age = float(rng.uniform(20, 80))
        lp = -1.0 + 0.9 * x1 - 0.6 * x2 + 0.3 * x3 + 0.3 * (sex == 2) + 0.02 * (age - 50)
        rows.append({"pid": i, "x1": x1, "x2": x2, "x3": x3, "sex": sex, "age": age,
                     # a level too rare to score: a group the engine names, not scores
                     "smoker": "unknown" if i % 97 == 0 else ("yes" if rng.random() < 0.3 else "no"),
                     # whole numbers with many values: codes or a count, which no answer settles
                     "visits": int(rng.integers(0, 13)),
                     "site": f"site_{int(rng.integers(0, 4))}",
                     "y": "yes" if rng.random() < 1 / (1 + np.exp(-lp)) else "no"})
    return pd.DataFrame(rows)


def main() -> int:
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        frame = _table()
        frame.to_csv(folder / "t.csv", index=False)
        run = GraphRun(folder / "t.csv", folder / "p")
        roles = {"pid": "identifier", "x1": "covariate", "x2": "covariate", "x3": "covariate",
                 "sex": "covariate", "age": "covariate", "smoker": "excluded",
                 "visits": "excluded", "site": "design"}
        state = d.ProjectState(
            lens=["clinical"], target="y", task="binary", purpose="prediction", roles=roles,
            event="yes", missing="complete_case", categorical=["sex"],
            grain=d.GrainSpec(grain="one_row_per_unit", id_column="pid"),
            split=d.SplitSpec(holdout=0.2, seed=6, folds=5), models=["linear", "ridge"],
            clusters=d.ClusterSpec(column="site"),
            intended_use=d.IntendedUseSpec(use="decision_support", subgroups=SUBGROUPS),
            updating=d.UpdatingSpec(method="shrinkage"))
        try:
            out = run.public(run.run(state, upto=["evaluation"]))
        finally:
            run.close()
    payload = {
        "meta": {"script": "turbotab/frontend/src/quest/results/fixtures/capture_evaluation.py",
                 "captured": date.today().isoformat(), "seed": SEED, "subgroups": SUBGROUPS,
                 "seconds": round(time.monotonic() - t0)},
        "fit": out["fit"], "evaluation": out["evaluation"]}
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {OUT.name} in {payload['meta']['seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

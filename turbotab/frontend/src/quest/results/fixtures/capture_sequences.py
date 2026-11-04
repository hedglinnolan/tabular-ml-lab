"""Captures the effects artifacts of declared model sequences that Estimate Results' Table 2 draws
only in part: what the real effects stage serves where a declared model reports no estimate, or
reports the exposure as several rows, each state built by an engine test's own construction and
run through the real design and effects stages (``estimand_fixtures.run``) on every row:

* ``dependency``: `fiber2` = 2·`fiber` a covariate, so Model 2, the primary, is refused whole in
  the dependency's words and the unadjusted model alone reports `fiber`
  (``test_exposure_in_a_dependency_through_the_stages``, the linear model);
* ``sole``: `fib` = 2·`activity` + 1, the one study factor, estimable only without the covariates,
  so Model 2 and Model 3 are refused as not estimable given them and the unadjusted model reports
  it (``test_featurewise_family_counts_its_tests``, feature-wise regression);
* ``blocked``: under multiple imputation a covariate `q` recorded on one row, which the sampler
  cannot impute, so the family is blocked and recorded: Model 2 alone, no coefficients, its
  refusal the imputation's (``test_imputation_width``'s words, ``EffectsRun._blocked``);
* ``levels``: the category `grp` (levels a, b and c; ``test_featurewise_family_counts_its_tests``'
  frame, whose outcome is raised by 3 where `grp` is c) as what is studied, so every model reports
  it as two rows, b and c against a.

Each refusal's words are the engine test's, written out from the construction, and each row the
unadjusted model reports is checked against statsmodels' glucose ~ exposure (a category as its
indicators, the first level the reference) under the covariance its table names, whose estimate,
p-value and 95% interval are kept beside it as ``truth``.

    OMP_NUM_THREADS=2 TURBOTAB_WORKERS=2 scripts/py turbotab/frontend/src/quest/results/fixtures/capture_sequences.py

writes ``estimate-sequences.json`` beside this file; nothing in it is written by hand.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))  # the repository's root: `turbotab`

import turbotab.core.models  # noqa: E402,F401 - registers the families
from turbotab.core.decisions import MissingSpec  # noqa: E402
from turbotab.core.tests import test_exposure_in_a_dependency_through_the_stages as DEP  # noqa: E402
from turbotab.core.tests import test_featurewise_family_counts_its_tests as FW  # noqa: E402
from turbotab.core.tests import test_imputation_width as MI  # noqa: E402
from turbotab.core.tests.acceptance import estimand_fixtures as ef  # noqa: E402

OUT = Path(__file__).with_name("estimate-sequences.json")
N = 400
BLOCKED = MI.LEAD + MI.ONCE  # "Multiple imputation cannot run on these data: `q` has fewer …"
# The covariance each served table names, as statsmodels fits it: HC3 with the t(n − p) reference
# (core/models/inference.py), or the model's own (the classical t-test).
COVARIANCE = {"HC3": {"cov_type": "HC3", "use_t": True}, "model": {}}


def _crude(frame: pd.DataFrame, effects: dict[str, Any], X: pd.DataFrame) -> dict[str, Any]:
    """The unadjusted model's served rows, each checked against statsmodels' glucose ~ ``X``
    under the covariance its table names; the truth returned, by row."""
    crude = ef.sequence(effects)["crude"]
    covariance = crude["inference"]["covariance"]
    fit = sm.OLS(frame["glucose"].to_numpy(float), sm.add_constant(X.astype(float))).fit(
        **COVARIANCE[covariance])
    limits = fit.conf_int(0.05)
    truth: dict[str, Any] = {}
    assert [r["feature"] for r in crude["effects"]] == list(X.columns), crude["effects"]
    for row in crude["effects"]:
        c = row["feature"]
        want = {"estimate": float(fit.params[c]), "p": float(fit.pvalues[c]),
                "ci_low": float(limits.loc[c, 0]), "ci_high": float(limits.loc[c, 1])}
        for k in ("estimate", "ci_low", "ci_high"):
            assert abs(row[k] - want[k]) <= 1e-9 * abs(want[k]), (c, k, row[k], want[k])
        assert abs(row["p"] - want["p"]) <= 1e-6 * want["p"], (c, row["p"], want["p"])
        truth[c] = want
    return {"covariance": covariance, "rows": truth}


def dependency(folder: Path) -> dict[str, Any]:
    frame = ef.cohort(n=N, seed=7).drop(columns=["bmi"])
    frame["fiber2"] = 2 * frame["fiber"]
    roles = {k: v for k, v in ef.ROLES.items() if k != "bmi"} | {"fiber2": "covariate"}
    answers = {k: v for k, v in ef.ANSWERS.items() if k != "bmi"} | {"fiber2": ef.CONFOUNDER}
    st = ef.state(target="glucose", task="regression", measure="mean_difference", roles=roles,
                  answers=answers, plan_locked=True)
    effects = ef.run(frame, folder, st, fit=False)["effects"]
    seq = ef.sequence(effects)
    assert seq["model_2"]["effects"] is None and seq["model_2"]["inference"]["refused"] == DEP.WORDS
    return {"artifact": effects, "truth": {"crude": _crude(frame, effects, frame[["fiber"]])},
            "refused": {"model_2": DEP.WORDS}}


def sole(folder: Path) -> dict[str, Any]:
    frame = FW._fib_frame()
    st = FW._state({"fib": "exposure"}, ["featurewise"], exposure="fib", amounts=FW.FIB_AMOUNTS)
    effects = ef.run(frame, folder, st, fit=False)["effects"]
    seq = ef.sequence(effects)
    assert seq["model_2"]["effects"] is None and seq["model_2"]["inference"]["refused"] == FW.FIB
    assert seq["model_3"]["effects"] is None
    assert seq["model_3"]["inference"]["refused"] == FW.FIB_THREE
    return {"artifact": effects, "truth": {"crude": _crude(frame, effects, frame[["fib"]])},
            "refused": {"model_2": FW.FIB, "model_3": FW.FIB_THREE}}


def blocked(folder: Path) -> dict[str, Any]:
    frame = ef.cohort(n=N, seed=7)
    q = np.full(N, np.nan)
    q[0] = 1.5
    frame["q"] = q
    roles = {**ef.ROLES, "q": "covariate"}
    answers = {**ef.ANSWERS, "q": ef.CONFOUNDER}
    st = ef.state(target="glucose", task="regression", measure="mean_difference", roles=roles,
                  answers=answers, missing=MissingSpec(strategy="multiple_imputation"),
                  plan_locked=True)
    effects = ef.run(frame, folder, st, fit=False)["effects"]
    [family] = effects["families"]
    [fit] = family["sequence"]
    assert fit["key"] == "model_2" and fit["effects"] is None, fit
    assert fit["inference"]["refused"] == BLOCKED, fit["inference"]["refused"]
    assert fit["inference"]["caption"] == f"No coefficients: {BLOCKED}"
    return {"artifact": effects, "refused": {"model_2": BLOCKED}}


def levels(folder: Path) -> dict[str, Any]:
    frame = FW._frame()
    st = FW._state({"grp": "exposure", "fiber": "covariate"}, ["linear"], exposure="grp")
    effects = ef.run(frame, folder, st, fit=False)["effects"]
    indicators = pd.DataFrame({f"grp_{v}": (frame["grp"] == v).astype(float) for v in ("b", "c")})
    for fit in effects["families"][0]["sequence"]:
        assert [r["feature"] for r in fit["effects"]] == list(indicators.columns), fit["effects"]
    return {"artifact": effects, "truth": {"crude": _crude(frame, effects, indicators)}}


def main() -> int:
    t0 = time.monotonic()
    out: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="estimate-sequences-") as folder:
        for name, build in (("dependency", dependency), ("sole", sole), ("blocked", blocked),
                            ("levels", levels)):
            out[name] = build(Path(folder) / name)
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                          text=True, cwd=Path(__file__).parent).stdout.strip()
    out = {"meta": {"journey": "estimate-sequences", "captured": date.today().isoformat(),
                    "head": head, "seconds": round(time.monotonic() - t0),
                    "script": "turbotab/frontend/src/quest/results/fixtures/capture_sequences.py"},
           **out}
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes) in {out['meta']['seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

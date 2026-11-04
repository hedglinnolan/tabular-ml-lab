"""Captures the effects artifacts of declared model sequences that serve what is studied as one row
under a name that is not its column's, or as an exposure family with a member Table 2 does not
draw: each state built from ``estimand_fixtures``' cohort (or the feature-wise tests' frame, with
its category `grp` and its two-level `arm`) and run through the real design and effects stages
(``estimand_fixtures.run``) on every row:

* ``density``: `fiber` energy-adjusted by the density method on `kcal`, so every model serves the
  one row `fiber_per_kcal` = `fiber` / `kcal`, a coefficient per unit of the density, not of
  `fiber`;
* ``cut``: `fiber` in the categories form at the one declared cut point 20, so every model serves
  the one indicator `fiber_C2` (`fiber` > 20, the lowest group holding its cut point);
* ``two_level``: the two-level string `arm` (x and y) as what is studied, so every model serves its
  one indicator `arm_y` (x the reference, as the one-hot step drops the first level);
* ``family_category``: the family {`fiber`, `grp`} under the linear model, so every model serves
  `fiber` as its own column and `grp` as two rows, b and c against a;
* ``family_level``: the family {`fiber`, `arm`}, so every model serves `fiber` and `arm_y`;
* ``family_apart``: the family {`fiber`, `sugar`} under feature-wise regression, which fits each
  member in its own model, so every model serves `fiber` and `sugar`, neither adjusted for the
  other.

Each row the unadjusted model reports is checked against statsmodels' glucose ~ its columns
written out here from the construction (``capture_sequences._crude``: the covariance its table
names, the estimate, p-value and 95% interval kept beside it as ``truth``): the members together
under the linear model, where each model names the members it holds among the columns it is
adjusted for (pp2-family-effect-words), and each member alone under feature-wise regression.

    OMP_NUM_THREADS=2 TURBOTAB_WORKERS=2 scripts/py turbotab/frontend/src/quest/results/fixtures/capture_terms.py

writes ``estimate-terms.json`` beside this file; nothing in it is written by hand.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))  # capture_sequences, beside this file

import capture_sequences as CS  # noqa: E402
from turbotab.core import decisions as d  # noqa: E402
from turbotab.core.tests import test_featurewise_family_counts_its_tests as FW  # noqa: E402
from turbotab.core.tests.acceptance import estimand_fixtures as ef  # noqa: E402

OUT = Path(__file__).with_name("estimate-terms.json")
CUT = 20.0


def _served(effects: dict[str, Any], features: list[str]) -> None:
    """Every declared model serves exactly ``features``, in that order."""
    for fit in effects["families"][0]["sequence"]:
        assert [r["feature"] for r in fit["effects"]] == features, (fit["key"], fit["effects"])


def _holds(effects: dict[str, Any], members: list[str]) -> None:
    """Every declared model names ``members`` among the columns it is adjusted for, after the
    covariates: the linear model holds them together, each adjusted for the others."""
    for fit in effects["families"][0]["sequence"]:
        assert fit["adjusted_for"][-len(members):] == members or (
            fit["key"] == "model_3" and fit["adjusted_for"][-len(members) - 1:-1] == members), fit
    assert ef.sequence(effects)["crude"]["adjusted_for"] == members


def _alone(frame: pd.DataFrame, effects: dict[str, Any], members: list[str]) -> dict[str, Any]:
    """``capture_sequences._crude`` for one fit per member: each unadjusted row checked against
    statsmodels' glucose ~ its member alone, under the covariance its table names."""
    crude = ef.sequence(effects)["crude"]
    covariance = crude["inference"]["covariance"]
    truth: dict[str, Any] = {}
    for m in members:
        fit = sm.OLS(frame["glucose"].to_numpy(float), sm.add_constant(frame[[m]].astype(float))
                     ).fit(**CS.COVARIANCE[covariance])
        limits = fit.conf_int(0.05)
        want = {"estimate": float(fit.params[m]), "p": float(fit.pvalues[m]),
                "ci_low": float(limits.loc[m, 0]), "ci_high": float(limits.loc[m, 1])}
        row = next(r for r in crude["effects"] if r["feature"] == m)
        for k in ("estimate", "ci_low", "ci_high"):
            assert abs(row[k] - want[k]) <= 1e-9 * abs(want[k]), (m, k, row[k], want[k])
        assert abs(row["p"] - want["p"]) <= 1e-6 * want["p"], (m, row["p"], want["p"])
        truth[m] = want
    return {"covariance": covariance, "rows": truth}


def density(folder: Path) -> dict[str, Any]:
    frame = ef.cohort(n=CS.N, seed=7)
    frame["kcal"] = np.random.default_rng(5).normal(2000, 400, CS.N).clip(900)
    roles = {**ef.ROLES, "kcal": "energy"}
    st = ef.state(target="glucose", task="regression", measure="mean_difference", roles=roles,
                  plan_locked=True,
                  energy_adjustment=d.EnergyAdjustment(method="density", energy_column="kcal",
                                                       nutrients=["fiber"]))
    effects = ef.run(frame, folder, st, fit=False)["effects"]
    _served(effects, ["fiber_per_kcal"])
    X = pd.DataFrame({"fiber_per_kcal": frame["fiber"] / frame["kcal"]})
    return {"artifact": effects, "truth": {"crude": CS._crude(frame, effects, X)}}


def cut(folder: Path) -> dict[str, Any]:
    frame = ef.cohort(n=CS.N, seed=7)
    st = ef.state(target="glucose", task="regression", measure="mean_difference", plan_locked=True,
                  exposure_forms={"fiber": d.ExposureFormSpec(form="categories", cuts=[CUT])})
    effects = ef.run(frame, folder, st, fit=False)["effects"]
    _served(effects, ["fiber_C2"])
    X = pd.DataFrame({"fiber_C2": (frame["fiber"] > CUT).astype(float)})
    return {"artifact": effects, "truth": {"crude": CS._crude(frame, effects, X)}}


def two_level(folder: Path) -> dict[str, Any]:
    frame = FW._frame()
    st = FW._state({"arm": "exposure", "fiber": "covariate"}, ["linear"], exposure="arm")
    effects = ef.run(frame, folder, st, fit=False)["effects"]
    _served(effects, ["arm_y"])
    X = pd.DataFrame({"arm_y": (frame["arm"] == "y").astype(float)})
    return {"artifact": effects, "truth": {"crude": CS._crude(frame, effects, X)}}


def family_category(folder: Path) -> dict[str, Any]:
    frame = FW._frame()
    st = FW._family({"fiber": "exposure", "grp": "exposure"}, ["linear"])
    effects = ef.run(frame, folder, st, fit=False)["effects"]
    assert effects["exposure"] == d.EXPOSURE_FAMILY and effects["exposures"] == ["fiber", "grp"]
    _served(effects, ["grp_b", "grp_c", "fiber"])
    _holds(effects, ["fiber", "grp"])
    X = pd.DataFrame({**{f"grp_{v}": (frame["grp"] == v).astype(float) for v in ("b", "c")},
                      "fiber": frame["fiber"]})
    return {"artifact": effects, "truth": {"crude": CS._crude(frame, effects, X)}}


def family_level(folder: Path) -> dict[str, Any]:
    frame = FW._frame()
    st = FW._family({"fiber": "exposure", "arm": "exposure"}, ["linear"])
    effects = ef.run(frame, folder, st, fit=False)["effects"]
    assert effects["exposure"] == d.EXPOSURE_FAMILY and effects["exposures"] == ["fiber", "arm"]
    _served(effects, ["arm_y", "fiber"])
    _holds(effects, ["fiber", "arm"])
    X = pd.DataFrame({"arm_y": (frame["arm"] == "y").astype(float), "fiber": frame["fiber"]})
    return {"artifact": effects, "truth": {"crude": CS._crude(frame, effects, X)}}


def family_apart(folder: Path) -> dict[str, Any]:
    frame = FW._frame()
    st = FW._family({"fiber": "exposure", "sugar": "exposure"}, ["featurewise"])
    effects = ef.run(frame, folder, st, fit=False)["effects"]
    assert effects["exposure"] == d.EXPOSURE_FAMILY and effects["exposures"] == ["fiber", "sugar"]
    _served(effects, ["fiber", "sugar"])
    for fit in effects["families"][0]["sequence"]:
        assert not {"fiber", "sugar"} & set(fit["adjusted_for"]), fit  # neither held by the other
    return {"artifact": effects, "truth": {"crude": _alone(frame, effects, ["fiber", "sugar"])}}


def main() -> int:
    t0 = time.monotonic()
    out: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="estimate-terms-") as folder:
        for name, build in (("density", density), ("cut", cut), ("two_level", two_level),
                            ("family_category", family_category), ("family_level", family_level),
                            ("family_apart", family_apart)):
            out[name] = build(Path(folder) / name)
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                          text=True, cwd=Path(__file__).parent).stdout.strip()
    out = {"meta": {"journey": "estimate-terms", "captured": date.today().isoformat(),
                    "head": head, "seconds": round(time.monotonic() - t0),
                    "script": "turbotab/frontend/src/quest/results/fixtures/capture_terms.py"},
           **out}
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes) in {out['meta']['seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

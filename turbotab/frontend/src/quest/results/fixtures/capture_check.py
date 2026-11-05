"""Captures the effects artifacts whose family carries the plan's check of the exposure's shape
(core/stages/effects.py ``FormCheck``, beside Model 2), as Estimate Results draws it beside Table 2:
each state built by the engine test's own construction
(``turbotab/core/tests/test_form_check_beside_the_estimate.py``) and run through the real design,
fit and effects stages (``estimand_fixtures.run``) on every row:

* ``line``: a straight line kept for `fiber` on the cohort with a known bend (``bent``: 0.25
  (fiber − 20)² added to `glucose`), so the check is the restricted cubic spline in its place,
  its Wald F test of the nonlinear terms and its curve against the line;
* ``spline``: a 5-knot restricted cubic spline kept for `fiber` on the same cohort, so the check is
  the straight line's one coefficient;
* ``refused``: a straight line kept under multiple imputation with `fiber` and `activity` partly
  blank, so the check is refused in the engine's words (the copies were imputed for the line);
* ``none``: no form declared for `fiber`, so no check is set (``check`` null).

Each served check is checked against statsmodels as the engine test checks it (the spline's knots,
F test, terms and curve, ``_assert_check_is_statsmodels``; the line's coefficient, least squares
with HC3), and that truth is kept beside the artifact (``truth``): the F test, the line's
coefficient and its interval, and the curve's points written out from statsmodels' coefficients.
The refusal's words are the engine test's (``imputed_refusal``).

    OMP_NUM_THREADS=2 TURBOTAB_WORKERS=2 scripts/py turbotab/frontend/src/quest/results/fixtures/capture_check.py

writes ``estimate-check.json`` beside this file; nothing in it is written by hand.
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
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))  # the repository's root: `turbotab`

import turbotab.core.models  # noqa: E402,F401 - registers the families
from turbotab.core import decisions as d  # noqa: E402
from turbotab.core.tests import test_form_check_beside_the_estimate as FC  # noqa: E402
from turbotab.core.tests.acceptance import estimand_fixtures as ef  # noqa: E402

OUT = Path(__file__).with_name("estimate-check.json")
SPLINE = d.ExposureFormSpec(form="spline", knots=5, scale="raw")


def _check(effects: dict[str, Any]) -> dict[str, Any]:
    [family] = effects["families"]
    check = family["check"]
    assert check is not None, family
    return check


def line(folder: Path) -> dict[str, Any]:
    frame = FC.bent()
    effects = ef.run(frame, folder, FC.numeric_state(exposure_forms={"fiber": FC.LINE}))["effects"]
    check = _check(effects)
    truth = FC._statsmodels_check(frame)
    FC._assert_check_is_statsmodels(check, truth)
    # The curve's points as statsmodels' coefficients give them (the engine test's own sum).
    knots = truth["knots"]
    grid = np.linspace(knots[0], knots[-1], 50)
    basis = FC.harrell(grid, knots) - FC.harrell(np.array([knots[0]]), knots)
    fit = truth["curve"]
    b = np.array([float(fit.params[n]) for n in FC.spline_names(5)])
    curve = float(fit.params["fiber"]) * (grid - knots[0]) + basis @ b
    slope = float(truth["line"].params["fiber"])
    return {"artifact": effects, "truth": {
        "test": {"statistic": truth["F"], "df_num": truth["df_num"], "df_den": truth["df_den"],
                 "p": truth["p"]},
        "reference": float(knots[0]),
        "points": [{"x": float(x), "curve": float(c), "line": float(slope * (x - knots[0]))}
                   for x, c in zip(grid, curve)]}}


def spline(folder: Path) -> dict[str, Any]:
    frame = FC.bent()
    effects = ef.run(frame, folder, FC.numeric_state(exposure_forms={"fiber": SPLINE}))["effects"]
    check = _check(effects)
    assert check["refused"] is None, check
    assert (check["kept"], check["shape"], check["column"]) == ("spline", "linear", "fiber")
    assert check["test"] is None and check["points"] == [] and check["knots"] is None
    fit = sm.OLS(frame["glucose"].astype(float),
                 ef.design_matrix(frame, ["fiber", *FC.COVARIATES])).fit(cov_type="HC3",
                                                                         use_t=True)
    low, high = (float(v) for v in fit.conf_int(alpha=0.05).loc["fiber"])
    want = {"estimate": float(fit.params["fiber"]), "ci_low": low, "ci_high": high,
            "p": float(fit.pvalues["fiber"])}
    got = check["line"]
    for k in ("estimate", "ci_low", "ci_high"):
        assert abs(got[k] - want[k]) <= 1e-8 * abs(want[k]), (k, got[k], want[k])
    assert abs(got["p"] - want["p"]) <= 1e-6 * want["p"] + 1e-300, (got["p"], want["p"])
    return {"artifact": effects, "truth": {"line": want}}


def refused(folder: Path) -> dict[str, Any]:
    frame = FC.bent(n=500, seed=8)
    rng = np.random.default_rng(80)
    frame.loc[rng.random(len(frame)) < 0.15, "fiber"] = np.nan
    frame.loc[rng.random(len(frame)) < 0.15, "activity"] = np.nan
    st = FC.numeric_state(exposure_forms={"fiber": FC.LINE},
                          missing=d.MissingSpec(strategy="multiple_imputation"))
    effects = ef.run(frame, folder, st)["effects"]
    check = _check(effects)
    words = FC.imputed_refusal("(by chained equations)")
    assert check["refused"] == words, check["refused"]
    assert check["caption"] == "The plan's check of the straight line could not be made: " + words
    assert check["test"] is None and check["knots"] is None and check["points"] == []
    assert check["line"] is None and check["reference"] is None
    return {"artifact": effects, "refused": words}


def none(folder: Path) -> dict[str, Any]:
    frame = FC.straight(n=400, seed=3)
    effects = ef.run(frame, folder, FC.numeric_state())["effects"]
    [family] = effects["families"]
    assert family["check"] is None, family["check"]
    return {"artifact": effects}


def main() -> int:
    t0 = time.monotonic()
    out: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="estimate-check-") as folder:
        for name, build in (("line", line), ("spline", spline), ("refused", refused),
                            ("none", none)):
            out[name] = build(Path(folder) / name)
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                          text=True, cwd=Path(__file__).parent).stdout.strip()
    out = {"meta": {"journey": "estimate-check", "captured": date.today().isoformat(),
                    "head": head, "seconds": round(time.monotonic() - t0),
                    "script": "turbotab/frontend/src/quest/results/fixtures/capture_check.py"},
           **out}
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes) in {out['meta']['seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

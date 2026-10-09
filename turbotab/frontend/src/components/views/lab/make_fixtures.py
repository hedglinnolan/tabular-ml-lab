"""The illustrative fixtures for /lab/views where the engine serves no artifact yet.

Run from the repository root: ``venv/bin/python turbotab/frontend/src/components/views/lab/make_fixtures.py``.
It writes ``illustrative.json`` beside itself. Every number is computed by the engine's own
functions where one exists (``decision_curve``, ``calibration``) on a seeded synthetic sample, so
the shapes are the engine's; what the engine does not compute yet (the binned calibration points,
the exposure curve with its band) is computed here and named in the views' engine contract items.
The captured journeys (``src/mocks/fixtures/m3-*.json``) supply the rest of the lab directly.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from turbotab.core.models.decision_curve import decision_curve, grid, useful_range
from turbotab.core.models.performance import calibration

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(20261009)


def r(x: float, places: int = 6) -> float:
    return float(round(float(x), places))


def expit(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


# ── a binary outcome, two models' out-of-fold risks ──────────────────────────
n = 1200
z1 = rng.normal(size=n)
z2 = rng.normal(size=n)
truth = expit(-1.5 + 1.1 * z1 + 0.6 * z2)
event = (rng.uniform(size=n) < truth).astype(float)
# The reported model is overconfident (slope below 1); the benchmark sees one predictor.
reported = expit(-1.6 + 1.45 * z1 + 0.8 * z2 + rng.normal(scale=0.25, size=n))
benchmark = expit(-1.45 + 1.0 * z1)

# Decision curve over a wider grid than the declared range, so the range can be shaded.
declared = (0.05, 0.50)
thresholds = grid(0.01, 0.70)
rows = decision_curve(event, {"boosted_trees": reported, "spline_benchmark": benchmark}, thresholds)
dc = {
    "family": "boosted_trees",
    "labels": {"boosted_trees": "Boosted trees", "spline_benchmark": "Spline benchmark"},
    "low": declared[0],
    "high": declared[1],
    "prevalence": r(event.mean()),
    "n": n,
    "rows": [{"threshold": r(x["threshold"], 4), "treat_all": r(x["treat_all"]), "treat_none": 0.0,
              "models": {k: r(v) for k, v in x["models"].items()}} for x in rows],
    "useful": list(useful_range([x for x in rows if declared[0] <= x["threshold"] <= declared[1]],
                                "boosted_trees") or []) or None,
}

# Calibration: the engine's (in the large, slope, lowess curve), plus ten bins of predicted risk
# with Wilson 95% intervals on each bin's observed share (not served by the engine yet).
cal = calibration("binary", event, reported).model_dump()
order = np.argsort(reported, kind="stable")
bins = []
for part in np.array_split(order, 10):
    k, m = float(event[part].sum()), len(part)
    p = k / m
    zq = 1.959964
    center = (p + zq * zq / (2 * m)) / (1 + zq * zq / m)
    half = zq * math.sqrt(p * (1 - p) / m + zq * zq / (4 * m * m)) / (1 + zq * zq / m)
    bins.append({"predicted": r(reported[part].mean()), "observed": r(p), "low": r(center - half),
                 "high": r(center + half), "n": m})
cal["curve"] = [{"x": r(c["x"]), "y": r(c["y"])} for c in cal["curve"]]
cal["bins"] = bins
cal["bins_method"] = "ten equal-count groups of predicted risk; Wilson 95% intervals"

# ── an exposure curve: a restricted cubic spline's difference from a reference, with its band ──
exposure = np.clip(rng.gamma(shape=4.0, scale=22.0, size=900), 2, None)  # g/day, right-skewed
xs = np.linspace(float(np.quantile(exposure, 0.01)), float(np.quantile(exposure, 0.99)), 41)
ref = float(np.median(exposure))


def shape(x: np.ndarray) -> np.ndarray:
    return 4.0 * np.log1p(x / 40.0) - 4.0 * math.log1p(ref / 40.0)


def width(x: np.ndarray) -> np.ndarray:
    return 0.6 + 0.022 * np.abs(x - ref) ** 1.15 / 4


curve_now = shape(xs)
curve_choice = 0.055 * (xs - ref)  # the same exposure entered as a straight line
band_now = width(xs)
band_choice = 0.35 + 0.006 * np.abs(xs - ref)
rug = np.quantile(exposure, np.linspace(0.005, 0.995, 120))
ec = {
    "exposure": "sugar",
    "unit": "g/day",
    "outcome": "glucose",
    "outcome_unit": "mg/dL",
    "reference": r(ref, 2),
    "now": {"label": "A curve (4 knots)", "x": [r(v, 3) for v in xs], "y": [r(v, 4) for v in curve_now],
            "low": [r(v, 4) for v in curve_now - band_now], "high": [r(v, 4) for v in curve_now + band_now]},
    "choice": {"label": "A straight line", "x": [r(v, 3) for v in xs], "y": [r(v, 4) for v in curve_choice],
               "low": [r(v, 4) for v in curve_choice - band_choice],
               "high": [r(v, 4) for v in curve_choice + band_choice]},
    "rug": [r(v, 2) for v in rug],
    "n": len(exposure),
}

out = {
    "about": "Illustrative: a seeded synthetic sample through the engine's own functions "
             "(make_fixtures.py). Not a captured journey.",
    "decision_curve": dc,
    "calibration": cal,
    "exposure_curve": ec,
}
(HERE / "illustrative.json").write_text(json.dumps(out, indent=1) + "\n")
print("wrote", HERE / "illustrative.json")

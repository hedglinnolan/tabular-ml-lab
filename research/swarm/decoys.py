"""Null decoys for the swarm (SWARM_PLAN §5): a generated table whose outcome no column predicts.

A decoy is drawn from one seed. ``numpy.random.SeedSequence(seed).spawn(2)`` gives two
independent random streams: the first draws the columns, the second the outcome. The outcome is
drawn from its own stream alone (:func:`outcome` is given no column), so it is independent of
every column by construction, whatever the columns hold: any association or predictive skill a
run reports on a decoy is a false discovery.

The columns are drawn to look like an ordinary outpatient table, related to one another as such
measures are (blood pressure rises with age and BMI, waist with BMI, HDL falls with BMI and is
lower in men), with a few blanks at random, so that nothing but the outcome's independence marks
the table. The blanks come from the columns' stream too, so whether a value is missing tells
nothing about the outcome either.

    scripts/py -m research.swarm.decoys <seed> <out.csv> [--rows N]

writes one decoy table (the intake, ``research/swarm/intake.py``, writes the pilot's).

**Distinct seeds** (SWARM_PLAN §5: the null decoy is mechanical only, "with distinct seeds"). The
decoy's runs are its mechanical arms, each a plan replayed on a fresh draw
(``core/export/replay.py``, ``FreshDraw("research.swarm.decoys:generate", <the intake's seed>,
<the run's seed>)``). One table and one split for every run made them close to one test (the plan
audit, 2026-10-19); :func:`seeds` gives run k its own seed, :data:`FIRST_SEED` + k, so no two runs
share an outcome stream, none is the analysts' table's (the intake's ``DECOY_SEED``), and each
run's false finding is its own chance.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROWS = 1600
OUTCOME = "admitted_2y"
# The mechanical runs' seeds start above the analysts' table's (the intake's DECOY_SEED): run k
# (1 up) draws its table at FIRST_SEED + k.
FIRST_SEED = 20262000
PREVALENCE = 0.2  # the probability of the outcome on every row, whatever its columns
# The share of rows left blank, by column, at random (from the columns' stream).
BLANKS = {"ldl": 0.07, "hdl": 0.07, "steps_per_day": 0.12, "drinks_per_week": 0.03}


def streams(seed: int) -> tuple[np.random.Generator, np.random.Generator]:
    """The columns' stream and the outcome's, independent of each other."""
    columns, outcome_ = np.random.SeedSequence(seed).spawn(2)
    return np.random.default_rng(columns), np.random.default_rng(outcome_)


def columns(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """An outpatient table's columns, drawn from ``rng`` alone."""
    age = np.clip(rng.normal(54, 14, n), 18, 90).round(0)
    male = rng.random(n) < 0.47
    bmi = np.clip(rng.normal(28.5, 5.5, n), 16, 60).round(1)
    waist = np.clip(30 + 2.2 * bmi + 6 * male + rng.normal(0, 6, n), 55, 170).round(1)
    sbp = np.clip(88 + 0.45 * age + 0.6 * bmi + rng.normal(0, 13, n), 80, 230).round(0)
    dbp = np.clip(30 + 0.35 * sbp + rng.normal(0, 7, n), 40, 130).round(0)
    ldl = np.clip(rng.normal(118 + 0.3 * (age - 54), 32, n), 30, 300).round(0)
    hdl = np.clip(62 - 9 * male - 0.6 * (bmi - 28.5) + rng.normal(0, 12, n), 15, 130).round(0)
    hba1c = np.clip(5.4 + 0.02 * (bmi - 28.5) + 0.008 * (age - 54) + rng.normal(0, 0.5, n),
                    4.0, 13.0).round(1)
    smoking = rng.choice(["never", "former", "current"], n, p=[0.55, 0.28, 0.17])
    drinks = np.where(rng.random(n) < 0.35, 0, rng.poisson(4.0, n)).astype(float)
    steps = np.clip(rng.lognormal(np.log(6500), 0.45, n) - 25 * (age - 54), 300, 30000)
    steps = (steps / 10).round(0) * 10
    sleep = np.clip(rng.normal(6.9, 1.1, n), 3, 12).round(1)
    education = rng.choice(["less than high school", "high school", "some college", "college"],
                           n, p=[0.12, 0.28, 0.30, 0.30])
    clinic = rng.choice(["North", "South", "East"], n, p=[0.40, 0.35, 0.25])
    out = {
        "patient_id": np.array([f"P{i + 1:05d}" for i in range(n)]),
        "clinic": clinic, "age": age, "sex": np.where(male, "M", "F"), "bmi": bmi,
        "waist_cm": waist, "sbp": sbp, "dbp": dbp, "ldl": ldl, "hdl": hdl, "hba1c": hba1c,
        "smoking": smoking, "drinks_per_week": drinks, "steps_per_day": steps,
        "sleep_hours": sleep, "education": education,
    }
    lab = rng.random(n) < BLANKS["ldl"]  # one lab panel: LDL and HDL are blank together
    for name, share in BLANKS.items():
        blank = lab if name in ("ldl", "hdl") else rng.random(n) < share
        out[name] = np.where(blank, np.nan, out[name].astype(float))
    return out


def outcome(seed: int, n: int, prevalence: float = PREVALENCE) -> np.ndarray:
    """The outcome, 1 with probability ``prevalence`` on every row, from the outcome's stream:
    no column enters it."""
    _, rng = streams(seed)
    return (rng.random(n) < prevalence).astype(int)


def generate(seed: int, n: int = ROWS) -> pd.DataFrame:
    """The decoy table of ``seed``: :func:`columns` from the first stream, :func:`outcome` from
    the second."""
    rng, _ = streams(seed)
    frame = pd.DataFrame(columns(rng, n))
    for name in ("age", "sbp", "dbp", "ldl", "hdl", "drinks_per_week", "steps_per_day"):
        frame[name] = frame[name].astype("Int64")  # whole numbers, written without a decimal
    frame[OUTCOME] = outcome(seed, n)
    return frame


def seeds(count: int, first: int = FIRST_SEED) -> list[int]:
    """The seeds of ``count`` mechanical decoy runs: ``first`` + 1 to ``first`` + ``count``,
    distinct by construction."""
    if count < 0:
        raise ValueError("a count of runs is not negative")
    return [first + k for k in range(1, count + 1)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("seed", type=int)
    parser.add_argument("out", type=Path)
    parser.add_argument("--rows", type=int, default=ROWS)
    args = parser.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    generate(args.seed, args.rows).to_csv(args.out, index=False)
    print(f"wrote {args.out} ({args.rows} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

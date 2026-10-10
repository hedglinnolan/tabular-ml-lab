"""NHANES-shaped files, one per cycle, for the stacking and trend tests (C1b, D6).

Each cycle has its own masked variance strata, two PSUs each (one stratum with three), people
clustered in PSUs with a PSU effect, and informative examination weights (an oversampled group with
weights about a fifth of the others'). Body mass index rises across cycles with a bend in the last
two; obesity is BMI ≥ 30. The 2017–March 2020 file carries ``WTMECPRP``, the others ``WTMEC2YR``.
Every draw is seeded; the truth is the generator's.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# release code -> (shift in mean BMI, seed)
SHIFTS = {7: 0.0, 8: 0.5, 9: 1.0, 66: 2.2, 10: 1.6, 6: -0.3, 5: -0.6, 1: -1.5, 2: -1.2, 3: -1.0}


def nhanes_cycle(code: int, *, strata: int = 8, n_per_psu: int = 30, seed: int | None = None,
                 strata_start: int | None = None, lonely: bool = False, shift: float | None = None,
                 weight: str | None = None, four_year: bool = False) -> pd.DataFrame:
    """One cycle's file: ``SEQN``, ``SDDSRVYR``, ``SDMVSTRA``, ``SDMVPSU``, the weight, ``RIDAGEYR``,
    ``RIAGENDR``, ``BMXBMI``. ``strata_start`` numbers the strata (default unique per release, as
    NCHS numbers them); ``lonely`` gives the last stratum one PSU."""
    rng = np.random.default_rng(seed if seed is not None else 1000 + code)
    start = strata_start if strata_start is not None else 10 * code
    shift = SHIFTS.get(code, 0.0) if shift is None else shift
    rows = []
    seqn = code * 100_000
    for h in range(strata):
        n_psu = 1 if (lonely and h == strata - 1) else (3 if h == 0 else 2)
        for j in range(n_psu):
            effect = rng.normal(0, 0.8)
            for _ in range(n_per_psu + int(rng.integers(-5, 6))):
                seqn += 1
                over = rng.random() < 0.4
                age = int(rng.integers(2, 80))
                female = int(rng.random() < 0.5)
                bmi = (24.5 + shift + effect + (1.8 if over else 0.0) + 0.04 * min(age, 60)
                       + rng.normal(0, 4.5))
                if age < 20:
                    bmi -= 5
                w = (rng.uniform(2000, 6000) if over else rng.uniform(10_000, 40_000))
                rows.append({"SEQN": seqn, "SDDSRVYR": code, "SDMVSTRA": start + h,
                             "SDMVPSU": j + 1, "W": w, "RIDAGEYR": age, "RIAGENDR": 1 + female,
                             "BMXBMI": round(bmi, 1)})
    frame = pd.DataFrame(rows)
    name = weight or ("WTMECPRP" if code == 66 else "WTMEC2YR")
    frame = frame.rename(columns={"W": name})
    if four_year:
        frame["WTMEC4YR"] = frame[name] * rng.uniform(0.45, 0.55, len(frame))
    # A few with no exam (weight zero), as NHANES's interview-only participants
    zero = rng.random(len(frame)) < 0.03
    frame.loc[zero, name] = 0.0
    frame.loc[zero, "BMXBMI"] = np.nan
    return frame


def four_cycles(**kw) -> dict[int, pd.DataFrame]:
    """2011–2012, 2013–2014, 2015–2016 and 2017–March 2020 (7.2 years; midpoints 2012, 2014, 2016,
    2018.6)."""
    return {code: nhanes_cycle(code, **kw) for code in (7, 8, 9, 66)}


__all__ = ["four_cycles", "nhanes_cycle"]

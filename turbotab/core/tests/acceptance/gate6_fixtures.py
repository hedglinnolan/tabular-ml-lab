"""The sixth intelligence gate's generators, draw for draw (``/private/tmp/turbotab-fix/gate6``).

Each function replays one probe's table with the probe's own seed and the draws in its order, so
a test reads the very table the gate read: ``q3_text_numbers.table`` (rng 66003),
``q1_factor_alcohol`` (rng 66001), ``q2_alcohol_substitution.table`` (rng 66002) and
``q9_nested.table`` (rng 66009). Shared by ``test_ledger_repair_3`` and the kind registry's
alternative fixtures (``test_discriminate``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def text_numbers_table(n: int = 500) -> pd.DataFrame:
    """q3_text_numbers.py (rng 66003): a CRP whose results below 0.2 are written ``<0.20``, a
    vitamin D written with decimal commas, a BMI exported from SAS with ``.`` on every 50th row
    (from row 7), and a systolic pressure."""
    rng = np.random.default_rng(66003)
    f = pd.DataFrame({"participant_id": [f"T{i:04d}" for i in range(n)]})
    f["age"] = rng.normal(50, 12, n).round(1)
    crp = np.exp(rng.normal(0.3, 1.0, n))
    f["crp"] = [("<0.20" if v < 0.2 else f"{v:.2f}") for v in crp]
    vitd = rng.normal(55, 18, n).clip(8)
    f["vitd"] = [f"{v:.1f}".replace(".", ",") for v in vitd]
    bmi = rng.normal(27, 4.5, n)
    f["bmi"] = [("." if i % 50 == 7 else f"{v:.1f}") for i, v in enumerate(bmi)]
    f["sbp"] = 0.3 * (bmi - 27) + (118 + 0.4 * (f["age"] - 50) + 1.2 * np.log(np.maximum(crp, 0.1))
                                   - 0.05 * vitd + rng.normal(0, 9, n)).round(1)
    return f


def alcohol_factor_frame(name: str, n: int = 800) -> pd.DataFrame:
    """q1_factor_alcohol.py (rng 66001): a 24-h recall's protein, carbohydrate and fat in grams,
    alcohol recorded in US standard drinks a day under ``name`` (45% abstainers, the rest a
    gamma(1.5, 1) to one decimal), and total energy in kcal built with 7 kcal per gram of the
    drinks' 14 g, times a 3% noise."""
    rng = np.random.default_rng(66001)
    P = rng.normal(80, 20, n).clip(20)
    C = rng.normal(250, 60, n).clip(50)
    F = rng.normal(75, 20, n).clip(15)
    drinks = np.where(rng.random(n) < 0.45, 0, rng.gamma(1.5, 1.0, n)).round(1)
    E = (4 * P + 4 * C + 9 * F + 7 * (14 * drinks)) * rng.normal(1, 0.03, n)
    return pd.DataFrame({"energy_kcal": E.round(0), "protein_g": P.round(1), "fat_g": F.round(1),
                         "carbohydrate_g": C.round(1), name: drinks})


def alcohol_substitution_table(name: str = "alcohol", n: int = 600) -> pd.DataFrame:
    """q2_alcohol_substitution.py (rng 66002): the same recall with alcohol in US standard drinks
    under ``name``, energy built at 98 kcal a drink, and a systolic pressure rising 1.5 mmHg a
    drink and falling 0.01 per gram of carbohydrate."""
    rng = np.random.default_rng(66002)
    f = pd.DataFrame({"participant_id": [f"A{i:04d}" for i in range(n)]})
    f["age"] = rng.normal(50, 12, n).round(1)
    P = rng.normal(80, 20, n).clip(20)
    C = rng.normal(250, 60, n).clip(50)
    F = rng.normal(75, 20, n).clip(15)
    drinks = np.where(rng.random(n) < 0.45, 0, rng.gamma(1.5, 1.0, n)).round(1)
    f["protein_g"] = P.round(1)
    f["fat_g"] = F.round(1)
    f["carbohydrate_g"] = C.round(1)
    f[name] = drinks
    f["energy_kcal"] = ((4 * P + 4 * C + 9 * F + 98 * drinks) * rng.normal(1, 0.03, n)).round(0)
    f["sbp"] = (118 + 0.4 * (f["age"] - 50) + 1.5 * drinks - 0.01 * C
                + rng.normal(0, 9, n)).round(1)
    return f


def nested_table(n: int = 500) -> pd.DataFrame:
    """q9_nested.py (rng 66009): saturated fat `sfa_g` a 25–45% part of `fat_g`, beside protein,
    carbohydrate, total energy (4P + 4C + 9F) and an LDL rising with `sfa_g`."""
    rng = np.random.default_rng(66009)
    f = pd.DataFrame({"participant_id": [f"N{i:04d}" for i in range(n)]})
    f["age"] = rng.normal(50, 12, n).round(1)
    P = rng.normal(80, 20, n).clip(20)
    C = rng.normal(250, 60, n).clip(50)
    F = rng.normal(75, 20, n).clip(15)
    f["protein_g"] = P.round(1)
    f["fat_g"] = F.round(1)
    f["carbohydrate_g"] = C.round(1)
    f["sfa_g"] = (F * rng.uniform(0.25, 0.45, n)).round(1)
    f["energy_kcal"] = (4 * P + 4 * C + 9 * F).round(0)
    f["ldl"] = (110 + 0.6 * f["sfa_g"] - 0.05 * C + rng.normal(0, 20, n)).round(1)
    return f


def minor_protein_frame(n: int = 600) -> pd.DataFrame:
    """A diet whose protein is a minor source (about 3% of energy: 15 g against 250 g of
    carbohydrate and 75 g of fat), recorded in grams, with total energy exact in kcal: the Atwater
    identity holds read in grams, and would hold were protein in kcal too (rng 66010)."""
    rng = np.random.default_rng(66010)
    P = rng.normal(15, 3, n).clip(5)
    C = rng.normal(250, 60, n).clip(50)
    F = rng.normal(75, 20, n).clip(15)
    return pd.DataFrame({"energy": (4 * P + 4 * C + 9 * F).round(0), "protein_g": P.round(1),
                         "carbohydrate_g": C.round(1), "fat_g": F.round(1)})

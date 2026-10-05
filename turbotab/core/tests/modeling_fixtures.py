"""Small fixtures for the model-side stages: an NHANES-shaped table, its ingest, and the split
and cohort artifacts the rows agent's stages produce (M1_CONTRACT §3 shapes)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from turbotab.core.decisions import (  # noqa: F401 - ColumnUnitSpec re-exported for tests
    ColumnUnitSpec, EnergyAdjustment, ProjectState, SplitSpec, SubstitutionSpec,
)
from turbotab.core.graph import Bundle, StageContext

NUTRIENTS = ["protein", "sugar", "carb", "fat_total", "fat_sat", "fat_mon", "fat_poly"]
NHANES_ROLES = {
    "SEQN": "identifier", "cycle_begin_year": "time", "age": "covariate", "gender": "covariate",
    "bp_sys": "covariate", "bp_di": "covariate", "weight": "covariate", "height": "covariate",
    "bmi": "covariate", "waist": "covariate", "kcal": "energy",
    **{n: "exposure" for n in NUTRIENTS},
    "hdl": "covariate", "triglycerides": "covariate", "meds_hbp": "covariate",
    "meds_chol": "covariate", "imputed_weight": "flag", "imputed_bmi": "flag",
}


def nhanes_like(n: int = 400, seed: int = 0, missing: bool = False, repeats: int = 1) -> pd.DataFrame:
    """The real export's columns (names and kinds), simulated; glucose depends on the diet."""
    rng = np.random.default_rng(seed)
    people = n // repeats
    seqn = np.repeat(np.arange(people) + 9966.0, repeats)[:n]
    age = np.repeat(rng.integers(20, 80, people).astype(float), repeats)[:n]
    gender = np.repeat(rng.choice(["female", "male"], people), repeats)[:n]
    intake = rng.normal(2100, 500, n).clip(700, 4500)
    protein = (10 + 0.035 * intake + rng.normal(0, 12, n)).clip(10)
    carb = (20 + 0.11 * intake + rng.normal(0, 30, n)).clip(30)
    sugar = (carb * rng.uniform(0.2, 0.5, n) + rng.normal(0, 5, n)).clip(5)
    fat_total = (5 + 0.035 * intake + rng.normal(0, 12, n)).clip(8)
    # Total energy is the Atwater sum plus a little from alcohol and rounding, as recalls record it.
    kcal = 4 * protein + 4 * carb + 9 * fat_total + rng.gamma(2.0, 40.0, n)
    # The parts do not add up to the total exactly, as in the real export (other fats, rounding).
    fat_sat = fat_total * rng.uniform(0.28, 0.38, n) + rng.normal(0, 2, n)
    fat_mon = fat_total * rng.uniform(0.32, 0.40, n) + rng.normal(0, 2, n)
    fat_poly = fat_total * rng.uniform(0.15, 0.25, n) + rng.normal(0, 2, n)
    weight = rng.normal(80, 15, n).clip(40)
    height = rng.normal(170, 10, n).clip(140)
    bmi = weight / (height / 100) ** 2
    glucose = (70 + 0.25 * age + 0.6 * bmi + 0.03 * carb - 0.05 * fat_total + 0.004 * kcal
               + 3 * (gender == "male") + rng.normal(0, 8, n))
    frame = pd.DataFrame({
        "SEQN": seqn, "cycle_begin_year": rng.choice([2001, 2003, 2005], n), "age": age,
        "gender": gender, "bp_sys": rng.normal(122, 15, n), "bp_di": rng.normal(70, 10, n),
        "weight": weight, "height": height, "bmi": bmi, "waist": rng.normal(95, 12, n),
        "kcal": kcal, "protein": protein, "sugar": sugar, "carb": carb, "fat_total": fat_total,
        "fat_sat": fat_sat, "fat_mon": fat_mon, "fat_poly": fat_poly,
        "hdl": rng.normal(52, 14, n), "glucose": glucose, "triglycerides": rng.normal(130, 50, n),
        "meds_hbp": rng.choice([True, False], n), "meds_chol": rng.choice([True, False], n),
        "imputed_weight": rng.random(n) < 0.02, "imputed_bmi": rng.random(n) < 0.02,
    })
    if missing:
        frame["meds_hbp"] = frame["meds_hbp"].astype(object)
        for col, share in (("bmi", 0.08), ("fat_total", 0.05), ("meds_hbp", 0.3), ("gender", 0.02)):
            frame.loc[rng.random(n) < share, col] = np.nan
    return frame


def ingest_frame(frame: pd.DataFrame, folder: Path) -> dict[str, str]:
    """Write ``frame`` as CSV, ingest it; returns the stage context's ``paths``."""
    from turbotab.core.datastore import ingest

    folder.mkdir(parents=True, exist_ok=True)
    source = folder / "table.csv"
    frame.to_csv(source, index=False)
    parquet = folder / "data" / "raw.parquet"
    ingest(source, parquet)
    return {"source": str(source), "data": str(parquet)}


def split_bundle(row_ids: Any, *, holdout: float = 0.2, folds: int = 5, seed: int = 0,
                 groups: Any = None, grouped_by: str | None = None) -> Bundle:
    """A split artifact as the contract shapes it; grouped when ``groups`` is given."""
    rng = np.random.default_rng(seed)
    ids = np.asarray(row_ids, dtype=np.int64)
    keys = np.asarray(groups) if groups is not None else ids
    units = np.unique(keys)
    rng.shuffle(units)
    n_hold = int(round(holdout * len(units)))
    held = set(units[:n_hold].tolist())
    train_units = units[n_hold:]
    fold_of = {u: i % folds for i, u in enumerate(train_units)}
    partition = np.array(["holdout" if k in held else "train" for k in keys.tolist()])
    fold = np.array([-1 if k in held else fold_of[k] for k in keys.tolist()])
    frame = pd.DataFrame({"row_id": ids, "partition": partition, "fold": fold})
    n_train = int((partition == "train").sum())
    return Bundle(
        data={"n_train": n_train, "n_holdout": int(len(ids) - n_train), "holdout": holdout,
              "seed": seed, "folds": folds, "grouped_by": grouped_by,
              "n_groups": int(len(units)) if groups is not None else None,
              "stratified": False, "note": "test split"},
        frames={"assignment": frame},
    )


def comparison_train_sets(split: Bundle, *, folds: int = 5, strata: Any = None,
                          groups: Any = None) -> set[frozenset]:
    """The training rows of every fold of the comparison substrate the fit stage draws for this
    split under prediction (MS6, ``models/folds.comparison_folds``): the split's own folds first,
    then the draws that make 10 repeats. Each set holds training rows only."""
    from turbotab.core.models.folds import comparison_folds

    a = split.frames["assignment"]
    train = a[a["partition"] == "train"]
    ids, own = train["row_id"].to_numpy(), train["fold"].to_numpy()
    columns, _, _ = comparison_folds([own], validation="kfold", scheme="random", n=len(ids),
                                     strata=strata, groups=groups, folds=folds,
                                     seed=int(split.data["seed"]))
    return {frozenset(ids[c != k].tolist()) for c in columns for k in np.unique(c)}


def cohort_bundle(row_ids: Any, predictors: list[str]) -> Bundle:
    ids = np.asarray(row_ids, dtype=np.int64)
    return Bundle(data={"steps": [], "n_final": int(len(ids)), "predictors": predictors},
                  frames={"rows": pd.DataFrame({"row_id": ids})})


def target_info(task: str, column: str = "glucose") -> dict[str, Any]:
    return {"column": column, "task": task, "detected_task": task, "confidence": "high",
            "reason": "", "histogram": None, "classes": None}


# The NHANES export's (and :func:`nhanes_like`'s) whole-valued predictors, as the table's author
# knows them (BLUEPRINT §14.3: whole numbers settle nothing by their values, so the fixture declares
# its truth): age in whole years, HDL and triglycerides in whole mg/dL, all amounts.
NHANES_TRUTH = {"code_or_count:age": "amount", "code_or_count:hdl": "amount",
                "code_or_count:triglycerides": "amount", "code_or_count:kcal": "amount"}


def state(**slots: Any) -> ProjectState:
    base: dict[str, Any] = {
        "lens": ["dietary"], "target": "glucose", "task": None, "purpose": "prediction",
        "roles": dict(NHANES_ROLES), "missing": "complete_case",
        "split": SplitSpec(holdout=0.2, seed=0, folds=5),
        "models": ["linear", "elastic_net", "boosted_trees"],
        "shape_confirmations": dict(NHANES_TRUTH),
    }
    base.update(slots)
    return ProjectState(**base)


def grams(*columns: str) -> dict[str, Any]:
    """The recorded units of energy sources a fixture's generator writes in grams: its declared
    truth, recorded as the user's answer (BLUEPRINT §14.3: a name's ``_g`` never settles the kcal
    each unit carries; the user's recorded unit, or the Atwater identity, does)."""
    from turbotab.core.decisions import ColumnUnitSpec

    return {c: ColumnUnitSpec(unit="g", days=None) for c in columns}


def energy(method: str = "residual", nutrients: list[str] | None = None, **kw: Any) -> EnergyAdjustment:
    if method == "none":
        return EnergyAdjustment(method="none")
    return EnergyAdjustment(method=method, energy_column="kcal",
                            nutrients=nutrients or ["protein", "carb", "fat_total"], **kw)


def substitution(donor: str = "fat_total", recipient: str = "carb", step: float = 100.0) -> SubstitutionSpec:
    return SubstitutionSpec(donor=donor, recipient=recipient, step_kcal=step)


def context(st: ProjectState, inputs: dict[str, Any], paths: dict[str, str],
            progress: list | None = None, cancelled: Any = None) -> StageContext:
    return StageContext(
        project_id="test", state=st, inputs=inputs, paths=paths,
        settings={"memory_budget_bytes": 1 << 30},
        on_progress=(lambda f, m: progress.append((f, m))) if progress is not None else None,
        is_cancelled=cancelled,
    )

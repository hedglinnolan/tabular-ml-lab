"""The NHANES fixture the previews' latency is measured on (``test_previews_4_latency.py``).

The fixture is the real NHANES export the project is driven on (``_tt_tmp_nhanes.csv``: 21,849
participants, 29 columns, NHANES 2001–2018; untracked, found by ``stage_harness._nhanes``). It has
no survey design columns, no follow-up, no batch, no multi-item scale, no second recall and one row
per person, so the previews that need those read columns added to its rows from a seeded stream of
their own (``augment``): every real column and every real row stays as it is, and each added column
is named for what it stands in for. The repeated-measures tables are the export's people seen
several times (``visits``, ``recalls``), their real values carried and the repeats' own drawn.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from turbotab.core import decisions as d
from turbotab.core.decisions import ProjectState

NUTRIENTS = ["protein", "sugar", "carb", "fat_total", "fat_sat", "fat_mon", "fat_poly"]
SCALE = [f"sat_{j}" for j in range(1, 9)]


def source() -> Path | None:
    from turbotab.core.tests.stage_harness import NHANES

    return NHANES if NHANES.is_file() else None


def augment(frame: pd.DataFrame, seed: int = 2026) -> pd.DataFrame:
    """The export with the columns its previews need, drawn from their own seeded stream."""
    rng = np.random.default_rng(seed)
    n = len(frame)
    out = frame.copy()
    strata = rng.integers(0, 15, n)
    out["SDMVSTRA"] = strata + 119
    out["SDMVPSU"] = rng.integers(1, 3, n)
    out["WTDRD1"] = np.round(rng.uniform(4000, 90000, n), 1)
    out["site"] = [f"S{k:02d}" for k in rng.integers(0, 40, n)]
    out["batch"] = rng.choice(["B1", "B2", "B3"], n)
    age = pd.to_numeric(out["age"], errors="coerce").fillna(45).to_numpy()
    out["supplement"] = (rng.random(n) < expit(-1 + 0.03 * (age - 45))).astype(int)
    trait = rng.standard_normal(n)
    for j, c in enumerate(SCALE):
        out[c] = np.clip(np.round(3 + 0.9 * trait + rng.normal(0, 0.8, n)), 1, 5).astype(int)
    protein = pd.to_numeric(out["protein"], errors="coerce").to_numpy()
    out["protein_d2"] = np.round(protein * np.exp(rng.normal(0, 0.35, n)), 2)
    hazard = 0.02 * np.exp(0.04 * (age - 50))
    event_time = rng.exponential(1 / hazard)
    follow = rng.uniform(1, 15, n)
    out["followup_years"] = np.round(np.minimum(event_time, follow), 3)
    out["cvd_event"] = (event_time <= follow).astype(int)
    return out


def visits(frame: pd.DataFrame, k: int = 3, seed: int = 7) -> pd.DataFrame:
    """Each participant seen up to ``k`` times, until an event: a supplement taken or not at each
    visit (prompted by the visit's blood pressure, which the previous visit's supplement lowers),
    systolic pressure at each visit, and the event, after which the participant is not seen."""
    rng = np.random.default_rng(seed)
    n = len(frame)
    age = pd.to_numeric(frame["age"], errors="coerce").fillna(45).to_numpy()
    female = (frame["gender"] == "female").astype(int).to_numpy()
    rows = []
    prev = np.zeros(n)
    alive = np.ones(n, dtype=bool)
    sbp = pd.to_numeric(frame["bp_sys"], errors="coerce").fillna(120).to_numpy()
    for t in range(k):
        sbp = sbp + rng.normal(0, 5, n) - 4 * prev
        a = (rng.random(n) < expit(-1 + 0.03 * (sbp - 120) + 1.2 * prev)).astype(int)
        y = (rng.random(n) < expit(-4 + 0.02 * (age - 50))).astype(int)
        rows.append(pd.DataFrame({"SEQN": frame["SEQN"].to_numpy(), "visit": t + 1, "supp": a,
                                  "sbp": np.round(sbp, 1), "female": female, "age": age,
                                  "event": y})[alive])
        alive &= y == 0
        prev = a.astype(float)
    return pd.concat(rows).sort_values(["SEQN", "visit"], kind="mergesort").reset_index(drop=True)


def recalls(frame: pd.DataFrame, seed: int = 11) -> pd.DataFrame:
    """Two 24-hour recalls per participant around the export's own intakes (day-to-day noise)."""
    rng = np.random.default_rng(seed)
    base = frame[["SEQN", "age", "gender", "kcal", "protein", "glucose"]].dropna()
    rows = []
    for day in (1, 2):
        noise = np.exp(rng.normal(0, 0.3, len(base)))
        rows.append(base.assign(recall=day, kcal=np.round(base["kcal"] * noise, 1),
                                protein=np.round(base["protein"] * noise
                                                 * np.exp(rng.normal(0, 0.2, len(base))), 2)))
    return pd.concat(rows).sort_values(["SEQN", "recall"], kind="mergesort").reset_index(drop=True)


ROLES = {"SEQN": "identifier", "cycle_begin_year": "time", "age": "covariate",
         "gender": "covariate", "bp_sys": "covariate", "bp_di": "covariate",
         "weight": "covariate", "height": "covariate", "bmi": "covariate", "waist": "covariate",
         "kcal": "energy", **{c: "exposure" for c in NUTRIENTS}, "hdl": "covariate",
         "triglycerides": "covariate", "meds_hbp": "excluded", "meds_chol": "excluded",
         "imputed_weight": "flag", "imputed_height": "flag", "imputed_bmi": "flag",
         "imputed_waist": "flag", "imputed_bp_sys": "flag", "imputed_bp_di": "flag",
         "SDMVSTRA": "design", "SDMVPSU": "design", "WTDRD1": "design", "site": "cluster",
         "batch": "excluded", "supplement": "covariate",
         **{c: "excluded" for c in SCALE}, "protein_d2": "excluded",
         "followup_years": "excluded", "cvd_event": "excluded"}
# BLUEPRINT §14.3: whole numbers settle nothing by their values; the export's author knows these
# are amounts (``modeling_fixtures.NHANES_TRUTH`` and the rest of its whole-valued columns), the
# supplement a code, the scale's answers amounts on their response scale.
SHAPES = {**{f"code_or_count:{c}": "amount"
             for c in ("age", "hdl", "triglycerides", "kcal", "glucose", "bp_sys", "bp_di",
                       "weight", "height", "waist", "cycle_begin_year", *SCALE)},
          "code_or_count:supplement": "code"}


def inference_state(**update) -> ProjectState:
    """Glucose against the diet, declared: protein the exposure, its confounders answered."""
    conf = dict(causes_exposure="yes", causes_outcome="yes", after_exposure="no")
    covariates = ["age", "gender", "bmi", "waist", "bp_sys", "bp_di", "weight", "height", "hdl",
                  "triglycerides", "supplement", *[c for c in NUTRIENTS if c != "protein"]]
    state = ProjectState(
        lens=["dietary"], target="glucose", task="regression", purpose="inference",
        roles=dict(ROLES), role_confirmations=dict(ROLES),
        grain=d.GrainSpec(grain="one_row_per_unit", id_column="SEQN"),
        shape_confirmations=dict(SHAPES),
        column_units={"kcal": d.ColumnUnitSpec(unit="kcal", days=1)},
        exclusions=[], missing=d.MissingSpec(strategy="complete_case"),
        split=d.SplitSpec(holdout=0.0, seed=0, folds=5), models=["linear"],
        survey=d.SurveySpec(estimand="sample"),
        estimand=d.EstimandSpec(exposure="protein", measure="mean_difference"),
        adjustment={c: d.AdjustmentAnswer(exposure="protein", **conf) for c in covariates})
    return state.model_copy(update=update)


__all__ = ["NUTRIENTS", "ROLES", "SCALE", "SHAPES", "augment", "inference_state", "recalls",
           "source", "visits"]

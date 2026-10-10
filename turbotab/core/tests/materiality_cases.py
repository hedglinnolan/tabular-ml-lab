"""The dietary NHANES proof of SURFACING_POLICY §9 ("The smallest end-to-end proof"), and the
calibration cases it gives ``materiality_calibration.json``.

The state is the ``dietary-inference`` capture's answers up to Models
(``docs/turbotab-next/review-packets/captures/dietary-inference.json``: the dietary lens, glucose
the outcome, sugar the exposure at fixed energy, the Goldberg screen at PAL 1.55 on one recall
day, the capture's adjustment answers, the standard energy model) with two simplifications that
keep the refit cheap and checkable by hand: complete cases in place of multiple imputation, and
every nutrient entering as a straight line. The every-row analysis is declared beside the primary
(Banna et al. 2017). The real stage graph runs in-process (``graph_runner.GraphRun``) on the
tracked NHANES fixture.

Regenerate the calibration file::

    venv/bin/python -m turbotab.core.tests.materiality_cases --write
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

from turbotab.core import decisions as d
from turbotab.core import materiality as M
from turbotab.core.decisions import ProjectState

JOURNEY = "dietary-inference"
NUTRIENTS = ["protein", "sugar", "carb", "fat_total", "fat_sat", "fat_mon", "fat_poly"]
ROLES = {"SEQN": "identifier", "cycle_begin_year": "covariate", "age": "covariate",
         "gender": "covariate", "bp_sys": "covariate", "bp_di": "covariate",
         "weight": "covariate", "height": "covariate", "bmi": "covariate", "waist": "covariate",
         "kcal": "energy", **{c: "exposure" for c in NUTRIENTS}, "hdl": "covariate",
         "triglycerides": "covariate", "meds_hbp": "covariate", "meds_chol": "covariate",
         "imputed_weight": "flag", "imputed_height": "flag", "imputed_bmi": "flag",
         "imputed_waist": "flag", "imputed_bp_sys": "flag", "imputed_bp_di": "flag"}
SHAPES = {f"code_or_count:{c}": "amount"
          for c in ("age", "hdl", "triglycerides", "kcal", "glucose", "bp_sys", "bp_di", "weight",
                    "height", "waist", "cycle_begin_year")}
# The capture's set_adjustment answers (seq 13–17).
CONFOUNDER = dict(causes_exposure="yes", causes_outcome="yes", after_exposure="no")
UNKNOWN_CAUSE = dict(causes_exposure="unknown", causes_outcome="unknown", after_exposure="no")
UNKNOWN_TIMING = dict(causes_exposure="unknown", causes_outcome="yes", after_exposure="unknown")
MEDIATOR = dict(causes_exposure="no", causes_outcome="yes", after_exposure="yes")
ADJUSTMENT = {**{c: CONFOUNDER for c in ("age", "gender", "cycle_begin_year")},
              **{c: UNKNOWN_CAUSE for c in NUTRIENTS if c != "sugar"},
              **{c: UNKNOWN_TIMING for c in ("weight", "height", "bmi", "waist")},
              **{c: MEDIATOR for c in ("bp_sys", "bp_di", "hdl", "triglycerides", "meds_hbp",
                                       "meds_chol")}}
GOLDBERG = d.GoldbergRule(
    column="kcal", energy_unit="kcal", days=1, sex="gender", female=["female"], male=["male"],
    age="age", weight="weight", weight_unit="kg", height="height", height_unit="cm",
    equation="schofield_height", pal=1.55, reason="implausible energy reports")
EVERY_ROW = d.SensitivityAnalysis(label="Every row", rules=[])


def proof_state(**update: Any) -> ProjectState:
    """The capture's answers up to Models, the plan not locked."""
    state = ProjectState(
        lens=["dietary"], target="glucose", task="regression", purpose="inference",
        roles=dict(ROLES), role_confirmations=dict(ROLES), shape_confirmations=dict(SHAPES),
        grain=d.GrainSpec(grain="one_row_per_unit", id_column="SEQN"),
        column_units={"kcal": d.ColumnUnitSpec(unit="kcal", days=1)},
        exclusions=[GOLDBERG], missing=d.MissingSpec(strategy="complete_case"),
        split=d.SplitSpec(holdout=0.0, seed=0, folds=5), models=["linear"],
        survey=d.SurveySpec(estimand="sample"),
        estimand=d.EstimandSpec(exposure="sugar", measure="mean_difference",
                                contrast="substitution"),
        adjustment={c: d.AdjustmentAnswer(exposure="sugar", **a) for c, a in ADJUSTMENT.items()},
        energy_adjustment=d.EnergyAdjustment(method="standard", energy_column="kcal",
                                             nutrients=list(NUTRIENTS)),
        sensitivity=[EVERY_ROW])
    return state.model_copy(update=update)


def nhanes() -> Path:
    from turbotab.core.tests.stage_harness import NHANES

    return NHANES


class Proof:
    """The proof's table, run through the real stage graph."""

    def __init__(self, folder: Path | None = None):
        from turbotab.core.tests.graph_runner import GraphRun

        self.run = GraphRun(nhanes(), folder or Path(tempfile.mkdtemp(prefix="materiality_")))

    def store(self) -> Any:
        from turbotab.core.datastore import DataStore

        return DataStore(self.run.raw, 2 << 30)

    def noticings(self, state: ProjectState) -> list[M.Noticing]:
        """The noticings as the engine measures them before the lock (outcome-blind)."""
        return M.noticings_for(state, self.store(), self.run.info)

    def artifacts(self, state: ProjectState) -> dict[str, Any]:
        """The stages' public artifacts for ``state`` (after the lock, the sensitivity stage's)."""
        out = self.run.run(state, upto=["sensitivity"])
        return self.run.public(out)

    def close(self) -> None:
        self.run.close()


def dietary_cases() -> tuple[list[M.Case], M.Ledger]:
    """The proof run once: the noticings before the lock, the sensitivity stage after it, and the
    calibration cases the ledger gives."""
    proof = Proof()
    try:
        state = proof_state()
        noticed = proof.noticings(state)
        locked = state.model_copy(update={"plan_locked": True})
        book = M.ledger(locked, noticed, proof.artifacts(locked))
    finally:
        proof.close()
    return M.cases_from(JOURNEY, book.rows), book


def build() -> dict[str, Any]:
    cases, _book = dietary_cases()
    return M.calibrate(cases, M.calibration())


if __name__ == "__main__":  # pragma: no cover - the regeneration command
    data = build()
    if "--write" in sys.argv:
        M.write_calibration(data)
        print(f"wrote {M.CALIBRATION_FILE}")
    else:
        import json

        print(json.dumps(data, indent=2, ensure_ascii=False))

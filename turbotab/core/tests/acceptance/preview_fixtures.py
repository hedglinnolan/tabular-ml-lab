"""The reference fixtures the consequence previews are held to (``test_previews_*.py``).

Each builder returns a :class:`preview_harness.Project`: a table, written as the researcher's CSV
(or the CDC's own XPT files), run through the real stage graph in-process for a recorded state, so
a preview is planned exactly as the server plans one and the answer can then be recorded and the
graph run again. The tables are the packages' own reference fixtures where one exists (ESTIMAND's
cohort, MS8's scale, the NCI package's NHANES-shaped recalls, the causal lane's cohort, TIMEVARY's
feedback cohort, WP12c's repeated recalls, WP12b's staggered-entry cohort, DATAIN's cut of the
public NHANES 2017–2018 files), each generated from a seeded truth by its author.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from turbotab.core import decisions as d
from turbotab.core.decisions import ProjectState
from turbotab.core.tests.acceptance.preview_harness import Project

A = d.AdjustmentAnswer
CONFOUNDER = dict(causes_exposure="yes", causes_outcome="yes", after_exposure="no")


# ── the declared plan: ESTIMAND's cohort with a site and a consequence of the exposure ─────────

def estimand_table(n: int = 600, seed: int = 7) -> pd.DataFrame:
    """ESTIMAND's cohort (``estimand_fixtures.cohort``) with two columns added from their own
    stream, so the cohort's columns are unchanged: ``site`` (24 recruiting sites) and ``hscrp``,
    which the fiber intake lowers and nothing else in the table causes (a consequence of the
    exposure)."""
    from turbotab.core.tests.acceptance import estimand_fixtures as ef

    frame = ef.cohort(n, seed=seed)
    rng = np.random.default_rng(seed + 100)
    frame["site"] = [f"S{k:02d}" for k in rng.integers(0, 24, len(frame))]
    frame["hscrp"] = np.round(np.exp(1.0 - 0.03 * (frame["fiber"] - 20) + rng.normal(0, 0.4, len(frame))), 3)
    return frame


ESTIMAND_ROLES = {"pid": "identifier", "age": "covariate", "sex": "covariate", "smoking": "covariate",
                  "activity": "covariate", "bmi": "covariate", "hscrp": "covariate",
                  "fiber": "exposure", "supplement": "exposure", "site": "cluster"}
# The answers for fiber: the generator's truth (estimand_fixtures), hscrp answered a consequence.
ESTIMAND_ANSWERS = {
    "age": CONFOUNDER, "sex": CONFOUNDER, "smoking": CONFOUNDER,
    "activity": dict(causes_exposure="no", causes_outcome="yes", after_exposure="no"),
    "bmi": dict(causes_exposure="unknown", causes_outcome="yes", after_exposure="unknown"),
    "hscrp": dict(causes_exposure="no", causes_outcome="no", after_exposure="yes"),
    "supplement": dict(causes_exposure="no", causes_outcome="yes", after_exposure="no"),
}
AMOUNTS = {"code_or_count:smoking": "amount", "code_or_count:activity": "amount",
           "code_or_count:supplement": "amount", "code_or_count:age": "amount"}


def estimand_state(**update: Any) -> ProjectState:
    state = ProjectState(
        lens=["clinical"], target="glucose", task="regression", purpose="inference",
        roles=dict(ESTIMAND_ROLES), role_confirmations=dict(ESTIMAND_ROLES),
        grain=d.GrainSpec(grain="one_row_per_unit", id_column="pid"),
        shape_confirmations=dict(AMOUNTS), exclusions=[],
        missing=d.MissingSpec(strategy="complete_case"),
        split=d.SplitSpec(holdout=0.0, seed=3, folds=5), models=["linear"],
        estimand=d.EstimandSpec(exposure="fiber", measure="mean_difference"),
        adjustment={c: A(exposure="fiber", **a) for c, a in ESTIMAND_ANSWERS.items()},
        model_sequence=d.ModelSequenceSpec(exposure="fiber", model_1=["age", "sex"]),
        clusters=d.ClusterSpec(column="site", adjust="cluster_only"))
    return state.model_copy(update=update)


def estimand_project(folder: Path, **update: Any) -> Project:
    return Project(estimand_table(), folder, estimand_state(**update),
                   upto=["design", "fit", "effects", "cohort", "split"])


def family_state() -> ProjectState:
    """The same cohort declared as an exposure family (fiber, supplement): each exposure's effect
    in its own model, every member shown."""
    roles = {**ESTIMAND_ROLES, "hscrp": "excluded", "site": "excluded"}
    answers = {c: A(exposure=d.EXPOSURE_FAMILY, **a) for c, a in ESTIMAND_ANSWERS.items()
               if c in ("age", "sex", "smoking", "activity")}
    return estimand_state(roles=roles, role_confirmations=dict(roles), clusters=None,
                          estimand=d.EstimandSpec(family=True, measure="mean_difference",
                                                  multiplicity="count_stated"),
                          adjustment=answers, model_sequence=None)


# ── the NCI package's NHANES-shaped recalls, with its survey design ─────────────────────────────

def survey_project(folder: Path) -> Project:
    from turbotab.core.tests.acceptance import test_nci_usual_intake as T

    state = ProjectState(
        lens=["dietary"], target="LBXTC", task="regression", purpose="inference",
        roles=dict(T.NHANES_ROLES), role_confirmations=dict(T.NHANES_ROLES),
        grain=d.GrainSpec(grain="one_row_per_unit", id_column="SEQN"),
        shape_confirmations=dict(T.NHANES_TRUTH), exclusions=[],
        survey=d.SurveySpec(estimand="population", weight="WTDRD1", strata="SDMVSTRA",
                            psu="SDMVPSU"),
        estimand=d.EstimandSpec(exposure="DR1TPROT", measure="mean_difference"),
        adjustment={c: A(exposure="DR1TPROT", **CONFOUNDER) for c in ("RIAGENDR", "RIDAGEYR")},
        missing=d.MissingSpec(strategy="complete_case"),
        split=d.SplitSpec(holdout=0.0, seed=0, folds=5), models=["linear"])
    return Project(T.nhanes_table(seed=21, n=900), folder, state,
                   upto=["usual_intake", "cohort", "split", "design", "fit"])


# ── MS8's reflective scale ───────────────────────────────────────────────────────────────────

def scales_project(folder: Path) -> Project:
    from turbotab.core.tests.acceptance import scales_fixtures as sf

    roles = {"pid": "identifier", "age": "covariate", "bmi": "covariate",
             **{c: "covariate" for c in sf.SAT}, "sat_ref": "excluded"}
    state = ProjectState(
        lens=["survey"], target="sbp", task="regression", purpose="inference",
        roles=roles, role_confirmations=dict(roles),
        grain=d.GrainSpec(grain="one_row_per_unit", id_column="pid"),
        missing=d.MissingSpec(strategy="complete_case"), exclusions=[],
        split=d.SplitSpec(holdout=0.0, seed=0, folds=5), models=["linear"],
        shape_confirmations={**{f"code_or_count:{c}": "amount" for c in sf.SAT},
                             "code_or_count:age": "amount"})
    return Project(sf.linear_scale_table(seed=11, n=600), folder, state,
                   upto=["design", "cohort", "split"])


# ── the causal lane's cohort ──────────────────────────────────────────────────────────────────

def causal_project(folder: Path) -> Project:
    from turbotab.core.tests.acceptance import test_causal_lane as T

    folder.mkdir(parents=True, exist_ok=True)
    frame = T.cohort(folder / "cohort.csv", n=800)
    answers = {"age": CONFOUNDER, "smoker": CONFOUNDER, "income": CONFOUNDER,
               "fiber": dict(causes_exposure="no", causes_outcome="yes", after_exposure="no"),
               "ldl": dict(causes_exposure="no", causes_outcome="yes", after_exposure="yes")}
    state = ProjectState(
        lens=["clinical"], target="sbp", task="regression", purpose="inference",
        roles=dict(T.CHAIN_ROLES), role_confirmations=dict(T.CHAIN_ROLES),
        grain=d.GrainSpec(grain="one_row_per_unit", id_column="person_id"),
        shape_confirmations={"code_or_count:smoker": "code", "code_or_count:heavy_user": "code",
                             "code_or_count:age": "amount"},
        reading_confirmations={"cluster:person_id": "no"},
        exclusions=[], missing=d.MissingSpec(strategy="complete_case"),
        split=d.SplitSpec(holdout=0.0, seed=0, folds=5), models=["linear"],
        estimand=d.EstimandSpec(exposure="heavy_user", measure="mean_difference"),
        adjustment={c: A(exposure="heavy_user", **a) for c, a in answers.items()})
    return Project(frame, folder / "project", state,
                   upto=["causal_design", "cohort", "split", "design"])


CAUSAL_ASSUMPTIONS = ["no_unmeasured_confounding", "positivity", "consistency", "time_ordering"]


# ── TIMEVARY's cohort with treatment–confounder feedback ─────────────────────────────────────

def timevary_project(folder: Path) -> Project:
    from turbotab.core.tests.acceptance import test_timevary as T
    from turbotab.core.tests.acceptance.timevary_fixtures import feedback_cohort

    state = T.feedback_state(time_varying=T.MSM_LANE, models=["linear"])
    return Project(feedback_cohort(n=800, visits=5), folder, state,
                   upto=["time_varying", "cohort", "split"])


# ── WP12c's repeated recalls ─────────────────────────────────────────────────────────────────

def calibration_project(folder: Path) -> Project:
    from turbotab.core.tests.acceptance import test_wp12c_calibration as T

    state = ProjectState(
        lens=["dietary"], target="ldl", task="regression", purpose="inference",
        roles=dict(T.ROLES), role_confirmations=dict(T.ROLES),
        grain=d.GrainSpec(grain="repeated", id_column="participant_id"),
        repeat_kind=d.RepeatSpec(repeat_kind="repeats"), unit="unit",
        aggregation=d.AggregationSpec(method="mean"), temporal=d.TemporalSpec(temporal=False),
        shape_confirmations={"code_or_count:age": "amount"},
        column_units={"energy_kcal": d.ColumnUnitSpec(unit="kcal", days=1)},
        exclusions=[], missing=d.MissingSpec(strategy="complete_case"),
        split=d.SplitSpec(holdout=0.0, seed=0, folds=5), models=["linear"],
        energy_adjustment=d.EnergyAdjustment(method="residual", energy_column="energy_kcal",
                                             nutrients=["protein_g"]))
    return Project(T.recall_table(n=500), folder, state, upto=["design", "cohort", "split"])


# ── an assay with three batches (prediction) ─────────────────────────────────────────────────

def batch_table(n: int = 300, p: int = 12, seed: int = 3) -> pd.DataFrame:
    """``p`` features measured in three batches, each batch shifting every feature by its own
    offset times a feature-specific factor (a location batch effect, as ComBat models it); the
    outcome depends on the first two features."""
    rng = np.random.default_rng(seed)
    batch = rng.choice(["B1", "B2", "B3"], n)
    shift = {"B1": 0.0, "B2": 1.5, "B3": -1.0}
    X = (rng.normal(0, 1, (n, p))
         + np.array([shift[b] for b in batch])[:, None] * rng.uniform(0.5, 1.5, p))
    frame = pd.DataFrame(X.round(4), columns=[f"m{j:02d}" for j in range(p)])
    frame.insert(0, "batch", batch)
    frame.insert(0, "sample_id", np.arange(n))
    frame["y"] = (X[:, 0] - 0.5 * X[:, 1] + rng.normal(0, 1, n)).round(4)
    return frame


def batch_project(folder: Path) -> Project:
    frame = batch_table()
    roles = {"sample_id": "identifier", "batch": "excluded",
             **{c: "exposure" for c in frame.columns if c.startswith("m")}}
    state = ProjectState(
        lens=["metabolomics"], target="y", task="regression", purpose="prediction",
        roles=roles, role_confirmations=dict(roles),
        grain=d.GrainSpec(grain="one_row_per_unit", id_column="sample_id"),
        exclusions=[], missing=d.MissingSpec(strategy="complete_case"),
        split=d.SplitSpec(holdout=0.2, seed=0, folds=5), models=["linear"])
    return Project(frame, folder, state, upto=["design", "cohort", "split", "findings"])


# ── WP12b's staggered-entry cohort ───────────────────────────────────────────────────────────

FOLLOW_ROLES = {"participant_id": "identifier", "fiber_g": "exposure", "age": "covariate",
                "followup_years": "time"}


def follow_up_project(folder: Path) -> Project:
    from turbotab.core.tests.acceptance.test_wp12b_cox_mixed_gee import _staggered_entry_cohort

    state = ProjectState(
        lens=["clinical"], target="cvd_event", task="time_to_event", event="1",
        purpose="inference", missing=d.MissingSpec(strategy="complete_case"),
        roles=dict(FOLLOW_ROLES), role_confirmations=dict(FOLLOW_ROLES), exclusions=[],
        split=d.SplitSpec(holdout=0.0, seed=0, folds=5), models=["cox"],
        grain=d.GrainSpec(grain="one_row_per_unit", id_column="participant_id"),
        follow_up=d.FollowUpSpec(time_column="followup_years"))
    return Project(_staggered_entry_cohort(), folder, state, upto=["cohort", "split"])


# ── a dietary table: a skewed outcome, total energy in kcal ──────────────────────────────────

def dietary_table(n: int = 800, seed: int = 5) -> pd.DataFrame:
    """MODELING_FIXTURES' NHANES-shaped table (``nhanes_like``) with a positive, right-skewed
    outcome added from its own stream: triglycerides, lognormal (skewness well above 2)."""
    from turbotab.core.tests import modeling_fixtures as mf

    frame = mf.nhanes_like(n, seed=seed)
    rng = np.random.default_rng(seed + 50)
    frame["tg"] = np.round(np.exp(rng.normal(4.7, 0.65, len(frame))), 1)
    return frame.drop(columns=["triglycerides"])


def dietary_project(folder: Path, target: str = "tg") -> Project:
    from turbotab.core.tests import modeling_fixtures as mf

    roles = {k: v for k, v in mf.NHANES_ROLES.items() if k != "triglycerides"}
    state = ProjectState(
        lens=["dietary"], target=target, task="regression", purpose="prediction",
        roles=roles, role_confirmations=dict(roles),
        grain=d.GrainSpec(grain="one_row_per_unit", id_column="SEQN"),
        shape_confirmations={"code_or_count:age": "amount",
                             "code_or_count:cycle_begin_year": "amount"})
    return Project(dietary_table(), folder, state,
                   upto=["target_info", "proposals", "roles", "findings"])


# ── DATAIN's cut of the public NHANES 2017–2018 files ─────────────────────────────────────────

def datain_project(folder: Path) -> tuple[Project, dict[str, str]]:
    """DEMO_J as the table, DR1TOT_J and BMX_J added to the project (``files/<id>/``, as the
    server keeps an added file), nothing joined yet."""
    from turbotab.core.datastore import ingest
    from turbotab.core.tests.acceptance.test_datain import unpack

    folder.mkdir(parents=True, exist_ok=True)
    state = ProjectState(lens=["dietary"])
    project = Project(unpack(folder, "DEMO_J"), folder, state, upto=["working", "findings"])
    files = {}
    for i, name in enumerate(("DR1TOT_J", "BMX_J")):
        fid = f"f{i:010x}"
        where = project.run.folder / "files" / fid
        where.mkdir(parents=True)
        info = ingest(unpack(folder, name), where / "raw.parquet")
        (where / "file.json").write_text(json.dumps({"id": fid, "name": f"{name}.XPT",
                                                     "n_rows": info.n_rows}))
        files[name] = fid
    project.project_dir = project.run.folder
    return project, files


__all__ = ["CAUSAL_ASSUMPTIONS", "ESTIMAND_ANSWERS", "ESTIMAND_ROLES", "batch_project",
           "batch_table", "calibration_project", "causal_project", "datain_project",
           "dietary_project", "dietary_table", "estimand_project", "estimand_state",
           "estimand_table", "family_state", "follow_up_project", "scales_project",
           "survey_project", "timevary_project"]

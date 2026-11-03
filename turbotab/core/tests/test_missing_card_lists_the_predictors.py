"""The missing-values card lists the predictors the other answers keep (INBOX:1027).

``proposals.missing.columns`` (the card's list and its count) took every column holding a predictor
role with a blank, so on the plasma panel's QC-RLSC plans it listed 36 columns, 16 of them features
the QC filters removed: the working table keeps a removed feature's column, and the repair takes it
out of the predictors (``repairs.unusable_columns``). The card now leaves out what
``decisions.left_out`` leaves out but the missing-values answer's own columns: a repair's unusable
columns and, under inference, the covariates the adjustment answers leave out. The missing-values
answer's columns stay listed: they are this card's own offer (``leave_out``), and that answer's
slot is not among the stage's reads.

The truth is pandas: the blanks on the rows with the outcome measured, among the columns built to
hold a predictor role, less the columns the recorded answers name. On the plasma panel the removed
features are named by the QC filters' published rules read with pandas (detected in fewer than 70%
of the pooled QCs; in a batch, fewer than five detected QCs or a detected value outside them).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from turbotab.core.decisions import EstimandSpec, MissingSpec, ProjectState
from turbotab.core.methods import qc_drift as Q
from turbotab.core.stages.proposals import proposals_stage
from turbotab.core.tests.stage_harness import Ingested

REPO = Path(__file__).resolve().parents[3]
PLASMA = REPO / "research" / "swarm" / "fixtures" / "plasma_panel.csv"

# Twelve rows, the outcome blank on the last two. `f1` and `f2` are the features a QC-RLSC answer
# removed (detection, uncorrectable); `x1` is the covariate the adjustment question is asked about.
N = 12
BLANK = {"y": [10, 11], "x1": [0, 1, 2, 10], "f1": [0, 1, 2, 3], "f2": [4], "f3": [5, 6, 11]}
ROLES = {"x1": "covariate", "x2": "covariate", "f1": "exposure", "f2": "exposure",
         "f3": "exposure", "run_order": "time", "sample_type": "design"}
PREDICTORS = ["x1", "x2", "f1", "f2", "f3"]


def _table(tmp_path: Path) -> tuple[Ingested, pd.DataFrame]:
    df = pd.DataFrame({"y": [2.0 + 1.5 * i for i in range(N)], "sample_type": "study",
                       "run_order": range(1, N + 1), "x1": [i % 3 + 0.5 for i in range(N)],
                       "x2": [2.0 * i for i in range(N)], "f1": [10.0 + i for i in range(N)],
                       "f2": [20.0 + i for i in range(N)], "f3": [30.0 + 0.7 * i for i in range(N)]})
    for column, rows in BLANK.items():
        df.loc[rows, column] = None
    source = tmp_path / "t.csv"
    df.to_csv(source, index=False)
    return Ingested(source, tmp_path / "p"), pd.read_csv(source)


def blanks(df: pd.DataFrame, columns: list[str], outcome: str) -> list[tuple[str, int, float]]:
    """Each column with a blank on the rows with the outcome measured: its count and share, the
    blankest first (then by name), as pandas counts them."""
    base = df[outcome].notna()
    n = int(base.sum())
    out = [(c, int(df.loc[base, c].isna().sum())) for c in columns]
    return sorted(((c, k, round(k / n, 4)) for c, k in out if k), key=lambda e: (-e[2], e[0]))


def listed(artifact) -> list[tuple[str, int, float]]:
    data = getattr(artifact, "data", artifact)
    return [(e["column"], e["n_missing"], e["share"]) for e in data["missing"]["columns"]]


def qc_answer(option: str = "qc_rlsc_lc") -> dict:
    """A recorded pooled-QC answer: QC-RLSC removing `f1` (detection) and `f2` (uncorrectable), or
    the exclusion of the QC rows, which removes no column."""
    params = {"column": "sample_type", "qc_levels": ["pooled_qc"], "order_column": "run_order",
              "batch_column": None, "features": ["f1", "f2", "f3"], "n_qc": 6,
              "platform": "lc_ms", "d_ratio_max": None,
              "dropped": {"detection": ["f1"], "uncorrectable": ["f2"]}, "kept": 1}
    if option == "exclude_rows":
        params = {"column": "sample_type", "levels": ["pooled_qc"]}
    return {Q.FINDING: {"action": "applied", "option": option, "params": params}}


def test_the_card_lists_no_feature_the_qc_filters_removed(tmp_path):
    """Fires: QC-RLSC's answer takes `f1` and `f2` out of the predictors, so the card lists the
    blanks of `x1` and `f3` only (3 and 2 of the 10 rows with `y` measured)."""
    table, df = _table(tmp_path)
    state = ProjectState(lens=["metabolomics"], target="y", roles=ROLES, findings=qc_answer())
    got = listed(table.run(proposals_stage, state))
    want = blanks(df, [c for c in PREDICTORS if c not in ("f1", "f2")], "y")
    assert want == [("x1", 3, 0.3), ("f3", 2, 0.2)]
    assert got == want


@pytest.mark.parametrize("findings", [None, "exclude_rows"], ids=["no answer", "QC rows excluded"])
def test_with_no_column_removed_the_card_lists_every_predictor_with_a_blank(tmp_path, findings):
    """Silent: with no pooled-QC answer, or one that excludes the QC rows and removes no column,
    every predictor with a blank is listed, `f1` and `f2` among them."""
    table, df = _table(tmp_path)
    state = ProjectState(lens=["metabolomics"], target="y", roles=ROLES,
                         findings=qc_answer(findings) if findings else None)
    want = blanks(df, PREDICTORS, "y")
    assert [c for c, _, _ in want] == ["f1", "x1", "f3", "f2"]
    assert listed(table.run(proposals_stage, state)) == want


def test_the_missing_values_answers_own_columns_stay_on_its_card(tmp_path):
    """Silent on the card's own answer: a missing-values answer that leaves `x1` out keeps it
    listed, as the card offers it (``decisions.left_out`` holds it; the card does not)."""
    from turbotab.core.decisions import left_out

    table, df = _table(tmp_path)
    state = ProjectState(lens=["metabolomics"], target="y", roles=ROLES,
                         missing=MissingSpec(strategy="complete_case", drop_columns=["x1"]))
    assert left_out(state) == ["x1"]
    assert listed(table.run(proposals_stage, state)) == blanks(df, PREDICTORS, "y")


ADJUSTMENT = {"x1": {"exposure": "f3", "causes_exposure": "no", "causes_outcome": "yes",
                     "after_exposure": "yes"}}


@pytest.mark.parametrize("purpose", ["inference", "prediction"])
def test_under_inference_a_covariate_the_adjustment_leaves_out_is_not_listed(tmp_path, purpose):
    """Fires under inference: `x1` answered as changed by `f3` and a cause of `y` is a mediator, which
    a total effect leaves out (``estimand.adjustment_left_out``), so its blanks drop no row and it
    is not listed. Silent under prediction, where the same answers leave nothing out."""
    from turbotab.core.estimand import adjustment_left_out

    table, df = _table(tmp_path)
    state = ProjectState(lens=["metabolomics"], target="y", roles=ROLES, purpose=purpose,
                         estimand=EstimandSpec(exposure="f3", measure="mean_difference"),
                         adjustment=ADJUSTMENT)
    gone = ["x1"] if purpose == "inference" else []
    assert adjustment_left_out(state) == gone
    want = blanks(df, [c for c in PREDICTORS if c not in gone], "y")
    assert listed(table.run(proposals_stage, state)) == want


# ── the plasma panel through QC-RLSC, through the real stage graph ───────────────────────────────

PLASMA_BASE = {"lens": ["metabolomics"], "orientation": "sample_major", "target": "disease",
               "grain": {"grain": "one_row_per_unit"}}


def removed_by_the_qc_rules(df: pd.DataFrame) -> set[str]:
    """The features the LC-MS QC filters remove, by their published rules read with pandas: detected
    (a value above zero) in fewer than 70% of the pooled QCs (Broadhurst et al. 2018), or, in some
    batch, detected in fewer than five QCs or detected outside the injection-order span of its
    detected QCs (its curve would be extrapolated; ``qc_drift.qc_rlsc``)."""
    qc = df["sample_type"].eq("pooled_qc")
    features = [c for c in df.columns if c.startswith("mz_")]
    detected = df[features].notna() & (df[features] > 0)
    out = {c for c in features if detected.loc[qc, c].mean() < 0.7}
    for c in features:
        if c in out:
            continue
        for _, batch in df.groupby("batch"):
            seen = detected.loc[batch.index, c]
            order = batch.loc[qc.loc[batch.index] & seen, "run_order"]
            outside = (batch["run_order"] < order.min()) | (batch["run_order"] > order.max())
            if len(order) < 5 or bool((outside & seen).any()):
                out.add(c)
                break
    return out


@pytest.mark.slow
def test_the_plasma_card_lists_the_blanks_of_the_features_qc_rlsc_keeps(tmp_path):
    """INBOX:1027's case: the plasma panel's pooled QCs answered by QC-RLSC (LC-MS filters, per
    batch). Every `mz_` feature is an exposure and age, sex and BMI covariates; the card lists the
    blanks pandas counts on the 100 participants (the QC rows leave) among the features the QC
    rules keep: 20 of the 36 with a blank, where the base listed all 36."""
    from turbotab.core.tests.graph_runner import GraphRun

    df = pd.read_csv(PLASMA)
    features = [c for c in df.columns if c.startswith("mz_")]
    roles = {**dict.fromkeys(features, "exposure"), **dict.fromkeys(["age", "sex", "bmi"], "covariate"),
             "sample_id": "identifier", "sample_type": "design", "run_order": "time",
             "batch": "design"}
    run = GraphRun(PLASMA, tmp_path / "p")
    try:
        first = run.run(ProjectState.model_validate({**PLASMA_BASE, "roles": roles}),
                        upto=["findings"])
        found = {f["id"]: f for f in getattr(first["findings"], "data", first["findings"])["findings"]}
        offer = next(o for o in found[Q.FINDING]["repairs"] if o["key"] == "qc_rlsc_lc")
        params = offer["decision"]["params"]
        removed = removed_by_the_qc_rules(df)
        assert len(removed) == 16
        assert removed == {c for gone in params["dropped"].values() for c in gone}
        state = ProjectState.model_validate({**PLASMA_BASE, "roles": roles, "findings": {
            Q.FINDING: {"action": "applied", "option": "qc_rlsc_lc", "params": params}}})
        out = run.run(state, upto=["proposals"])
    finally:
        run.close()
    working = {c["name"] for c in getattr(out["working"], "data", out["working"])["columns"]}
    assert removed <= working  # the working table keeps a removed feature's column
    participants = df[df["sample_type"].ne("pooled_qc")]
    every = blanks(participants, [*features, "age", "sex", "bmi"], "disease")
    want = blanks(participants, [c for c in [*features, "age", "sex", "bmi"] if c not in removed],
                  "disease")
    assert (len(every), len(want)) == (36, 20)
    assert listed(out["proposals"]) == want

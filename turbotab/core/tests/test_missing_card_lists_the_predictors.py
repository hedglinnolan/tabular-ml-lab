"""The missing-values card lists the predictors the other answers keep (INBOX:1027).

``proposals.missing.columns`` (the card's list and its count) took every column holding a predictor
role with a blank, so on the plasma panel's QC-RLSC plans it listed 36 columns, 16 of them features
the QC filters removed: the working table keeps a removed feature's column, and the repair takes it
out of the predictors (``repairs.unusable_columns``). The card now leaves out the columns the
answers other than the missing-values one leave out: a repair's unusable columns and, under
inference, the covariates the adjustment answers leave out of the primary model, but not those a
model declared beside it holds, whose blanks cost that model rows: the "further adjusted for" model
(``estimand.secondary_columns``) and a declared modification's (the modifier, and an interaction's
confounders of its second exposure: ``interaction.adjusted_sets``). The missing-values answer's
columns stay listed: they are this card's own offer (``leave_out``).

The truth is pandas: the blanks on the rows with the outcome measured, among the columns built to
hold a predictor role, less the columns the recorded answers name. On the plasma panel the removed
features are named by the QC filters' published rules read with pandas (detected in fewer than 70%
of the pooled QCs; in a batch, fewer than five detected QCs or a detected value outside them).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from turbotab.core.decisions import EstimandSpec, MissingSpec, ModificationSpec, ProjectState
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
    a total effect leaves out of the primary model (``estimand.adjustment_left_out``), and no model
    declared beside it holds (not among ``estimand.secondary_columns``; no modification declared),
    so its blanks drop no row and it is not listed. Silent under prediction, where the same answers
    leave nothing out."""
    from turbotab.core.estimand import adjustment_left_out, secondary_columns

    table, df = _table(tmp_path)
    state = ProjectState(lens=["metabolomics"], target="y", roles=ROLES, purpose=purpose,
                         estimand=EstimandSpec(exposure="f3", measure="mean_difference"),
                         adjustment=ADJUSTMENT)
    gone = ["x1"] if purpose == "inference" else []
    assert adjustment_left_out(state) == gone
    assert secondary_columns(state) == []
    want = blanks(df, [c for c in PREDICTORS if c not in gone], "y")
    assert listed(table.run(proposals_stage, state)) == want


# `x1`'s answers that leave it out of the primary model but put it in the declared "further adjusted
# for" model beside it (``estimand.secondary_columns``; the secondary stage fits it there, and its
# blanks cost that model rows or are imputed in it).
FURTHER = {
    "timing unknown": {"exposure": "f3", "causes_exposure": "no", "causes_outcome": "yes",
                       "after_exposure": "unknown"},
    "mediator, further adjusted for": {**ADJUSTMENT["x1"], "further": True},
}


@pytest.mark.parametrize("answers", list(FURTHER.values()), ids=list(FURTHER))
def test_a_covariate_the_further_adjusted_model_holds_stays_listed(tmp_path, answers):
    """Silent on the secondary model's columns: a covariate of unknown timing, and a mediator the
    user asks to be further adjusted for, leave the primary model (``adjustment_left_out``) but
    enter the "further adjusted for" one, so `x1` stays listed with its 3 blanks."""
    from turbotab.core.estimand import adjustment_left_out, secondary_columns

    table, df = _table(tmp_path)
    state = ProjectState(lens=["metabolomics"], target="y", roles=ROLES, purpose="inference",
                         estimand=EstimandSpec(exposure="f3", measure="mean_difference"),
                         adjustment={"x1": answers})
    assert adjustment_left_out(state) == secondary_columns(state) == ["x1"]
    want = blanks(df, PREDICTORS, "y")
    assert ("x1", 3, 0.3) in want
    assert listed(table.run(proposals_stage, state)) == want


def test_a_missing_answer_that_takes_the_exposure_out_lists_the_mediator(tmp_path):
    """The card reads the missing-values answer through the adjustment answers: one that leaves the
    exposure `f3` out ends the estimand (``estimand.current_estimand``), so no answer leaves the
    mediator `x1` out and the fresh card lists it, beside `f3`, that answer's own column."""
    table, df = _table(tmp_path)
    state = ProjectState(lens=["metabolomics"], target="y", roles=ROLES, purpose="inference",
                         estimand=EstimandSpec(exposure="f3", measure="mean_difference"),
                         adjustment=ADJUSTMENT,
                         missing=MissingSpec(strategy="complete_case", drop_columns=["f3"]))
    want = blanks(df, PREDICTORS, "y")
    assert ("x1", 3, 0.3) in want and ("f3", 2, 0.2) in want
    assert listed(table.run(proposals_stage, state)) == want


# `x1` answered as a cause of neither `f3` nor `y`: the primary model leaves it out and the "further
# adjusted for" model does not hold it. A declared modification's model (``methods.interaction``)
# can: the modifier is one of its predictors, as is each confounder an interaction's answers for its
# second exposure add (``interaction.adjusted_sets``), so their blanks cost that model rows.
NEITHER = {"exposure": "f3", "causes_exposure": "no", "causes_outcome": "no", "after_exposure": "no"}
CONFOUNDER = {"causes_exposure": "yes", "causes_outcome": "yes", "after_exposure": "no"}
MODIFIED = {
    "x1 the modifier": {"x1": ModificationSpec(kind="effect_modification", exposure="f3")},
    "x1 a confounder of the second exposure": {"f1": ModificationSpec(
        kind="interaction", exposure="f3",
        answers={c: CONFOUNDER for c in ["x1", "x2", "f2"]})},
}


def modified_state(modifications: dict | None) -> ProjectState:
    return ProjectState(lens=["metabolomics"], target="y", roles=ROLES, purpose="inference",
                        estimand=EstimandSpec(exposure="f3", measure="mean_difference"),
                        adjustment={"x1": NEITHER}, modifications=modifications)


@pytest.mark.parametrize("modifications", list(MODIFIED.values()), ids=list(MODIFIED))
def test_a_covariate_a_declared_modification_holds_stays_listed(tmp_path, modifications):
    """Silent on a declared modification's columns: `x1`, which the adjustment answers leave out of
    the primary model, is the declared modifier, or a confounder of the interaction's second
    exposure `f1`, so the modification's model holds it and `x1` stays listed with its 3 blanks."""
    from turbotab.core.estimand import adjustment_left_out, secondary_columns
    from turbotab.core.methods.interaction import adjusted_sets, declared

    table, df = _table(tmp_path)
    state = modified_state(modifications)
    assert (adjustment_left_out(state), secondary_columns(state)) == (["x1"], [])
    found = declared(state)
    assert list(found) == list(modifications)
    assert any(m == "x1" or "x1" in adjusted_sets(state, m, s)[1] for m, s in found.items())
    want = blanks(df, PREDICTORS, "y")
    assert want == [("f1", 4, 0.4), ("x1", 3, 0.3), ("f3", 2, 0.2), ("f2", 1, 0.1)]
    assert listed(table.run(proposals_stage, state)) == want


NOT_HELD = {
    "none declared": None,
    "x1 withdrawn": {"x1": ModificationSpec(kind="effect_modification", exposure="f3",
                                            withdrawn=True, post_hoc=True)},
    "x1 no confounder of the second exposure": {"f1": ModificationSpec(
        kind="interaction", exposure="f3",
        answers={"x1": {"causes_exposure": "no", "causes_outcome": "no", "after_exposure": "no"},
                 "x2": CONFOUNDER, "f2": CONFOUNDER})},
}


@pytest.mark.parametrize("modifications", list(NOT_HELD.values()), ids=list(NOT_HELD))
def test_a_covariate_no_declared_modification_holds_is_not_listed(tmp_path, modifications):
    """Fires: with no modification declared, one withdrawn (no longer estimated), or an interaction
    whose answers make `x1` a cause of neither its second exposure nor `y`, no model holds `x1`, so
    it is not listed."""
    from turbotab.core.methods.interaction import adjusted_sets, declared

    table, df = _table(tmp_path)
    state = modified_state(modifications)
    assert not any(m == "x1" or "x1" in adjusted_sets(state, m, s)[1]
                   for m, s in declared(state).items())
    want = blanks(df, [c for c in PREDICTORS if c != "x1"], "y")
    assert listed(table.run(proposals_stage, state)) == want


@pytest.mark.parametrize("declared_f1", [True, False], ids=["f1 the modifier", "none declared"])
def test_a_removed_feature_a_declared_modification_holds_stays_listed(tmp_path, declared_f1):
    """Both ways on a repair's columns: QC-RLSC removes `f1` and `f2`, and under inference `f1`
    declared as the modifier is a predictor of the modification's model (which reads its column
    from the working table), so `f1` stays listed with its 4 blanks; with none declared it is not
    listed. `f2` is listed in neither."""
    table, df = _table(tmp_path)
    state = ProjectState(lens=["metabolomics"], target="y", roles=ROLES, purpose="inference",
                         estimand=EstimandSpec(exposure="f3", measure="mean_difference"),
                         findings=qc_answer(), modifications={"f1": ModificationSpec(
                             kind="effect_modification", exposure="f3")} if declared_f1 else None)
    kept = ["x1", "x2", "f3", *(["f1"] if declared_f1 else [])]
    want = blanks(df, kept, "y")
    assert want == ([("f1", 4, 0.4)] if declared_f1 else []) + [("x1", 3, 0.3), ("f3", 2, 0.2)]
    assert listed(table.run(proposals_stage, state)) == want


@pytest.mark.xfail(strict=True, reason="INBOX:1027 repair V2-1: the integrator adds 'modifications' "
                                       "to the proposals Stage's reads (with its version bump); "
                                       "then drop this marker")
def test_declaring_a_modification_rekeys_the_card():
    """So the stage's key changes when a modification that holds `x1` is declared (the test above:
    its fresh card lists `x1`), or the card cached before it, which omits `x1`, is served."""
    from turbotab.core.graph import stage_keys
    from turbotab.core.stages import build_graph
    from turbotab.core.stages.proposals import answered_out

    order = build_graph().order()
    plain, modified = modified_state(None), modified_state(MODIFIED["x1 the modifier"])
    assert (answered_out(plain), answered_out(modified)) == (["x1"], [])

    def key(state: ProjectState) -> str | None:
        return stage_keys(order, state.model_dump(mode="json"), "fingerprint")["proposals"]

    assert key(plain) is not None
    assert key(modified) != key(plain)


@pytest.mark.xfail(strict=True, reason="INBOX:1027 repair P2: the integrator adds 'missing' to the "
                                       "proposals Stage's reads (with its version bump); then drop "
                                       "this marker")
def test_a_missing_answer_that_takes_the_exposure_out_rekeys_the_card():
    """So the stage's key changes with a missing-values answer that leaves the exposure out (the
    test above: its fresh card lists `x1`), or the card cached without that answer, which omits
    `x1`, is served."""
    from turbotab.core.estimand import adjustment_left_out
    from turbotab.core.graph import stage_keys
    from turbotab.core.stages import build_graph

    order = build_graph().order()
    answers = dict(lens=["metabolomics"], target="y", roles=ROLES, purpose="inference",
                   estimand=EstimandSpec(exposure="f3", measure="mean_difference"),
                   adjustment=ADJUSTMENT)
    plain = ProjectState(**answers)
    dropped = ProjectState(**answers, missing=MissingSpec(strategy="complete_case",
                                                          drop_columns=["f3"]))
    assert (adjustment_left_out(plain), adjustment_left_out(dropped)) == (["x1"], [])

    def key(state: ProjectState) -> str | None:
        return stage_keys(order, state.model_dump(mode="json"), "fingerprint")["proposals"]

    assert key(plain) is not None
    assert key(dropped) != key(plain)


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

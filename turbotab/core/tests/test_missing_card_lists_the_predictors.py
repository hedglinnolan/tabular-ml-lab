"""The missing-values card lists every predictor a model of the plan holds, and no column a repair
removed from every model (INBOX:1027).

``proposals.missing.columns`` (the card's list and its count) took every column holding a predictor
role with a blank, so on the plasma panel's QC-RLSC plans it listed 36 columns, 16 of them features
the QC filters removed: the working table keeps a removed feature's column, and the repair takes it
out of the predictors (``repairs.unusable_columns``). The card now leaves out a repair's unusable
columns, less each a declared modification's model holds (the modifier, and an interaction's
confounders of its second exposure: ``interaction.adjusted_sets``), whose blanks cost that model
rows. It leaves out nothing else. A covariate the adjustment answers leave out of the primary model
stays listed: another model of the plan can hold it (the "further adjusted for" model, a declared
modification's, the time-varying lane's and, after the lock, the specification curve's), so its
blanks need the missing-data answer. The missing-values answer's columns stay listed: they are this
card's own offer (``leave_out``).

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


# `x1`'s answers that leave it out of the primary model (``estimand.adjustment_left_out``) and out of
# the "further adjusted for" model: a mediator, which a total effect leaves out, and a confounder the
# exposure affects, which the time-varying lane adjusts for (``time_varying.affected_confounders``).
LEFT_OUT = {
    "a mediator": (ADJUSTMENT["x1"], []),
    "a confounder the exposure affects": ({"exposure": "f3", "causes_exposure": "yes",
                                           "causes_outcome": "yes", "after_exposure": "yes"},
                                          ["x1"]),
}


@pytest.mark.parametrize("answers, affected", list(LEFT_OUT.values()), ids=list(LEFT_OUT))
def test_a_covariate_the_adjustment_leaves_out_stays_listed_beside_no_removed_feature(
        tmp_path, answers, affected):
    """Both ways under inference: QC-RLSC removes `f1` and `f2`, and the adjustment answers leave
    `x1` out of the primary model, as a mediator or as a confounder the exposure affects. Another
    model of the plan can hold `x1` (the time-varying lane's, which adjusts for the second; after
    the lock, the specification curve's departure that adds a covariate the answers left out), so
    its blanks need the missing-data answer: the card lists `x1` with its 3 blanks and `f3` with
    its 2 of the 10 rows with `y` measured, and neither removed feature."""
    from turbotab.core.estimand import adjustment_left_out, secondary_columns
    from turbotab.core.time_varying import affected_confounders

    table, df = _table(tmp_path)
    state = ProjectState(lens=["metabolomics"], target="y", roles=ROLES, purpose="inference",
                         estimand=EstimandSpec(exposure="f3", measure="mean_difference"),
                         adjustment={"x1": answers}, findings=qc_answer())
    assert (adjustment_left_out(state), secondary_columns(state)) == (["x1"], [])
    assert affected_confounders(state) == affected
    want = blanks(df, ["x1", "x2", "f3"], "y")
    assert want == [("x1", 3, 0.3), ("f3", 2, 0.2)]
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


# `x1` answered as a cause of neither `f3` nor `y`: the primary model leaves it out and the "further
# adjusted for" model does not hold it. A declared modification's model (``methods.interaction``)
# can, or not: the modifier is one of its predictors, as is each confounder an interaction's answers
# for its second exposure, here `x2`, add (``interaction.adjusted_sets``). The QC-RLSC answer
# removes `f1` and `f2`, so the interaction's answers are asked for `x1` alone.
NEITHER = {"exposure": "f3", "causes_exposure": "no", "causes_outcome": "no", "after_exposure": "no"}
CONFOUNDER = {"causes_exposure": "yes", "causes_outcome": "yes", "after_exposure": "no"}
NO_CAUSE = {"causes_exposure": "no", "causes_outcome": "no", "after_exposure": "no"}
MODIFICATIONS = {
    "none declared": None,
    "x1 the modifier": {"x1": ModificationSpec(kind="effect_modification", exposure="f3")},
    "x1 withdrawn": {"x1": ModificationSpec(kind="effect_modification", exposure="f3",
                                            withdrawn=True, post_hoc=True)},
    "x1 a confounder of the second exposure": {"x2": ModificationSpec(
        kind="interaction", exposure="f3", answers={"x1": CONFOUNDER})},
    "x1 no confounder of the second exposure": {"x2": ModificationSpec(
        kind="interaction", exposure="f3", answers={"x1": NO_CAUSE})},
}


@pytest.mark.parametrize("modifications", list(MODIFICATIONS.values()), ids=list(MODIFICATIONS))
def test_a_covariate_the_adjustment_leaves_out_stays_listed_whatever_the_modifications(
        tmp_path, modifications):
    """Both ways, whatever modification is declared: `x1`, which the adjustment answers leave out of
    the primary model, stays listed with its 3 blanks, whether a modification's model holds it or
    none does (after the lock the specification curve's departure that adds it can); the removed
    `f1` and `f2`, which no model holds, are not listed."""
    from turbotab.core.estimand import adjustment_left_out, secondary_columns
    from turbotab.core.methods.interaction import adjusted_sets, declared

    table, df = _table(tmp_path)
    state = ProjectState(lens=["metabolomics"], target="y", roles=ROLES, purpose="inference",
                         estimand=EstimandSpec(exposure="f3", measure="mean_difference"),
                         adjustment={"x1": NEITHER}, modifications=modifications,
                         findings=qc_answer())
    assert (adjustment_left_out(state), secondary_columns(state)) == (["x1"], [])
    held = {c for m, s in declared(state).items() for c in [m, *adjusted_sets(state, m, s)[1]]}
    assert not held & {"f1", "f2"}
    want = blanks(df, ["x1", "x2", "f3"], "y")
    assert want == [("x1", 3, 0.3), ("f3", 2, 0.2)]
    assert listed(table.run(proposals_stage, state)) == want


def removed_state(declared_f1: bool) -> ProjectState:
    """Under inference, QC-RLSC's answer removing `f1` and `f2`, and `f1` declared the modifier of
    `f3`'s effect or no modification declared."""
    return ProjectState(lens=["metabolomics"], target="y", roles=ROLES, purpose="inference",
                        estimand=EstimandSpec(exposure="f3", measure="mean_difference"),
                        findings=qc_answer(), modifications={"f1": ModificationSpec(
                            kind="effect_modification", exposure="f3")} if declared_f1 else None)


@pytest.mark.parametrize("declared_f1", [True, False], ids=["f1 the modifier", "none declared"])
def test_a_removed_feature_a_declared_modification_holds_stays_listed(tmp_path, declared_f1):
    """Both ways on a repair's columns: QC-RLSC removes `f1` and `f2`, and under inference `f1`
    declared as the modifier is a predictor of the modification's model (which reads its column
    from the working table), so `f1` stays listed with its 4 blanks; with none declared it is not
    listed. `f2` is listed in neither."""
    table, df = _table(tmp_path)
    state = removed_state(declared_f1)
    kept = ["x1", "x2", "f3", *(["f1"] if declared_f1 else [])]
    want = blanks(df, kept, "y")
    assert want == ([("f1", 4, 0.4)] if declared_f1 else []) + [("x1", 3, 0.3), ("f3", 2, 0.2)]
    assert listed(table.run(proposals_stage, state)) == want


# `x2` declared the second exposure of an interaction with `f3`. Once QC-RLSC removed `f1` and `f2`
# its answers are asked for `x1` alone; they also hold `f1`, answered before that removal, as a
# confounder of `x2` (the interaction's model adjusts for it: ``interaction.adjusted_sets``) or as a
# cause of neither.
SECOND = {"f1 a confounder of x2": (CONFOUNDER, ["f1"]), "f1 no cause of x2": (NO_CAUSE, [])}


@pytest.mark.parametrize("answer, adds", list(SECOND.values()), ids=list(SECOND))
def test_a_removed_feature_an_interaction_adjusts_for_stays_listed(tmp_path, answer, adds):
    """Both ways on the second exposure's confounders: QC-RLSC removes `f1` and `f2`, and `x2` is
    declared the second exposure of an interaction with `f3`. When its answers make `f1` a
    confounder of `x2`, the interaction's model adjusts for `f1`, reading its column from the
    working table, so `f1` stays listed with its 4 blanks of the 10 rows with `y` measured; when
    they make it a cause of neither, no model holds it and it is not listed. `f2` is listed in
    neither."""
    from turbotab.core.methods.interaction import adjusted_sets, declared, missing_answers

    table, df = _table(tmp_path)
    spec = ModificationSpec(kind="interaction", exposure="f3",
                            answers={"x1": NO_CAUSE, "f1": answer})
    state = ProjectState(lens=["metabolomics"], target="y", roles=ROLES, purpose="inference",
                         estimand=EstimandSpec(exposure="f3", measure="mean_difference"),
                         modifications={"x2": spec}, findings=qc_answer())
    assert (list(declared(state)), missing_answers(state, "x2", spec)) == (["x2"], [])
    assert adjusted_sets(state, "x2", spec)[1] == adds
    want = blanks(df, ["x1", "x2", "f3", *adds], "y")
    assert want == ([("f1", 4, 0.4)] if adds else []) + [("x1", 3, 0.3), ("f3", 2, 0.2)]
    assert listed(table.run(proposals_stage, state)) == want


@pytest.mark.xfail(strict=True, reason="INBOX:1027 repair V2-1: the integrator adds 'modifications' "
                                       "to the proposals Stage's reads (with its version bump); "
                                       "then drop this marker")
def test_declaring_a_removed_feature_the_modifier_rekeys_the_card():
    """So the stage's key changes when the removed `f1` is declared the modifier (the test above:
    its fresh card lists `f1`), or the card cached before it, which omits `f1`, is served."""
    from turbotab.core.graph import stage_keys
    from turbotab.core.stages import build_graph
    from turbotab.core.stages.proposals import answered_out

    order = build_graph().order()
    plain, modified = removed_state(False), removed_state(True)
    assert ("f1" in answered_out(plain), "f1" in answered_out(modified)) == (True, False)

    def key(state: ProjectState) -> str | None:
        return stage_keys(order, state.model_dump(mode="json"), "fingerprint")["proposals"]

    assert key(plain) is not None
    assert key(modified) != key(plain)


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

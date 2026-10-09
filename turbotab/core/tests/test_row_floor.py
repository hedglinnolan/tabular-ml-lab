"""Rows enough to analyze (``turbotab.core.row_floor``): the metabolomics zero-row crash (INBOX).

``metabolomics_untargeted.csv`` with complete cases: once the features' roles are settled, every
participant is blank in at least one feature, and the design stage failed on scikit-learn's "Found
array with 0 sample(s)". An answer that would leave fewer rows than the design needs is refused
when it is recorded, naming which predictors' blanks remove the rows and offering a fill; the
cohort stage says the same of a table that changed under recorded answers; no stage downstream is
handed an empty frame. The blank counts are checked against pandas reading the CSV directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbotab.core.decisions import (
    ExclusionRule, MissingSpec, ProjectState, Refusal, SetExclusions, fold_onto, validate,
)
from turbotab.core.row_floor import FEWEST_ROWS
from turbotab.core.tests.stage_harness import SAMPLES, Ingested

METABOLOMICS = SAMPLES / "metabolomics_untargeted.csv"


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return pd.read_csv(METABOLOMICS)


@pytest.fixture(scope="module")
def table(tmp_path_factory) -> Ingested:
    return Ingested(METABOLOMICS, tmp_path_factory.mktemp("metabolomics"))


@pytest.fixture(scope="module")
def store(table):
    with table.store() as s:
        yield s


@pytest.fixture(scope="module")
def findings(frame):
    """The findings stage's words for the table under the metabolomics lens (the left-censored
    features among them), as the validators read them from the findings artifact."""
    from turbotab import packs
    from turbotab.core.stages.findings import speak_for

    spoken = speak_for(frame, ["metabolomics"], "responder", [],
                       packs.findings(frame, ["metabolomics"]))
    return {"findings": [f for _, f in spoken]}


def features(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith("mz_")]


def roles(frame: pd.DataFrame) -> dict[str, str]:
    return {"sample_id": "identifier", "sample_type": "excluded", "run_order": "excluded",
            "batch": "excluded", "age": "covariate", "sex": "covariate", "bmi": "covariate",
            **{c: "exposure" for c in features(frame)}}


def state(frame: pd.DataFrame, **slots) -> ProjectState:
    base = dict(lens=["metabolomics"], target="responder", task="binary", event="1",
                purpose="prediction", grain={"grain": "one_row_per_unit"}, roles=roles(frame),
                missing=MissingSpec(strategy="complete_case"))
    base.update(slots)
    return ProjectState(**base)


def ranked_blanks(rows: pd.DataFrame, frame: pd.DataFrame) -> list[tuple[str, int]]:
    """The reference: each predictor blank in any of ``rows``, most blanks first, then in table
    order, read by pandas from the CSV."""
    preds = [c for c, r in roles(frame).items() if r in ("exposure", "covariate")]
    counts = rows[preds].isna().sum()
    return sorted(((c, int(k)) for c, k in counts.items() if k), key=lambda ck: (-ck[1], preds.index(ck[0])))


def named(ranked: list[tuple[str, int]]) -> str:
    (a, ka), (b, kb), (c, kc) = ranked[:3]
    return (f"each of them is blank in at least one of the {len(ranked)} predictors with blanks, "
            f"most often `{a}` (in {ka} of them), `{b}` ({kb}), `{c}` ({kc}) and "
            f"{len(ranked) - 3} more")


def ctx_for(st: ProjectState, store, frame, findings) -> dict:
    return {"state": st, "store": store, "columns": list(frame.columns), "target": st.target,
            "artifact": lambda name: findings if name == "findings" else None}


# ── through the stage harness: the reproduction ─────────────────────────────


def test_the_cohort_names_the_blanks_that_leave_no_row_instead_of_handing_the_design_none(
        table, frame):
    """The INBOX reproduction through the stage harness: the features' roles settled, complete
    cases. The cohort had 0 rows and the design failed on scikit-learn's "Found array with 0
    sample(s)"; now the cohort stage refuses, saying which predictors' blanks remove which rows,
    and that filling them is the way back."""
    from turbotab.core.stages.rows import cohort_stage

    measured = frame[frame["responder"].notna()]
    ranked = ranked_blanks(measured, frame)
    preds = [c for c, r in roles(frame).items() if r in ("exposure", "covariate")]
    assert int(measured[preds].notna().all(axis=1).sum()) == 0  # the reference: no row is complete
    with pytest.raises(ValueError) as refused:
        table.run(cohort_stage, state(frame))
    said = str(refused.value)
    assert said == (
        f"The answers recorded leave none of the {len(frame)} rows in the table to analyze, and the "
        f"design needs at least {FEWEST_ROWS}. “`responder` recorded” removes "
        f"{len(frame) - len(measured)} rows; complete cases remove {len(measured)} rows: "
        f"{named(ranked)}. Fill the blanks instead of complete cases (the missing-values question).")
    # Filled, every measured row stays: the cohort computes.
    out = table.run(cohort_stage, state(frame, missing=MissingSpec(
        strategy="impute", below_detection="censoring_aware", censored_columns=features(frame))))
    assert out.data["n_final"] == len(measured)


def test_a_design_handed_no_rows_says_so_in_plain_words(tmp_path):
    """No stage is handed an empty frame now, but one that is says so: never scikit-learn's
    "Found array with 0 sample(s)"."""
    from turbotab.core.stages.modeling import design_stage
    from turbotab.core.tests import modeling_fixtures as mf

    paths = mf.ingest_frame(mf.nhanes_like(60, seed=1), tmp_path)
    empty = mf.split_bundle(np.array([], dtype=np.int64))
    with pytest.raises(ValueError, match=r"^There are no training rows to build the models on"):
        design_stage(mf.context(mf.state(models=["linear"]),
                                {"split": empty, "target_info": mf.target_info("regression")}, paths))


def test_explores_thirds_of_a_column_with_no_value_are_one_blank_group():
    """The explore stage's crash on the empty frame (``np.quantile`` of nothing): a column with no
    value among the rows is one ``(blank)`` group."""
    from turbotab.core.models.decision_curve import subgroup_labels

    assert subgroup_labels(np.array([], dtype=float), "thirds").tolist() == []
    assert subgroup_labels(np.array([np.nan, np.nan]), "thirds").tolist() == ["(blank)", "(blank)"]


# ── at the decision validator ────────────────────────────────────────────────


def test_confirming_the_features_under_complete_cases_is_refused_with_a_fill(
        frame, store, findings):
    """The INBOX's path: the features rode along unconfirmed when complete cases were recorded;
    confirming them is refused, naming the blanks, with the censoring-aware fill (the leash's own
    way for values below detection) as the first way back. Once it is recorded, the same
    confirmation is accepted."""
    now = state(frame, roles_unconfirmed=features(frame))
    confirm = {"kind": "confirm_readings",
               "items": [{"reading": "role", "column": c, "value": "exposure"} for c in features(frame)]}
    ctx = ctx_for(now, store, frame, findings)
    with pytest.raises(Refusal) as refused:
        validate(confirm, ctx)
    assert refused.value.code == "too_few_rows"
    measured = frame[frame["responder"].notna()]
    assert refused.value.message == (
        f"Recorded, this answer would leave none of the {len(measured)} rows analyzed now, and the "
        f"design needs at least {FEWEST_ROWS}. Complete cases would remove {len(measured)} rows: "
        f"{named(ranked_blanks(measured, frame))}. Fill the blanks instead of complete cases (the "
        f"missing-values question).")
    fill = refused.value.exits[0]
    assert fill["label"] == ("Fill the blanks instead: single fill in each training fold, values "
                             "below detection censoring-aware")
    assert fill["decision"]["strategy"] == "impute"
    assert fill["decision"]["below_detection"] == "censoring_aware"
    assert refused.value.exits[-1] == {"label": "Keep the answers as they are", "decision": None}
    validate(fill["decision"], ctx)  # the way back is one the record accepts
    filled = fold_onto(now, validate(fill["decision"], ctx))
    validate(confirm, ctx_for(filled, store, frame, findings))


def test_complete_cases_on_settled_features_are_refused_and_keep_the_columns_left_out(
        frame, store, findings):
    """Recorded once the features are settled, complete cases are refused at the missing-values
    question itself; the fill keeps the columns the answer leaves out. Leaving out every feature
    with a blank keeps the rows, and is accepted."""
    now = state(frame, missing=None)
    ctx = ctx_for(now, store, frame, findings)
    measured = frame[frame["responder"].notna()]
    gappy = [c for c in features(frame) if measured[c].isna().any()]
    with pytest.raises(Refusal) as refused:
        validate({"kind": "set_missing", "strategy": "complete_case", "drop_columns": ["mz_0001"]}, ctx)
    assert refused.value.code == "too_few_rows"
    assert refused.value.exits[0]["decision"]["drop_columns"] == ["mz_0001"]
    validate({"kind": "set_missing", "strategy": "complete_case", "drop_columns": gappy}, ctx)
    # Under inference the leash lets only multiple imputation through, below detection
    # censoring-aware; a way forward back to complete cases is never offered as a fill.
    inference = ctx_for(state(frame, missing=None, purpose="inference"), store, frame, findings)
    with pytest.raises(Refusal) as refused:
        validate({"kind": "set_missing", "strategy": "complete_case"}, inference)
    fills = [e["decision"] for e in refused.value.exits if e["decision"]]
    assert [(f["strategy"], f["below_detection"]) for f in fills] == [
        ("multiple_imputation", "censoring_aware")]


def test_a_rule_that_removes_every_row_is_refused_with_the_rule_dropped():
    """Not only complete cases: an eligibility rule that would leave too few rows is refused, its
    exit the answer without that rule. A rule that keeps enough is accepted."""
    from turbotab.core.datastore import DataStore, ingest
    import tempfile
    from pathlib import Path

    rng = np.random.default_rng(0)
    toy = pd.DataFrame({"y": rng.normal(size=40), "x": rng.normal(size=40),
                        "age": rng.integers(20, 80, 40).astype(float)})
    with tempfile.TemporaryDirectory() as folder:
        source, parquet = Path(folder) / "toy.csv", Path(folder) / "toy.parquet"
        toy.to_csv(source, index=False)
        ingest(source, parquet)
        with DataStore(parquet, 1 << 30) as store:
            now = ProjectState(target="y", task="regression", purpose="inference",
                               roles={"x": "exposure", "age": "covariate"},
                               missing=MissingSpec(strategy="complete_case"))
            ctx = {"state": now, "store": store, "columns": list(toy.columns), "target": "y"}
            keep = ExclusionRule(column="age", low=20, high=79, reason="adults")
            late = ExclusionRule(column="age", low=90, high=99, reason="the oldest")
            with pytest.raises(Refusal) as refused:
                validate(SetExclusions(rules=[keep, late]), ctx)
            assert refused.value.code == "too_few_rows"
            assert refused.value.message == (
                f"Recorded, this answer would leave none of the 40 rows analyzed now, and the design "
                f"needs at least {FEWEST_ROWS}. “`age` within `90`–`99`” would remove 40 rows. Drop "
                f"or widen the rule that removes them (the eligibility question).")
            dropped = refused.value.exits[0]
            assert dropped["label"] == "Drop the rule: `age` within `90`–`99`"
            assert dropped["decision"] == SetExclusions(rules=[keep]).model_dump(mode="json")
            validate(dropped["decision"], ctx)


def test_an_answer_that_removes_no_row_is_never_refused_for_the_rows(frame, store, findings):
    """The check counts only where an answer changes what the cohort reads, and refuses only an
    answer that leaves fewer rows than now: one that keeps or restores rows is accepted however few
    there are (a log recorded before the check, already at none)."""
    from turbotab.core.row_floor import fill_exits

    validate({"kind": "set_purpose", "purpose": "prediction"},
             ctx_for(state(frame, missing=None), store, frame, findings))
    zero = ctx_for(state(frame), store, frame, findings)
    fills = fill_exits(state(frame), zero)  # each accepted by ``validate`` on the way
    assert fills and all(f["decision"]["strategy"] == "impute" for f in fills)
    validate({"kind": "set_purpose", "purpose": "prediction"}, zero)


# ── the verifier's cases: every answer, every analysis, every exit ───────────


def toy_store(frame: pd.DataFrame, folder):
    """``frame`` ingested as a table, and what a context reads of it: the store, the oriented
    table a value repair is evaluated on, and the working table (a pass-through: identity rows)."""
    from types import SimpleNamespace

    from turbotab.core.datastore import DataStore, ingest

    source, parquet = folder / "toy.csv", folder / "toy.parquet"
    frame.to_csv(source, index=False)
    info = ingest(source, parquet).to_dict()
    tables = {"oriented": SimpleNamespace(data={"columns": info["columns"]},
                                          files={"table.parquet": parquet}),
              "working": SimpleNamespace(data={"row_map": "identity", "aggregation": None},
                                         files={"table.parquet": parquet})}
    return DataStore(parquet, 1 << 30), tables.get


def renal(n: int = 120, complete: int = 10, seed: int = 0) -> pd.DataFrame:
    """The verifier's renal table: twelve labs with blanks, so that complete cases keep the first
    ``complete`` rows; ``sbp`` holds the code 999 in six of them (and in three rows with blanks)."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"egfr_decline": np.tile([0, 1], n // 2),
                          "age_years": rng.integers(30, 80, n).astype(float),
                          "sbp": np.round(rng.normal(130, 15, n))})
    labs = [f"lab_{c}" for c in "abcdefghijkl"]
    for c in labs:
        frame[c] = np.round(rng.normal(5, 1, n), 2)
    for i in range(complete, n):
        for c in rng.choice(labs, int(rng.integers(1, 3)), replace=False):
            frame.loc[i, c] = np.nan
    frame.loc[[0, 1, 2, 3, 4, 5, 50, 60, 70], "sbp"] = 999.0
    return frame


def renal_state(**slots) -> ProjectState:
    base = dict(target="egfr_decline", task="binary", event="1", purpose="prediction",
                roles={"age_years": "covariate", "sbp": "covariate",
                       **{f"lab_{c}": "exposure" for c in "abcdefghijkl"}},
                missing=MissingSpec(strategy="complete_case"))
    base.update(slots)
    return ProjectState(**base)


def test_a_repair_that_blanks_values_under_complete_cases_is_refused_on_the_values_it_leaves(
        tmp_path):
    """The verifier's renal table: complete cases keep 10 rows, and `sbp` holds the code 999 in six
    of them. Treating 999 as missing was accepted, because the check read the working table as it
    stood, and the cohort then failed with 4 rows. It is refused now, counted on the values the
    repair would leave (the working stage's own SQL), with the counts pandas gives; after a fill,
    the same repair is accepted."""
    from turbotab.core.decisions import ApplyRepair

    frame = renal()
    store, bundle = toy_store(frame, tmp_path)
    repaired = frame.assign(sbp=frame["sbp"].mask(frame["sbp"] == 999))
    preds = list(renal_state().roles)
    complete = frame[preds].notna().all(axis=1)
    kept = repaired[preds].notna().all(axis=1)
    assert (int(complete.sum()), int(kept.sum())) == (10, 4)  # the reference
    with store:
        now = renal_state()
        ctx = {"state": now, "store": store, "columns": list(frame.columns),
               "target": "egfr_decline", "bundle": bundle}
        repair = ApplyRepair(finding_id="sentinel_missing__sbp", option="set_missing",
                             params={"values": {"sbp": [999.0]}})
        with pytest.raises(Refusal) as refused:
            validate(repair, ctx)
        assert refused.value.code == "too_few_rows"
        assert refused.value.message == (
            f"Recorded, this answer would leave 4 of the 10 rows analyzed now, and the design needs "
            f"at least {FEWEST_ROWS}. Complete cases would remove 6 rows: each of them is blank in "
            f"`sbp`. Fill the blanks instead of complete cases (the missing-values question).")
        fill = refused.value.exits[0]
        assert fill["decision"]["kind"] == "set_missing" and fill["decision"]["strategy"] == "impute"
        filled = fold_onto(now, validate(fill["decision"], ctx))
        validate(repair, {**ctx, "state": filled})
        # Without the tables a repair is evaluated on, the check reads the table as it stands.
        validate(repair, {k: v for k, v in ctx.items() if k != "bundle"})


def test_an_outcome_left_one_value_is_refused_where_it_had_two(tmp_path):
    """The verifier's renal_1class table: the 12 complete rows all have `egfr_decline` 0. Settling
    the labs' roles under complete cases was accepted, and the fit crashed with "IndexError: list
    index out of range". The answer is refused now, saying the outcome is left one value."""
    frame = renal(complete=12)
    frame.loc[:11, "egfr_decline"] = 0
    store, bundle = toy_store(frame, tmp_path)
    labs = [f"lab_{c}" for c in "abcdefghijkl"]
    with store:
        now = renal_state(roles_unconfirmed=labs)
        ctx = {"state": now, "store": store, "columns": list(frame.columns),
               "target": "egfr_decline", "bundle": bundle}
        confirm = {"kind": "confirm_readings",
                   "items": [{"reading": "role", "column": c, "value": "exposure"} for c in labs]}
        with pytest.raises(Refusal) as refused:
            validate(confirm, ctx)
    assert refused.value.code == "one_outcome_value"
    assert refused.value.message.startswith(
        "Recorded, this answer would leave 12 rows, and `egfr_decline` is `0` in every one of them: "
        "the models need rows with another value of the outcome to learn from. Complete cases "
        f"would remove {len(frame) - 12} rows: each of them is blank in at least one of the 12 "
        "predictors with blanks, most often ")
    assert refused.value.exits[0]["decision"]["strategy"] == "impute"


def test_a_sensitivity_analysis_that_keeps_too_few_rows_is_refused(tmp_path):
    """The verifier's survey case: a sensitivity analysis whose rules keep no row was accepted, and
    the sensitivity stage showed scikit-learn's "Found array with 0 sample(s)". Every analysis is
    counted now, its rows those its own rules keep: refused, with the analysis left out (or its
    rule dropped) as the exits; one that keeps enough is accepted."""
    from turbotab.core.decisions import SensitivityAnalysis, SetSensitivity

    toy = pd.DataFrame({"y": np.linspace(0, 1, 40) ** 2, "x": np.linspace(-1, 1, 40),
                        "age": 20 + 1.5 * np.arange(40)})
    store, bundle = toy_store(toy, tmp_path)
    with store:
        now = ProjectState(target="y", task="regression", purpose="inference",
                           roles={"x": "exposure", "age": "covariate"},
                           missing=MissingSpec(strategy="complete_case"))
        ctx = {"state": now, "store": store, "columns": list(toy.columns), "target": "y",
               "bundle": bundle}
        oldest = ExclusionRule(column="age", low=90, high=99, reason="the oldest")
        adults = ExclusionRule(column="age", low=30, high=79, reason="adults")
        answer = SetSensitivity(analyses=[SensitivityAnalysis(label="adults", rules=[adults]),
                                          SensitivityAnalysis(label="the oldest", rules=[oldest])])
        with pytest.raises(Refusal) as refused:
            validate(answer, ctx)
        assert refused.value.code == "too_few_rows"
        assert refused.value.message == (
            f"Recorded, this answer would leave the sensitivity analysis “the oldest” no rows, "
            f"against 40 in the primary analysis, and its fit needs at least {FEWEST_ROWS}. "
            f"“`age` within `90`–`99`” would remove 40 rows. Drop or widen the rule that removes "
            f"them (the sensitivity question).")
        labels = [e["label"] for e in refused.value.exits]
        # Dropping its one rule would leave it the primary analysis (no rule): not offered.
        assert labels == ["Leave out the analysis “the oldest”", "Keep the answers as they are"]
        for e in refused.value.exits[:-1]:
            validate(e["decision"], ctx)  # each way back is one the record accepts
        left = SetSensitivity.model_validate(refused.value.exits[0]["decision"])
        assert [a.label for a in left.analyses] == ["adults"]


def test_every_rule_exit_is_one_the_record_accepts_and_names_its_rule(tmp_path):
    """The verifier's rules [age 30–35, sbp 0–90]: the refusal offered "Drop the rule on
    `age_years`", which was itself refused (the sbp rule alone keeps no row). An exit is offered
    only when the record accepts it, so here the two rules are dropped together; and two rules on
    one column are two exits, each named by its own range."""
    toy = pd.DataFrame({"y": np.linspace(0, 1, 40), "x": np.linspace(-1, 1, 40),
                        "age": 20 + 1.5 * np.arange(40), "sbp": np.linspace(100, 160, 40)})
    store, bundle = toy_store(toy, tmp_path)
    with store:
        now = ProjectState(target="y", task="regression", purpose="inference",
                           roles={"x": "exposure", "age": "covariate", "sbp": "covariate"},
                           missing=MissingSpec(strategy="complete_case"))
        ctx = {"state": now, "store": store, "columns": list(toy.columns), "target": "y",
               "bundle": bundle}
        narrow = ExclusionRule(column="age", low=30, high=35, reason="a narrow band")
        low_sbp = ExclusionRule(column="sbp", low=0, high=90, reason="hypotension")
        assert int(((toy.age >= 30) & (toy.age <= 35)).sum()) == 4  # alone, too few too
        with pytest.raises(Refusal) as refused:
            validate(SetExclusions(rules=[narrow, low_sbp]), ctx)
        assert [e["label"] for e in refused.value.exits] == [
            "Drop the rules: `age` within `30`–`35` and `sbp` within `0`–`90`",
            "Keep the answers as they are"]
        validate(refused.value.exits[0]["decision"], ctx)

        older = ExclusionRule(column="age", low=60, reason="older")
        younger = ExclusionRule(column="age", high=30, reason="younger")
        assert (int((toy.age >= 60).sum()), int((toy.age <= 30).sum())) == (13, 7)
        with pytest.raises(Refusal) as refused:
            validate(SetExclusions(rules=[older, younger]), ctx)
        assert [e["label"] for e in refused.value.exits] == [
            "Drop the rule: `age` at least `60`", "Drop the rule: `age` at most `30`",
            "Keep the answers as they are"]
        for e in refused.value.exits[:-1]:
            validate(e["decision"], ctx)


def test_a_refused_revert_never_offers_the_fill_already_recorded(tmp_path):
    """The verifier's revert: undoing the fill brings complete cases back and is refused, and its
    first exit was a fill identical to the answer recorded, so taking it changed nothing. No exit
    is the answer already recorded."""
    from datetime import datetime, timezone

    from turbotab.core.decisions import DecisionRecord, Revert, SetMissing, fold, parse_decision

    frame = renal(complete=4)  # complete cases keep 4 rows
    store, bundle = toy_store(frame, tmp_path)
    at = datetime.now(timezone.utc)
    answers = [{"kind": "set_target", "column": "egfr_decline"},
               {"kind": "set_purpose", "purpose": "prediction"},
               {"kind": "set_roles", "roles": renal_state().roles},
               {"kind": "set_missing", "strategy": "complete_case"},
               {"kind": "set_missing", "strategy": "impute"}]
    records = [DecisionRecord(id=f"r{i}", seq=i, at=at, decision=parse_decision(d))
               for i, d in enumerate(answers, start=1)]
    now = fold(records)
    with store:
        ctx = {"state": now, "store": store, "columns": list(frame.columns),
               "target": "egfr_decline", "bundle": bundle, "records": records}
        with pytest.raises(Refusal) as refused:
            validate(Revert(decision_id="r5"), ctx)
    assert refused.value.code == "too_few_rows"
    for e in refused.value.exits:
        if e["decision"] is not None:
            assert fold_onto(now, validate(e["decision"], ctx)) != now, e["label"]


def test_the_design_and_the_sensitivity_stage_say_one_value_and_too_few_rows_plainly(tmp_path):
    """No stage is handed what a model cannot learn from: training rows whose outcome is one value
    fail the design in plain words (the fit crashed with an IndexError), and a sensitivity analysis
    that keeps too few rows is not fit, with its reason beside it (never scikit-learn's)."""
    from turbotab.core.decisions import SensitivityAnalysis
    from turbotab.core.stages.modeling import design_stage
    from turbotab.core.stages.sensitivity import sensitivity_stage
    from turbotab.core.tests import modeling_fixtures as mf

    frame = mf.nhanes_like(60, seed=1)
    frame["high"] = (frame["glucose"] > frame["glucose"].median()).astype(int)
    paths = mf.ingest_frame(frame, tmp_path)
    split = mf.split_bundle(np.arange(60))
    train = split.frames["assignment"].query("partition == 'train'")["row_id"].to_numpy()
    one = frame.assign(high=0)
    one.loc[~one.index.isin(train), "high"] = 1  # the other value only among the held-out rows
    one_paths = mf.ingest_frame(one, tmp_path / "one")
    binary = mf.state(target="high", models=["linear"])
    with pytest.raises(ValueError) as refused:
        design_stage(mf.context(binary, {"split": split, "target_info": mf.target_info("binary", "high")},
                                one_paths))
    assert str(refused.value) == (
        f"`high` is `0` in every one of the {len(train)} training rows, so the models have no other "
        f"value of the outcome to learn from. The row flow says which answers removed the rows "
        f"with other values.")

    state = mf.state(models=["linear"], sensitivity=[SensitivityAnalysis(
        label="the oldest", rules=[ExclusionRule(column="age", low=90, high=99, reason="the oldest")])])
    inputs = {"split": split, "target_info": mf.target_info("regression")}
    design = design_stage(mf.context(state, inputs, paths))
    from turbotab.core.datastore import DataStore

    with DataStore(paths["data"], 1 << 30) as store:
        ingest = store.info().to_dict()
    out = sensitivity_stage(mf.context(state, {**inputs, "design": design, "ingest": ingest}, paths))
    fits = {f["label"]: f for f in out.data["families"][0]["fits"]}
    assert fits["the oldest"]["concerns"] == [
        f"This analysis keeps no eligible rows outside the held-out set, and a fit needs at least "
        f"{FEWEST_ROWS}: widen its rules, or leave it out."]
    assert fits["Primary"]["coefficients"]


# ── the residue: every exit row-checked, none that reproduces the primary ─────


def _missing_refusals(purpose: str):
    """The missing-values answers other validators refuse with a complete-case exit, and the
    state each is refused in: multiple imputation under prediction; under inference a single
    fill, passive imputation beside a spline, and single-level imputation on repeated rows."""
    from turbotab.core.decisions import ExposureFormSpec, GrainSpec

    labs = {f"lab_{c}": "exposure" for c in "abcdefghijkl"}
    if purpose == "prediction":  # at the missing-values question: no answer yet
        return [("imputation_with_the_outcome", renal_state(missing=None),
                 {"kind": "set_missing", "strategy": "multiple_imputation"})]
    inference = dict(purpose="inference", missing=None,
                     roles={"age_years": "covariate", "sbp": "covariate", **labs})
    return [
        ("single_fill_under_inference", renal_state(**inference),
         {"kind": "set_missing", "strategy": "impute"}),
        ("passive_imputation_with_nonlinear_terms",
         renal_state(**inference, exposure_forms={"lab_a": ExposureFormSpec(form="spline", knots=3)}),
         {"kind": "set_missing", "strategy": "multiple_imputation", "imputation_model": "passive"}),
        ("single_level_imputation_on_clustered_rows",
         renal_state(**inference, grain=GrainSpec(grain="repeated", id_column="pt"), unit="row"),
         {"kind": "set_missing", "strategy": "multiple_imputation",
          "imputation_levels": "single_level"}),
    ]


@pytest.mark.parametrize("purpose", ["prediction", "inference"])
def test_every_missing_values_refusal_offers_complete_cases_only_where_they_leave_rows(purpose,
                                                                                       tmp_path):
    """The integrator's residue: multiple imputation under prediction, and under inference the
    single fill, passive imputation beside a nonlinear term and single-level imputation on repeated
    rows, were refused with complete cases among their exits, never counted. On the renal table
    complete cases keep 4 rows (pandas, below), so taking that exit was refused in turn. It is not
    offered now, and every exit that is offered is accepted. With 40 complete rows it is offered,
    and accepted."""
    from turbotab.core.decisions import Refusal as Refused

    for complete, offered in ((4, False), (40, True)):
        frame = renal(complete=complete)
        frame["pt"] = np.repeat(np.arange(len(frame) // 2), 2)  # two rows per participant
        preds = [c for c in renal_state().roles]
        assert int(frame[preds].notna().all(axis=1).sum()) == complete  # the reference
        folder = tmp_path / f"{purpose}-{complete}"
        folder.mkdir()
        store, bundle = toy_store(frame, folder)
        with store:
            for code, now, answer in _missing_refusals(purpose):
                ctx = {"state": now, "store": store, "columns": list(frame.columns),
                       "target": "egfr_decline", "bundle": bundle}
                with pytest.raises(Refused) as refused:
                    validate(answer, ctx)
                assert refused.value.code == code
                ways = [e["decision"] for e in refused.value.exits if e["decision"] is not None]
                ways = [w if isinstance(w, dict) else w.model_dump(mode="json") for w in ways]
                complete_cases = [w for w in ways if w.get("strategy") == "complete_case"]
                assert bool(complete_cases) is offered, (code, complete, refused.value.exits)
                for way in ways:
                    validate(way, ctx)  # every exit offered is one the record accepts


def test_a_sensitivity_exit_never_leaves_the_analysis_the_primary(tmp_path):
    """The integrator's residue: "Drop the rule from <label>" could leave a sensitivity analysis
    with the primary's rules (no rule, under a primary with none), which is the primary analysis
    again. Such an exit is not offered; leaving the analysis out is. Dropping a rule, or the rules
    together, where that leaves a different analysis is still offered."""
    from turbotab.core.decisions import SensitivityAnalysis, SetSensitivity

    toy = pd.DataFrame({"y": np.linspace(0, 1, 40) ** 2, "x": np.linspace(-1, 1, 40),
                        "age": 20 + 1.5 * np.arange(40), "sbp": np.linspace(100, 160, 40)})
    store, bundle = toy_store(toy, tmp_path)
    adults = ExclusionRule(column="age", low=30, high=79, reason="adults")
    oldest = ExclusionRule(column="age", low=90, high=99, reason="the oldest")
    older = ExclusionRule(column="age", low=60, reason="older")
    low_sbp = ExclusionRule(column="sbp", low=0, high=90, reason="hypotension")
    assert int((toy.age >= 60).sum()) == 13 and int((toy.sbp <= 90).sum()) == 0  # the reference
    with store:
        for primary, rules, labels in (
                # without `the oldest` it is the primary; without both, every row
                ([adults], [adults, oldest],
                 ["Drop the rules from “a”: `age` within `30`–`79` and `age` within `90`–`99`",
                  "Leave out the analysis “a”"]),
                # without `hypotension` it is the older adults, not the primary (every row)
                ([], [older, low_sbp], ["Drop the rule from “a”: `sbp` within `0`–`90`",
                                        "Leave out the analysis “a”"]),
                # without `hypotension` it is the primary
                ([older], [older, low_sbp],
                 ["Drop the rules from “a”: `age` at least `60` and `sbp` within `0`–`90`",
                  "Leave out the analysis “a”"])):
            now = ProjectState(target="y", task="regression", purpose="inference",
                               roles={"x": "exposure", "age": "covariate", "sbp": "covariate"},
                               missing=MissingSpec(strategy="complete_case"), exclusions=primary)
            ctx = {"state": now, "store": store, "columns": list(toy.columns), "target": "y",
                   "bundle": bundle}
            answer = SetSensitivity(analyses=[SensitivityAnalysis(label="b", rules=[adults]),
                                              SensitivityAnalysis(label="a", rules=rules)])
            with pytest.raises(Refusal) as refused:
                validate(answer, ctx)
            assert refused.value.code == "too_few_rows"
            assert [e["label"] for e in refused.value.exits] == [*labels,
                                                                 "Keep the answers as they are"]
            for e in refused.value.exits[:-1]:
                taken = SetSensitivity.model_validate(validate(e["decision"], ctx).model_dump())
                for a in taken.analyses:
                    if a.label == "a":  # what the exit made of it is never the primary
                        assert sorted(r.model_dump_json() for r in a.rules) != sorted(
                            r.model_dump_json() for r in primary), e["label"]

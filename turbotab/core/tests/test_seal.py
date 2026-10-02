"""Tier A: the seal (M2_CONTRACT §3; ROADMAP lockbox constitution §01–§05).

Every held-out claim rests on these: the basis is one of four recorded states and an undetermined
seal is never a clean one; a chronological split puts every training unit's last time before every
held-out unit's; elastic net's inner cross-validation never splits a unit; a family that ties its
baseline says so; held-out scores live outside the public artifact until the seal is opened; Decision
A waits for a re-seal; and every decision after the opening is marked post-seal.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from turbotab.core import decisions as d
from turbotab.core import seal
from turbotab.core.decisions import (
    DecisionLog, DecisionRecord, GrainSpec, ProjectState, Refusal, RepeatSpec, SplitSpec, TemporalSpec,
)
from turbotab.core.graph import Bundle
from turbotab.core.models.baseline import MIN_GAIN, versus_baseline
from turbotab.core.models.inner_cv import inner_splits, with_grouped_inner_cv
from turbotab.core.stages.modeling import design_stage, fit_stage
from turbotab.core.stages.rows import cohort_stage, split_stage
from turbotab.core.stages.target import target_info_stage
from turbotab.core.tests import modeling_fixtures as mf
from turbotab.core.tests.stage_harness import SAMPLES, Ingested


@pytest.fixture(scope="module")
def dietary(tmp_path_factory):
    return Ingested(SAMPLES / "dietary_recalls.csv", tmp_path_factory.mktemp("dietary"))


@pytest.fixture(scope="module")
def clinical(tmp_path_factory):
    return Ingested(SAMPLES / "clinical_longitudinal.csv", tmp_path_factory.mktemp("clinical"))


def drawn(table: Ingested, state: ProjectState) -> Bundle:
    """The split stage on ``table`` under ``state``, through the real cohort and target stages."""
    info = table.run(target_info_stage, state)
    cohort = table.run(cohort_stage, state, {"target_info": info})
    return table.run(split_stage, state, {"cohort": cohort, "target_info": info})


DIETARY_ROLES = {"participant_id": "identifier", "age": "covariate", "bmi": "covariate",
                 "energy_kcal": "energy", "protein_g": "exposure", "fat_g": "exposure"}


def dietary_state(**slots) -> ProjectState:
    base = dict(target="hba1c", task="regression", roles=dict(DIETARY_ROLES),
                missing="complete_case", split=SplitSpec(holdout=0.2, seed=3, folds=5))
    base.update(slots)
    return ProjectState(**base)


def sides(table: Ingested, split: Bundle, column: str) -> pd.DataFrame:
    a = split.frames["assignment"]
    units = table.frame([column]).loc[a["row_id"].to_numpy(), column].to_numpy()
    return pd.DataFrame({"unit": units, "part": a["partition"].to_numpy(), "fold": a["fold"].to_numpy()})


# ── the basis: four recorded states, never two ───────────────────────────────


def test_a_stated_repeated_grain_groups_the_seal_by_its_column(dietary):
    split = drawn(dietary, dietary_state(grain=GrainSpec(grain="repeated", id_column="participant_id")))
    basis = split.data["basis"]
    assert basis["state"] == "grouped" and basis["column"] == "participant_id"
    assert basis["source"] == "grain" and not basis["exploratory"] and not split.data["exploratory"]
    assert basis["label"] == "grouped by `participant_id`" and basis["n_units"] == 300
    s = sides(dietary, split, "participant_id")
    assert s.groupby("unit")["part"].nunique().max() == 1  # no person on both sides of the seal


def test_no_seal_is_drawn_without_a_grain(dietary):
    """M2_CONTRACT §12.2: the seal requires grain. Skipping the question never draws a seal, so it
    can never yield an undetermined one either; the split stops and says the grain comes first."""
    with pytest.raises(ValueError, match="grain answer"):
        drawn(dietary, dietary_state())
    draw = seal.seal_inputs(dietary_state(), np.arange(10), None, "regression", holdout=0.2, seed=0)
    assert draw.basis is None and draw.refusal == seal.GRAIN_FIRST


def test_one_row_per_unit_said_over_a_repeating_identifier_is_abandoned_and_exploratory(dietary):
    split = drawn(dietary, dietary_state(grain=GrainSpec(grain="one_row_per_unit")))
    basis = split.data["basis"]
    assert basis["state"] == "abandoned" and basis["column"] == "participant_id"
    assert basis["label"] == "repetition found but grouping abandoned"
    assert basis["exploratory"] and split.data["exploratory"]
    assert split.data["grouped_by"] is None  # drawn by row, as answered, and said so
    assert "exploratory" in basis["sentence"]


def test_i_dont_know_is_undetermined_never_a_clean_lock(dietary):
    split = drawn(dietary, dietary_state(grain=GrainSpec(grain="unknown")))
    basis = split.data["basis"]
    assert basis["state"] == "undetermined" and basis["exploratory"] and split.data["exploratory"]
    assert basis["label"] == "undetermined" and "not a verified clean split" in basis["sentence"]
    assert "not known" in basis["sentence"] and "`participant_id` repeats" in basis["sentence"]
    assert split.data["grouped_by"] is None  # drawn by row, as the basis says
    # A repeated grain with no column naming the unit is not "I don't know": grouping was abandoned.
    roles = {c: r for c, r in DIETARY_ROLES.items() if r != "identifier"}
    split = drawn(dietary, dietary_state(roles=roles, grain=GrainSpec(grain="repeated")))
    assert split.data["basis"]["state"] == "abandoned" and split.data["exploratory"]


def test_a_verified_one_row_per_unit_seal_is_clean_and_distinct_from_undetermined():
    frame = pd.DataFrame({"pid": [f"p{i}" for i in range(40)], "x": np.arange(40.0)})
    state = ProjectState(roles={"pid": "identifier", "x": "covariate"},
                         grain=GrainSpec(grain="one_row_per_unit"))
    basis, column = seal.decide_basis(state, frame, ["pid"])
    assert (basis.state, basis.exploratory, column) == ("one_row_per_unit", False, None)
    unknown, _ = seal.decide_basis(ProjectState(grain=GrainSpec(grain="unknown")), frame, [])
    assert unknown.state == "undetermined" and unknown.model_dump() != basis.model_dump()
    # the grain the Router states from a unique identifier is clean too, and says it was stated
    stated, _ = seal.decide_basis(state.model_copy(update={"grain": GrainSpec(
        grain="one_row_per_unit", id_column="pid")}), frame, ["pid"], stated=True)
    assert (stated.state, stated.source, stated.exploratory) == ("one_row_per_unit", "stated", False)
    assert "every `pid` appears once" in stated.sentence


GRAIN_ANSWERS = [
    None,
    GrainSpec(grain="repeated", id_column="pid"),
    GrainSpec(grain="repeated", id_column="gone"),  # a column the table does not have
    GrainSpec(grain="repeated", id_column="site"),  # too few units to hold any out whole
    GrainSpec(grain="repeated"),
    GrainSpec(grain="one_row_per_unit"),
    GrainSpec(grain="one_row_per_unit", id_column="pid", acknowledged=True),
    GrainSpec(grain="unknown"),
]


@pytest.mark.parametrize("repeats", [True, False], ids=["pid-repeats", "pid-unique"])
@pytest.mark.parametrize("grain", GRAIN_ANSWERS, ids=lambda g: "unanswered" if g is None else
                         f"{g.grain}-{g.id_column}")
@pytest.mark.parametrize("stated", [False, True], ids=["answered", "stated"])
def test_an_undetermined_basis_comes_only_from_answering_i_dont_know(grain, repeats, stated):
    """Tier A (M2_CONTRACT §12.2): ``undetermined`` iff the grain answer is ``unknown``, over every
    grain answer and data shape; no answer draws no seal at all rather than an undetermined one."""
    n = 40
    pid = [f"p{i // 2}" for i in range(n)] if repeats else [f"p{i}" for i in range(n)]
    frame = pd.DataFrame({"pid": pid, "site": ["a", "b", "c", "d"] * (n // 4)})
    state = ProjectState(roles={"pid": "identifier"}, grain=grain)
    if grain is None:
        with pytest.raises(ValueError, match="grain answer"):
            seal.decide_basis(state, frame, ["pid"], stated=stated)
        return
    basis, _ = seal.decide_basis(state, frame, ["pid"], stated=stated and grain.grain == "one_row_per_unit")
    assert (basis.state == "undetermined") == (grain.grain == "unknown"), basis
    assert basis.exploratory == (basis.state in ("abandoned", "undetermined"))


def test_too_few_units_to_hold_out_whole_is_abandoned():
    frame = pd.DataFrame({"site": ["a", "b", "c"] * 10})
    state = ProjectState(grain=GrainSpec(grain="repeated", id_column="site"))
    basis, column = seal.decide_basis(state, frame, [])
    assert basis.state == "abandoned" and basis.n_units == 3 and column is None
    assert basis.exploratory


# ── the chronological split ──────────────────────────────────────────────────


CLINICAL_ROLES = {"subject_id": "identifier", "visit": "time", "visit_date": "time", "age": "covariate",
                  "sbp": "covariate", "glucose": "covariate"}


def clinical_state(**slots) -> ProjectState:
    base = dict(target="progressed", task="binary", roles=dict(CLINICAL_ROLES),
                missing="complete_case", split=SplitSpec(holdout=0.2, seed=1, folds=5),
                grain=GrainSpec(grain="repeated", id_column="subject_id"),
                repeat_kind=RepeatSpec(repeat_kind="time_points", time_column="visit_date"),
                unit="row", temporal=TemporalSpec(temporal=True, time_column="visit_date"))
    base.update(slots)
    return ProjectState(**base)


def test_a_chronological_split_puts_every_training_time_before_every_held_out_time(clinical):
    split = drawn(clinical, clinical_state())
    chronology = split.data["chronology"]
    assert chronology["drawn"] and chronology["time_column"] == "visit_date"
    a = split.frames["assignment"]
    frame = clinical.frame(["subject_id", "visit_date"]).loc[a["row_id"].to_numpy()]
    frame["part"] = a["partition"].to_numpy()
    frame["when"] = pd.to_datetime(frame["visit_date"])
    # Grouped too: no subject on both sides.
    assert frame.groupby("subject_id")["part"].nunique().max() == 1
    # Within the grouping, by each subject's last visit: every training subject's comes first.
    last = frame.groupby("subject_id").agg(when=("when", "max"), part=("part", "first"))
    held, train = last[last["part"] == "holdout"], last[last["part"] == "train"]
    assert len(held) == 40 and len(train) == 160  # the latest 20% of 200 subjects
    assert train["when"].max() < held["when"].min()
    assert chronology["boundary"] == str(held["when"].min().date())
    assert chronology["n_units"] == 40 and not split.data["exploratory"]
    # The training folds stay grouped.
    t = frame[frame["part"] == "train"].assign(fold=a.loc[a["partition"] == "train", "fold"].to_numpy())
    assert t.groupby("subject_id")["fold"].nunique().max() == 1


def test_chronological_by_row_when_no_unit_repeats():
    times = np.array([5.0, 1.0, 3.0, 9.0, 7.0, 2.0, 8.0, 4.0, 6.0, 10.0])
    mask, chronology = seal.chronological_holdout(times, None, 0.3, 0, "year", dated=False)
    assert sorted(times[mask]) == [8.0, 9.0, 10.0] and times[~mask].max() < times[mask].min()
    assert chronology.drawn and chronology.boundary == "8"


def test_temporal_without_a_time_column_draws_at_random_and_says_so(clinical):
    split = drawn(clinical, clinical_state(temporal=TemporalSpec(temporal=True),
                                           repeat_kind=RepeatSpec(repeat_kind="time_points")))
    assert not split.data["chronology"]["drawn"] and split.data["exploratory"]
    assert split.data["basis"]["state"] == "grouped"  # still grouped: never by row


def test_a_named_time_column_that_cannot_order_refuses(clinical):
    state = clinical_state(temporal=TemporalSpec(temporal=True, time_column="sex"))
    with pytest.raises(ValueError, match="`sex`"):
        drawn(clinical, state)


# ── what a holdout of this size can measure ──────────────────────────────────


def test_below_the_floor_cross_validation_alone_comes_first_and_nothing_is_removed():
    small, cv_first, reason = seal.holdout_options("regression", 300)
    assert cv_first and small[0].holdout == 0 and "floor of 100" in reason
    assert sorted(o.holdout for o in small) == list(seal.HOLDOUTS)  # the shelf is never shortened
    large, cv_first, _ = seal.holdout_options("regression", 5000)
    assert not cv_first and large[0].holdout == seal.USUAL and large[-1].holdout == 0
    usual = next(o for o in large if o.holdout == 0.2)
    assert usual.n_holdout == 1000 and "R²" in usual.measures and not usual.below_floor
    # A binary outcome counts events, and its floor is cited, not a convention.
    rare, cv_first, _ = seal.holdout_options("binary", 5000, [4750, 250])
    assert cv_first and next(o for o in rare if o.holdout == 0.2).below_floor
    assert not seal.floor_for("binary").convention and seal.floor_for("binary").source
    assert seal.floor_for("regression").convention


def test_the_precision_of_a_held_out_r2_shrinks_with_n():
    def width(n):
        text, _ = seal.measure("regression", n)
        return float(text.split("±")[1].rstrip("."))
    assert width(100) > width(400) > width(1600)
    assert width(400) == pytest.approx(1.96 * np.sqrt(2 / 400), abs=0.006)


# ── grouped inner cross-validation ───────────────────────────────────────────


def test_grouped_inner_splits_never_split_a_unit():
    groups = np.repeat([f"u{i}" for i in range(37)], 3)
    for train, test in inner_splits(groups, 5):
        assert not set(groups[train]) & set(groups[test])


def test_elastic_net_inner_cv_never_splits_a_unit_in_any_fit(tmp_path, monkeypatch):
    """Every elastic-net fit (each outer fold and the refit) tunes its penalty on grouped folds."""
    from sklearn.linear_model import ElasticNetCV

    frame = mf.nhanes_like(360, seed=11)
    frame["SEQN"] = np.repeat(np.arange(120), 3)  # each person three times
    paths = mf.ingest_frame(frame, tmp_path)
    split = mf.split_bundle(np.arange(len(frame)), groups=frame["SEQN"].to_numpy(), grouped_by="SEQN")
    seen = []
    original = ElasticNetCV.fit

    def spy(self, X, y, **kw):
        seen.append((np.asarray(X.index), self.cv))
        return original(self, X, y, **kw)

    monkeypatch.setattr(ElasticNetCV, "fit", spy)
    st = mf.state(energy_adjustment=mf.energy("none"), models=["elastic_net"])
    ti = mf.target_info("regression")
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
    person = frame["SEQN"].to_numpy()
    assert len(seen) == 6  # five outer folds and the refit on every training row
    for row_ids, cv in seen:
        assert isinstance(cv, list) and len(cv) >= 2
        for train, test in cv:
            assert not set(person[row_ids[train]]) & set(person[row_ids[test]])


def test_an_ungrouped_split_leaves_the_inner_cv_as_it_was():
    from sklearn.linear_model import ElasticNetCV
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([("model", ElasticNetCV(cv=5))])
    assert with_grouped_inner_cv(pipe, None).named_steps["model"].cv == 5


# ── no better than the baseline, within a stated tolerance ───────────────────


def test_a_gain_inside_the_tolerance_is_no_better_than_the_baseline():
    base = [0.0] * 5
    tie = versus_baseline("r2", [0.004, -0.002, 0.006, 0.001, 0.003], base)
    assert tie.verdict == "no_better" and tie.tolerance == pytest.approx(MIN_GAIN)
    assert "0.01 (a convention)" in tie.tolerance_basis and "5 folds" in tie.tolerance_basis
    assert versus_baseline("r2", [0.12, 0.10, 0.11, 0.13, 0.09], base).verdict == "better"
    assert versus_baseline("r2", [-0.03, -0.01, -0.02, 0.0, -0.04], base).verdict == "worse"
    # Noisy folds widen the tolerance to one standard error of the gain.
    noisy = versus_baseline("auc", [0.62, 0.40, 0.58, 0.45, 0.55], [0.5] * 5)
    assert noisy.tolerance > MIN_GAIN and noisy.verdict == "no_better"


def test_a_fit_on_noise_says_it_is_no_better_than_the_baseline_first(tmp_path):
    frame = mf.nhanes_like(400, seed=5)
    frame["glucose"] = np.random.default_rng(9).normal(100, 10, len(frame))  # nothing to learn
    paths = mf.ingest_frame(frame, tmp_path)
    split = mf.split_bundle(np.arange(len(frame)), seed=2)
    st = mf.state(energy_adjustment=mf.energy("none"), models=["linear", "elastic_net"])
    ti = mf.target_info("regression")
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
    for m in fit.data["models"]:
        verdict = m["versus_baseline"]["verdict"]
        assert verdict in ("no_better", "worse"), m["versus_baseline"]
        assert m["concerns"][0].startswith("No better than" if verdict == "no_better" else "Predicts worse")


# ── withholding ──────────────────────────────────────────────────────────────


def test_served_fit_withholds_until_opened_whatever_the_artifact_holds():
    data = {"n_holdout": 50, "models": [{"family": "linear", "holdout": {"r2": 0.31}}]}
    closed = seal.serve_fit(data, opened=False, scores=lambda: pytest.fail("read while sealed"))
    assert closed["holdout_sealed"] is True and closed["models"][0]["holdout"] is None
    opened = seal.serve_fit(data, opened=True, scores=lambda: {"linear": {"r2": 0.27}})
    assert opened["holdout_sealed"] is False and opened["models"][0]["holdout"] == {"r2": 0.27}
    assert data["models"][0]["holdout"] == {"r2": 0.31}  # the input is not modified
    nothing = seal.serve_fit({"n_holdout": 0, "models": [{"family": "linear", "holdout": None}]},
                             opened=False, scores=lambda: None)
    assert nothing["holdout_sealed"] is False


# ── refusals ─────────────────────────────────────────────────────────────────


def record(seq: int, decision, rid: str | None = None, post_seal: bool = False) -> DecisionRecord:
    return DecisionRecord(id=rid or f"r{seq}", seq=seq, at=datetime.now(timezone.utc),
                          decision=decision, post_seal=post_seal)


def sealed_log():
    return [
        record(1, d.SetTarget(column="hba1c")),
        record(2, d.SetGrain(grain="repeated", id_column="participant_id"), "grain"),
        record(3, d.SetUnit(unit="row"), "unit"),
        record(4, d.SetSplit(holdout=0.2, seed=0, folds=5), "split-1"),
        record(5, d.SetSplit(holdout=0.25, seed=0, folds=5), "split-2"),
    ]


@pytest.mark.parametrize("decision", [
    d.SetOrientation(orientation="feature_major"),
    d.SetGrain(grain="one_row_per_unit"),
    d.SetUnit(unit="unit"),
    d.SetAggregation(method="mean"),
], ids=lambda x: x.kind)
def test_decision_a_is_refused_once_the_seal_is_drawn_with_the_reseal_path(decision):
    records = sealed_log()
    ctx = {"state": d.fold(records), "records": records}
    with pytest.raises(Refusal) as refused:
        seal._decision_a_waits_for_a_reseal(decision, ctx)
    assert refused.value.code == "sealed"
    [exit_] = refused.value.exits
    assert "Re-seal" in exit_["label"]
    assert exit_["decision"] == {"kind": "revert", "decision_id": "split-2"}  # the live seal's record
    # Before the seal is drawn the same answer is not refused (by this rule).
    before = records[:3]
    seal._decision_a_waits_for_a_reseal(decision, {"state": d.fold(before), "records": before})


def test_decision_a_under_cross_validation_only_names_the_folds_not_held_out_rows():
    """M2_CONTRACT §12.3: with nothing held out, the refusal says the folds name rows as they were."""
    records = [*sealed_log()[:3], record(4, d.SetSplit(holdout=0.0, seed=0, folds=5), "split-cv")]
    with pytest.raises(Refusal) as refused:
        seal._decision_a_waits_for_a_reseal(d.SetUnit(unit="unit"),
                                            {"state": d.fold(records), "records": records})
    message = refused.value.message
    assert "folds" in message and "held-out rows" not in message and "seal names" not in message
    [exit_] = refused.value.exits
    assert exit_["label"].startswith("Re-draw") and "folds" in exit_["label"]
    assert exit_["decision"] == {"kind": "revert", "decision_id": "split-cv"}
    with pytest.raises(Refusal) as by_revert:
        seal._revert_keeps_the_seal(d.Revert(decision_id="grain"), {"records": records})
    assert "folds" in by_revert.value.message


def test_the_same_answer_again_is_not_a_change():
    records = sealed_log()
    seal._decision_a_waits_for_a_reseal(d.SetUnit(unit="row"), {"state": d.fold(records), "records": records})


def test_a_revert_that_would_change_what_a_row_is_waits_for_a_reseal_too():
    records = sealed_log()
    ctx = {"records": records}
    with pytest.raises(Refusal) as refused:
        seal._revert_keeps_the_seal(d.Revert(decision_id="grain"), ctx)
    assert refused.value.code == "sealed"
    # Withdrawing the seal itself is the way through: reverting both split records unseals.
    seal._revert_keeps_the_seal(d.Revert(decision_id="split-2"), ctx)
    unsealed = [*records, record(6, d.Revert(decision_id="split-2")), record(7, d.Revert(decision_id="split-1"))]
    seal._revert_keeps_the_seal(d.Revert(decision_id="grain"), {"records": unsealed})


def opened_log():
    return [*sealed_log(), record(6, d.SelectModels(models=["linear"])), record(7, d.OpenSeal(), "open")]


def test_the_seal_opens_once_on_a_fresh_fit_and_is_never_reverted():
    fresh = {"n_holdout": 120, "models": []}
    records = sealed_log()
    state = d.fold(records)
    seal._open_seal_once_on_a_fresh_fit(d.OpenSeal(), {"state": state, "artifact": lambda s: fresh})
    with pytest.raises(Refusal) as stale:
        seal._open_seal_once_on_a_fresh_fit(d.OpenSeal(), {"state": state, "artifact": lambda s: None})
    assert stale.value.code == "fit_not_fresh"
    with pytest.raises(Refusal) as none:
        seal._open_seal_once_on_a_fresh_fit(d.OpenSeal(), {"state": ProjectState(), "artifact": lambda s: fresh})
    assert none.value.code == "no_seal"
    cv_only = state.model_copy(update={"split": SplitSpec(holdout=0.0)})
    with pytest.raises(Refusal) as nothing:
        seal._open_seal_once_on_a_fresh_fit(d.OpenSeal(), {"state": cv_only, "artifact": lambda s: fresh})
    assert nothing.value.code == "nothing_sealed"
    opened = opened_log()
    with pytest.raises(Refusal) as twice:
        seal._open_seal_once_on_a_fresh_fit(d.OpenSeal(), {"state": d.fold(opened),
                                                           "artifact": lambda s: fresh})
    assert twice.value.code == "seal_already_open"
    with pytest.raises(Refusal) as undo:
        seal._revert_keeps_the_seal(d.Revert(decision_id="open"), {"records": opened})
    assert undo.value.code == "seal_stays_open"


# ── post-seal marking ────────────────────────────────────────────────────────


def test_every_decision_after_the_opening_is_marked_post_seal(tmp_path):
    log = DecisionLog(tmp_path / "decisions.jsonl")
    sentence = lambda decision, before: seal.post_seal_sentence("Models were chosen.", before)  # noqa: E731
    first = log.append(d.SelectModels(models=["linear"]), sentence=sentence)
    opening = log.append(d.OpenSeal(), sentence=sentence)
    after = log.append(d.SelectModels(models=["elastic_net"]), sentence=sentence)
    undo = log.append(d.Revert(decision_id=after.id), sentence=sentence)
    assert (first.post_seal, opening.post_seal, after.post_seal, undo.post_seal) == (False, False, True, True)
    assert first.sentence == "Models were chosen."
    assert after.sentence == "After the held-out rows were opened, models were chosen."
    # Read back from the file: the flag is on each line as it was written, never rewritten.
    assert [r.post_seal for r in DecisionLog(log.path).records()] == [False, False, True, True]
    slots = {"models"}
    assert seal.post_seal_changes(log.records(), slots) == [undo.id]  # `after` is reverted


def test_the_post_seal_sentence_keeps_a_data_value_or_an_acronym_as_it_is():
    opened = ProjectState(seal_opened=True)
    assert seal.post_seal_sentence("`177` rows were excluded.", opened) == (
        "After the held-out rows were opened, `177` rows were excluded.")
    assert seal.post_seal_sentence("NHANES weights were used.", opened).endswith("NHANES weights were used.")
    assert seal.post_seal_sentence("Rows were excluded.", ProjectState()) == "Rows were excluded."

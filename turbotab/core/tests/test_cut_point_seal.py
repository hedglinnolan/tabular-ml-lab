"""A temporal seal by cut-point (SWARM_PLAN §8 and §10's watch note on package 3).

Every row whose time column reads at or after a value is held out, a unit with rows on both sides
held out whole, and the other rows train; the training folds' order is its own answer, random
(repeated or not) or time-ordered, so the same held-out rows can be scored under either. Each
expected row set below is computed from the table the test writes, with numpy or pandas, never
read from the engine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from turbotab.core import decisions as d
from turbotab.core import seal, voice
from turbotab.core.decisions import (CutPoint, DecisionLog, GrainSpec, ProjectState, Refusal,
                                     SplitSpec, TemporalSpec, validate)
from turbotab.core.stages.rows import cohort_stage, split_stage
from turbotab.core.stages.target import target_info_stage
from turbotab.core.tests.stage_harness import Ingested

CYCLES = (2011.0, 2013.0, 2015.0, 2016.99, 2017.0, 2019.0)


def table(tmp_path, grouped: bool = False, dated: bool = False) -> tuple[Ingested, pd.DataFrame]:
    """Sixty rows, ten in each of six cycles; with ``grouped``, each person in two of them."""
    rng = np.random.default_rng(7)
    cycle = np.repeat(CYCLES, 10)
    frame = pd.DataFrame({"cycle": cycle, "x": rng.normal(size=60)})
    frame["y"] = frame["x"] + rng.normal(scale=0.5, size=60)
    if grouped:  # person p appears in rows p and p + 30: a person seen in 2015 and again in 2017
        frame["pid"] = [f"p{i % 30}" for i in range(60)]
    if dated:
        frame["visit"] = pd.to_datetime("2019-12-30") + pd.to_timedelta(np.arange(60) % 4, unit="D")
        frame["visit"] = frame["visit"].dt.strftime("%Y-%m-%d")
    path = tmp_path / "cycles.csv"
    frame.to_csv(path, index=False)
    return Ingested(path, tmp_path), frame


def state(cut: CutPoint | None, *, fold_order="random", holdout=0.0, grouped=False, **slots):
    roles = {"x": "covariate", **({"pid": "identifier"} if grouped else {})}
    grain = (GrainSpec(grain="repeated", id_column="pid") if grouped
             else GrainSpec(grain="one_row_per_unit"))
    split = SplitSpec(holdout=holdout, seed=3, folds=3, cut_point=cut,
                      fold_order=fold_order if cut is not None else None,
                      **slots.pop("scheme", {}))
    return ProjectState(target="y", task="regression", roles=roles, missing="complete_case",
                        grain=grain, split=split, **slots)


def drawn(t: Ingested, s: ProjectState):
    info = t.run(target_info_stage, s)
    cohort = t.run(cohort_stage, s, {"target_info": info})
    return t.run(split_stage, s, {"cohort": cohort, "target_info": info})


def held_rows(split) -> set[int]:
    a = split.frames["assignment"]
    return set(a.loc[a["partition"] == "holdout", "row_id"].tolist())


# ── the boundary ─────────────────────────────────────────────────────────────


def test_every_row_at_or_after_the_cut_point_is_held_out_and_every_earlier_row_trains(tmp_path):
    t, frame = table(tmp_path)
    split = drawn(t, state(CutPoint(column="cycle", at=2017)))
    expect = set(np.flatnonzero(frame["cycle"].to_numpy() >= 2017.0).tolist())
    assert held_rows(split) == expect and len(expect) == 20
    # The boundary: 2017 itself is held out; 2016.99, a hair before it, trains.
    a = split.frames["assignment"].set_index("row_id")
    assert set(a.loc[frame.index[frame["cycle"] == 2017.0], "partition"]) == {"holdout"}
    assert set(a.loc[frame.index[frame["cycle"] == 2016.99], "partition"]) == {"train"}
    chron = split.data["chronology"]
    assert chron["drawn"] and chron["cut_point"] == "2017" and chron["time_column"] == "cycle"
    assert chron["sentence"] == (
        "Held out by a cut-point at `2017` on `cycle`: `20` rows held out and `40` train. A "
        "cut-point holds out every row at or after it, with the earlier rows of any unit it holds "
        "out; the other rows train.")
    assert split.data["n_holdout"] == 20 and split.data["n_train"] == 40


def test_a_cut_point_written_as_text_reads_as_a_number_on_a_numeric_column(tmp_path):
    t, _ = table(tmp_path)
    as_text = drawn(t, state(CutPoint(column="cycle", at="2017")))
    as_number = drawn(t, state(CutPoint(column="cycle", at=2017.0)))
    assert held_rows(as_text) == held_rows(as_number)


def test_a_dated_column_cuts_at_its_date(tmp_path):
    t, frame = table(tmp_path, dated=True)
    split = drawn(t, state(CutPoint(column="visit", at="2020-01-01")))
    expect = set(np.flatnonzero(pd.to_datetime(frame["visit"]) >= pd.Timestamp("2020-01-01")).tolist())
    assert held_rows(split) == expect and 0 < len(expect) < 60
    assert split.data["chronology"]["cut_point"] == "2020-01-01"


def test_a_unit_seen_on_both_sides_is_held_out_whole_and_its_earlier_rows_are_counted(tmp_path):
    t, frame = table(tmp_path, grouped=True)
    split = drawn(t, state(CutPoint(column="cycle", at=2017), grouped=True))
    last = frame.groupby("pid")["cycle"].transform("max")
    expect = set(np.flatnonzero(last.to_numpy() >= 2017.0).tolist())
    assert held_rows(split) == expect
    a = split.frames["assignment"]
    sides = frame.loc[a["row_id"], "pid"].groupby(a["partition"].to_numpy()).unique()
    assert not set(sides["holdout"]) & set(sides["train"])  # no person on both sides
    earlier = int(((last >= 2017.0) & (frame["cycle"] < 2017.0)).sum())
    units = int(frame.loc[last >= 2017.0, "pid"].nunique())
    assert earlier == 20 and units == 20
    assert split.data["chronology"]["sentence"] == (
        f"Held out by a cut-point at `2017` on `cycle`: `{units}` units whole, `{len(expect)}` rows, "
        f"and `{60 - len(expect)}` rows train. A cut-point holds out every row at or after it, with "
        f"the earlier rows of any unit it holds out; the other rows train. Whole units are held "
        f"out, earlier rows included: `{earlier}` of the held-out rows were observed before "
        f"`2017`.")


# ── what is refused, and what is not ─────────────────────────────────────────


@pytest.mark.parametrize("cut, words", [
    (CutPoint(column="wave", at=2017), "no column by that name"),
    (CutPoint(column="cycle", at="soon"), "`soon` does not read as a number"),
    (CutPoint(column="cycle", at=2030), "nothing would be held out"),
    (CutPoint(column="cycle", at=2011), "no row would train"),
])
def test_a_cut_point_that_cannot_hold_rows_out_and_leave_some_to_train_refuses(tmp_path, cut, words):
    t, _ = table(tmp_path)
    with pytest.raises(ValueError, match=words):
        drawn(t, state(cut))


def test_the_split_answer_is_refused_where_the_draw_would_be(tmp_path):
    t, frame = table(tmp_path)
    with t.store() as store:
        ctx = {"columns": list(frame.columns), "target": "y", "task": "regression",
               "store": store, "state": ProjectState(target="y", task="regression")}
        body = {"kind": "set_split", "holdout": 0.0, "seed": 0, "fold_order": "random"}
        with pytest.raises(Refusal) as found:
            validate({**body, "cut_point": {"column": "cycle", "at": 2030}}, ctx)
        assert found.value.code == "cut_point_holds_nothing"
        with pytest.raises(Refusal) as found:
            validate({**body, "cut_point": {"column": "wave", "at": 2017}}, ctx)
        assert found.value.code == "unknown_column"
        validate({**body, "cut_point": {"column": "cycle", "at": 2017}}, ctx)  # silent


def test_a_cut_point_holds_out_no_share_as_well_and_answers_its_folds_order():
    cut = CutPoint(column="cycle", at=2017)
    with pytest.raises(ValidationError, match="holdout"):
        SplitSpec(holdout=0.2, cut_point=cut, fold_order="random")
    with pytest.raises(ValidationError, match="folds' order"):
        SplitSpec(holdout=0.0, cut_point=cut)
    with pytest.raises(ValidationError, match="cut-point"):
        SplitSpec(holdout=0.2, fold_order="random")  # no cut-point: the draw sets the folds
    with pytest.raises(ValidationError, match="internal"):
        SplitSpec(holdout=0.0, cut_point=cut, fold_order="time", validation="internal_external",
                  cluster="site")
    # Silent: each answer that holds together.
    SplitSpec(holdout=0.0, cut_point=cut, fold_order="time")
    SplitSpec(holdout=0.0, cut_point=cut, fold_order="random", validation="repeated_kfold")
    assert SplitSpec(holdout=0.2).cut_point is None and SplitSpec(holdout=0.2).fold_order is None


# ── the folds' order is its own answer ───────────────────────────────────────


def test_the_same_held_out_rows_take_random_repeated_folds_or_time_ordered_ones(tmp_path):
    t, frame = table(tmp_path)
    cut = CutPoint(column="cycle", at=2017)
    random = drawn(t, state(cut, fold_order="random", scheme={"validation": "repeated_kfold",
                                                              "repeats": 3}))
    timed = drawn(t, state(cut, fold_order="time"))
    assert held_rows(random) == held_rows(timed)  # the draw does not move
    assert random.data["fold_scheme"] == "random" and timed.data["fold_scheme"] == "time_ordered"
    assert random.data["repeats"] == 3 and {"fold_r1", "fold_r2"} <= set(random.frames["assignment"])
    # Time-ordered: each later fold's rows come after every row of the folds before it.
    a = timed.frames["assignment"]
    train = a[a["partition"] == "train"].assign(cycle=frame.loc[a.loc[a["partition"] == "train",
                                                                       "row_id"], "cycle"].to_numpy())
    spans = train.groupby("fold")["cycle"].agg(["min", "max"]).sort_index()
    assert (spans["max"].to_numpy()[:-1] <= spans["min"].to_numpy()[1:]).all()
    # Random folds mix the cycles: some fold holds rows from more than one.
    mixed = train.assign(fold=random.frames["assignment"].set_index("row_id").loc[
        train["row_id"], "fold"].to_numpy()).groupby("fold")["cycle"].nunique()
    assert mixed.max() > 1


def test_the_draw_by_share_still_orders_its_folds_by_time(tmp_path):
    """Silent: a chronological draw by share keeps time-ordered folds; only the cut-point frees
    them."""
    t, _ = table(tmp_path)
    s = state(None, holdout=0.2, temporal=TemporalSpec(temporal=True, time_column="cycle"))
    assert drawn(t, s).data["fold_scheme"] == "time_ordered"


def test_a_changed_folds_order_keeps_the_held_out_rows_and_is_the_schemes_answer(tmp_path):
    log = DecisionLog(tmp_path / "decisions.jsonl")
    cut = {"column": "cycle", "at": 2017.0}
    log.append(d.SetSplit(holdout=0.0, seed=0, cut_point=cut, fold_order="time"))
    log.append(d.SetValidation(validation="repeated_kfold", fold_order="random"))
    split = log.state().split
    assert split.cut_point == CutPoint(**cut) and split.fold_order == "random"
    assert split.validation == "repeated_kfold"
    # A scheme answer that says nothing of the folds' order keeps the one recorded.
    log.append(d.SetValidation(validation="kfold"))
    assert log.state().split.fold_order == "random" and log.state().split.cut_point is not None


def test_recorded_as_a_split_a_changed_folds_order_is_sent_to_the_scheme():
    cut = CutPoint(column="cycle", at=2017)
    now = ProjectState(target="y", split=SplitSpec(holdout=0.0, seed=0, cut_point=cut,
                                                    fold_order="time"))
    body = {"kind": "set_split", "holdout": 0.0, "seed": 0, "cut_point": cut.model_dump(),
            "fold_order": "random"}
    with pytest.raises(Refusal) as found:
        validate(body, {"state": now})
    assert found.value.code == "scheme_is_its_own"
    assert found.value.exits[0]["decision"]["kind"] == "set_validation"
    assert found.value.exits[0]["decision"]["fold_order"] == "random"
    assert found.value.exits[0]["decision"]["cut_point"] is None  # a scheme never names the draw
    # Silent: another cut-point is another draw, the split's own answer.
    validate({**body, "cut_point": {"column": "cycle", "at": 2015.0}}, {"state": now})


def test_a_scheme_written_from_the_split_keeps_its_cut_point_and_never_moves_it():
    """The nested cross-validation offer writes its scheme from the split's own fields
    (``stages.modeling``), the cut-point among them: that one parses and is accepted, and a
    scheme naming another cut-point is refused, since a scheme never moves the draw."""
    split = SplitSpec(holdout=0.0, seed=0, fold_order="time",
                      cut_point=CutPoint(column="cycle", at=2017))
    now = ProjectState(target="y", split=split)
    offer = {"kind": "set_validation", **split.model_dump(mode="json", exclude={"holdout", "seed"}),
             "nested_cv": True}
    validate(offer, {"state": now})
    log_state = d.fold([d.DecisionRecord(id="a", seq=1, at=d.datetime.now(d.timezone.utc),
                                         decision=d.SetSplit(**split.model_dump())),
                        d.DecisionRecord(id="b", seq=2, at=d.datetime.now(d.timezone.utc),
                                         decision=d.parse_decision(offer))])
    assert log_state.split.cut_point == split.cut_point and log_state.split.nested_cv
    with pytest.raises(Refusal) as found:
        validate({**offer, "cut_point": {"column": "cycle", "at": 2015}}, {"state": now})
    assert found.value.code == "scheme_keeps_the_draw"


def test_a_folds_order_needs_a_cut_point_to_answer_it():
    plain = ProjectState(target="y", split=SplitSpec(holdout=0.2, seed=0))
    with pytest.raises(Refusal) as found:
        validate({"kind": "set_validation", "fold_order": "random"}, {"state": plain})
    assert found.value.code == "no_cut_point"
    cut = ProjectState(target="y", split=SplitSpec(holdout=0.0, seed=0, fold_order="time",
                                                   cut_point=CutPoint(column="cycle", at=2017)))
    validate({"kind": "set_validation", "fold_order": "random"}, {"state": cut})
    validate({"kind": "set_validation"}, {"state": plain})


# ── what else reads whether rows are held out ────────────────────────────────


def test_a_cut_point_seal_holds_rows_out_and_can_be_opened():
    cut = ProjectState(target="y", split=SplitSpec(holdout=0.0, seed=0, fold_order="random",
                                                   cut_point=CutPoint(column="cycle", at=2017)))
    plain = ProjectState(target="y", split=SplitSpec(holdout=0.0, seed=0))
    assert seal.holds_rows_out(cut) and not seal.holds_rows_out(plain)
    validate({"kind": "open_seal"}, {"state": cut})  # nothing refuses without a fit to read
    with pytest.raises(Refusal) as found:
        validate({"kind": "open_seal"}, {"state": plain})
    assert found.value.code == "nothing_sealed"
    assert "cycle" in seal.draw_columns(cut) and "cycle" not in seal.draw_columns(plain)


def test_under_a_cut_point_the_temporal_answer_draws_nothing_again():
    cut = CutPoint(column="cycle", at=2017)
    now = ProjectState(target="y", split=SplitSpec(holdout=0.0, seed=0, fold_order="random",
                                                   cut_point=cut))
    after = now.model_copy(update={"temporal": TemporalSpec(temporal=True, time_column="cycle")})
    assert seal._redrawn_by(now, after, {}) is None
    share = ProjectState(target="y", split=SplitSpec(holdout=0.2, seed=0))  # fires without it
    assert "by time instead" in seal._redrawn_by(
        share, share.model_copy(update={"temporal": after.temporal}), {})


def test_once_opened_another_cut_point_moves_rows_and_a_new_seed_does_not():
    cut = CutPoint(column="cycle", at=2017)
    then = ProjectState(target="y", split=SplitSpec(holdout=0.0, seed=0, fold_order="random",
                                                    cut_point=cut))
    moved = then.model_copy(update={"split": then.split.model_copy(
        update={"cut_point": CutPoint(column="cycle", at=2015)})})
    clause = seal._membership_change(then, moved, {})
    assert clause == "holding out from `2015` on `cycle` instead of from `2017` would draw other rows"
    reseeded = then.model_copy(update={"split": then.split.model_copy(update={"seed": 4})})
    assert seal._membership_change(then, reseeded, {}) is None
    share = then.model_copy(update={"split": SplitSpec(holdout=0.2, seed=0)})
    assert seal._membership_change(then, share, {}) == (
        "holding out `20%` instead of the rows at or after `2017` would draw other rows")


# ── the preview ──────────────────────────────────────────────────────────────


def test_the_split_preview_says_the_rows_at_or_after_the_cut_point_are_held_out(tmp_path):
    from turbotab.core import fact_previews, row_previews  # noqa: F401 - registers the builders
    from turbotab.core.consequences import PreviewContext, plan
    from turbotab.core.datastore import DataStore, ingest

    _, frame = table(tmp_path)
    ingest(tmp_path / "cycles.csv", tmp_path / "cycles.parquet")
    s = state(None, holdout=0.2)
    with DataStore(tmp_path / "cycles.parquet", 2 << 30) as store:
        ctx = PreviewContext(project_id="p", state=s, datastore=store, artifact={}.get,
                             training_row_ids=None, cohort_row_ids=None, sealed_row_ids=None)
        cut = {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 3, "fold_order": "random",
               "cut_point": {"column": "cycle", "at": 2017}}
        flow = plan(d.parse_decision(cut), ctx, basis="b").views[0]
        share = plan(d.parse_decision({"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 3}),
                     ctx, basis="b").views[0]
    held = int((frame["cycle"] >= 2017).sum())
    assert flow.caption.startswith(f"`{held}` rows held out, at or after `2017` on `cycle`.")
    assert share.caption.startswith("`12` rows held out, by row.")  # silent: a share's words stay


# ── the methods sentence ─────────────────────────────────────────────────────


def test_the_split_sentence_says_what_the_cut_point_held_out():
    cut = {"column": "cycle", "at": 2017.0}
    text = voice.sentence_for(d.SetSplit(holdout=0.0, seed=3, folds=5, cut_point=cut,
                                         fold_order="random"),
                              ProjectState(target="glucose", task="regression"))
    assert text == ("A cut-point at `2017` on `cycle` held out every row at or after it for one "
                    "final score, with the earlier rows of any unit it held out; performance was "
                    "estimated on the rest by `5`-fold cross-validation; models were compared with "
                    "each other and with the no-predictor baseline on `5`-fold cross-validation "
                    "repeated `10` times, by the corrected repeated k-fold t (Nadeau & Bengio 2003; "
                    "Bouckaert & Frank 2004); R² was measured against the training rows' mean and "
                    "pooled over every out-of-fold prediction.")
    timed = voice.sentence_for(d.SetSplit(holdout=0.0, seed=3, folds=5, cut_point=cut,
                                          fold_order="time"),
                               ProjectState(target="glucose", task="regression"))
    inference = voice.sentence_for(d.SetSplit(holdout=0.0, seed=3, folds=5, cut_point=cut,
                                              fold_order="random"),
                                   ProjectState(target="glucose", task="regression",
                                                purpose="inference"))
    assert inference.startswith("A cut-point at `2017` on `cycle` held out every row")  # not "No rows"
    assert timed == ("A cut-point at `2017` on `cycle` held out every row at or after it for one "
                     "final score, with the earlier rows of any unit it held out; models were "
                     "compared on the rest by cross-validation over `5` time-ordered folds, each "
                     "scored by models fit on earlier units; R² was measured against the training "
                     "rows' mean and pooled over every out-of-fold prediction.")


def test_the_other_sentences_that_read_the_split_count_a_cut_point_as_rows_held_out(tmp_path):
    """A cut-point holds rows out with no share beside it: the revert that brings one back names
    it, the families' sentence says nothing of a choice made with no rows held out, and under
    inference the split's sentence still states its plan. Each stays as it was for a share."""
    log = DecisionLog(tmp_path / "decisions.jsonl")
    log.append(d.SetSplit(holdout=0.0, seed=0, cut_point={"column": "cycle", "at": 2017.0},
                          fold_order="random"))
    second = log.append(d.SetSplit(holdout=0.2, seed=0))
    text = voice.sentence_for(d.Revert(decision_id=second.id), log.state(),
                              {"records": log.records()})
    assert text == "Decision `#2` was reverted, so the split is a cut-point at `2017` on `cycle` again."
    third = log.append(d.SetSplit(holdout=0.0, seed=0))
    text = voice.sentence_for(d.Revert(decision_id=third.id), log.state(), {"records": log.records()})
    assert text == "Decision `#3` was reverted, so the split is a `20%` holdout again."

    cut = SplitSpec(holdout=0.0, seed=0, fold_order="random",
                    cut_point=CutPoint(column="cycle", at=2017))
    models = d.SelectModels(models=["linear", "random_forest"])
    with_cut = voice.sentence_for(models, ProjectState(target="y", task="regression", split=cut))
    with_none = voice.sentence_for(models, ProjectState(target="y", task="regression",
                                                        split=SplitSpec(holdout=0.0)))
    assert "with no rows held out" not in with_cut and "with no rows held out" in with_none

    inference = ProjectState(target="y", task="regression", purpose="inference")
    assert voice.split_states_plan(d.SetSplit(**cut.model_dump()), inference)
    assert not voice.split_states_plan(d.SetSplit(holdout=0.0), inference)

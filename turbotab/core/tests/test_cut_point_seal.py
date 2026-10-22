"""A temporal seal by cut-point (SWARM_PLAN §8 and §10's watch note on package 3).

Every row whose time column reads at or after a value is held out, a unit with rows on both sides
held out whole, and every other analyzed row trains; the training folds' order is its own answer,
random (repeated or not) or time-ordered, so the same held-out rows can be scored under either. Each
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
    # One row per unit: the sentence names no unit (the verifier's P14).
    assert chron["sentence"] == (
        "Held out by a cut-point at `2017` on `cycle`: `20` rows held out and `40` train. A "
        "cut-point holds out every row at or after it, and every other analyzed row trains.")
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
        f"the earlier rows of any unit it holds out, and every other analyzed row trains. Whole "
        f"units are held out, earlier rows included: `{earlier}` of the held-out rows were "
        f"observed before `2017`.")


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
                    "final score; performance was "
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
                     "final score; models were "
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


# ── repair round 1: the verifier's problems, each from the table its test writes ──


def _write(tmp_path, frame: pd.DataFrame, name: str = "cut.csv") -> Ingested:
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return Ingested(path, tmp_path)


def test_with_rows_excluded_the_seal_counts_the_analyzed_rows_it_holds_out_and_trains(tmp_path):
    """P1: the seal's counts are the analyzed rows', as the split's own are, so with an exclusion
    the note never says rows train that do not; rows with no readable time are counted the same
    way. Every count here is pandas' over the table written."""
    rng = np.random.default_rng(11)
    frame = pd.DataFrame({"cycle": np.repeat(CYCLES, 10), "x": rng.normal(size=60),
                          "age": np.tile([30, 30, 12], 20)})
    frame["y"] = frame["x"] + rng.normal(scale=0.5, size=60)
    frame.loc[[3, 4, 5], "cycle"] = np.nan  # three undated rows, one of them a child
    t = _write(tmp_path, frame)
    s = state(CutPoint(column="cycle", at=2017)).model_copy(update={
        "roles": {"x": "covariate", "age": "covariate"},
        "exclusions": [d.ExclusionRule(column="age", low=18, reason="adults")]})
    split = drawn(t, s)
    adult = frame["age"] >= 18
    held = int((adult & (frame["cycle"] >= 2017)).sum())
    train = int((adult & ~(frame["cycle"] >= 2017)).sum())
    undated = int((adult & frame["cycle"].isna()).sum())
    assert (held, train, undated) == (13, 27, 2)
    assert split.data["n_holdout"] == held and split.data["n_train"] == train
    sentence = (f"Held out by a cut-point at `2017` on `cycle`: `{held}` rows held out and "
                f"`{train}` train. A cut-point holds out every row at or after it, and every other "
                f"analyzed row trains. `{undated}` rows with no readable `cycle` train and are "
                f"never held out.")
    assert split.data["chronology"]["sentence"] == sentence
    assert split.data["note"].startswith(f"`{held}` rows are held out and `{train}` train;")
    assert sentence in split.data["note"]


def test_the_split_sentence_names_units_only_where_the_seal_keeps_units_whole():
    """P14: the methods sentence says a unit's earlier rows are held out only where the seal is
    grouped by a unit (the seal plan's basis); with one row per unit it names no unit."""
    split = d.SetSplit(holdout=0.0, seed=3, folds=5, cut_point={"column": "cycle", "at": 2017.0},
                       fold_order="random")
    s = ProjectState(target="glucose", task="regression", purpose="inference")
    grouped = {"seal_plan": {"basis": {"state": "grouped", "column": "pid", "source": "grain",
                                       "exploratory": False}}}
    single = {"seal_plan": {"basis": {"state": "one_row_per_unit", "column": None,
                                      "source": "grain", "exploratory": False}}}
    head = "A cut-point at `2017` on `cycle` held out every row at or after it for one final score"
    assert voice.sentence_for(split, s, grouped).startswith(
        f"{head}, with the earlier rows of any unit it held out; ")
    assert voice.sentence_for(split, s, single).startswith(f"{head}; ")


def test_a_grouped_preview_says_whole_units_are_held_out_with_their_earlier_rows(tmp_path):
    """P2: where a unit spans the cut-point the preview says each such unit is held out whole and
    counts its earlier rows; by row it says the rows at or after it, as before."""
    from turbotab.core import fact_previews, row_previews  # noqa: F401 - registers the builders
    from turbotab.core.consequences import PreviewContext, plan
    from turbotab.core.datastore import DataStore, ingest

    _, frame = table(tmp_path, grouped=True)
    ingest(tmp_path / "cycles.csv", tmp_path / "cycles.parquet")
    s = state(None, holdout=0.2, grouped=True)
    cut = {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 3, "fold_order": "random",
           "cut_point": {"column": "cycle", "at": 2017}}
    with DataStore(tmp_path / "cycles.parquet", 2 << 30) as store:
        ctx = PreviewContext(project_id="p", state=s, datastore=store, artifact={}.get,
                             training_row_ids=None, cohort_row_ids=None, sealed_row_ids=None)
        flow = plan(d.parse_decision(cut), ctx, basis="b").views[0]
    last = frame.groupby("pid")["cycle"].transform("max")
    held = int((last >= 2017).sum())
    earlier = int(((last >= 2017) & (frame["cycle"] < 2017)).sum())
    assert (held, earlier) == (40, 20)
    assert flow.caption.startswith(
        f"`{held}` rows held out: every unit seen at or after `2017` on `cycle`, whole "
        f"(`{earlier}` rows before it). About")


def test_time_ordered_folds_under_repeated_k_fold_are_said_to_run_once(tmp_path):
    """P3: time-ordered folds are the same in every repeat, so the draw runs them once, and the
    split's sentence says so; random folds under repeated k-fold are still said to repeat."""
    cut = {"column": "cycle", "at": 2017.0}
    s = ProjectState(target="glucose", task="regression")
    timed = voice.sentence_for(d.SetSplit(holdout=0.0, seed=3, folds=5, cut_point=cut,
                                          fold_order="time", validation="repeated_kfold",
                                          repeats=3), s)
    assert timed == ("A cut-point at `2017` on `cycle` held out every row at or after it for one "
                     "final score; models were compared on the rest by cross-validation over `5` "
                     "time-ordered folds, each scored by models fit on earlier units, run once, "
                     "since time-ordered folds are the same in every repeat; R² was measured "
                     "against the training rows' mean and pooled over every out-of-fold "
                     "prediction.")
    random = voice.sentence_for(d.SetSplit(holdout=0.0, seed=3, folds=5, cut_point=cut,
                                           fold_order="random", validation="repeated_kfold",
                                           repeats=3), s)
    assert "`5`-fold cross-validation repeated `3` times, each score the mean" in random
    t, _ = table(tmp_path)
    split = drawn(t, state(CutPoint(**cut), fold_order="time",
                           scheme={"validation": "repeated_kfold", "repeats": 3}))
    assert split.data["repeats"] == 1 and "fold_r1" not in split.frames["assignment"]


# ── the guards the verifier's mutations left untested (P9) ───────────────────


def _undated(tmp_path, n_undated: int) -> tuple[Ingested, pd.DataFrame]:
    """Fifty rows over five cycles, the first ``n_undated`` with no cycle."""
    rng = np.random.default_rng(5)
    frame = pd.DataFrame({"cycle": np.repeat([2011.0, 2013.0, 2015.0, 2017.0, 2019.0], 10),
                          "x": rng.normal(size=50)})
    frame["y"] = frame["x"] + rng.normal(scale=0.5, size=50)
    frame.loc[:n_undated - 1, "cycle"] = np.nan
    return _write(tmp_path, frame, f"undated{n_undated}.csv"), frame


def test_a_tenth_of_the_rows_undated_is_drawn_and_the_undated_rows_train(tmp_path):
    """Silent: 5 of 50 rows with no cycle is a tenth, no more, so the cut-point is drawn; the five
    train, never held out, and the seal counts them."""
    t, frame = _undated(tmp_path, 5)
    split = drawn(t, state(CutPoint(column="cycle", at=2017)))
    a = split.frames["assignment"].set_index("row_id")
    undated = frame.index[frame["cycle"].isna()]
    assert len(undated) == 5 and set(a.loc[undated, "partition"]) == {"train"}
    assert split.data["chronology"]["sentence"].endswith(
        "`5` rows with no readable `cycle` train and are never held out.")


def test_more_than_a_tenth_of_the_rows_undated_refuses_the_cut_point(tmp_path):
    """Fires: 6 of 50 rows with no cycle is more than a tenth, and any of them may fall at or after
    the cut-point, so the draw refuses, and so does the answer."""
    t, frame = _undated(tmp_path, 6)
    with pytest.raises(ValueError, match="`6` of `50` rows have no readable `cycle`"):
        drawn(t, state(CutPoint(column="cycle", at=2017)))
    with t.store() as store:
        ctx = {"columns": list(frame.columns), "target": "y", "store": store,
               "state": ProjectState(target="y", task="regression")}
        body = {"kind": "set_split", "holdout": 0.0, "seed": 0, "fold_order": "random",
                "cut_point": {"column": "cycle", "at": 2017}}
        with pytest.raises(Refusal) as found:
            validate(body, ctx)
        assert found.value.code == "cut_point_undated"
    t5, frame5 = _undated(tmp_path, 5)
    with t5.store() as store:
        validate(body, {**ctx, "columns": list(frame5.columns), "store": store})  # silent


def _spanning(tmp_path, before_only: int) -> tuple[Ingested, pd.DataFrame]:
    """Twenty people, each seen twice; the first ``before_only`` in 2011 and 2013, every other
    one in 2015 and again in 2019."""
    rng = np.random.default_rng(9)
    first = [2011.0 if i < before_only else 2015.0 for i in range(20)]
    second = [2013.0 if i < before_only else 2019.0 for i in range(20)]
    frame = pd.DataFrame({"pid": [f"p{i}" for i in range(20)] * 2, "cycle": first + second,
                          "x": rng.normal(size=40)})
    frame["y"] = frame["x"] + rng.normal(scale=0.5, size=40)
    return _write(tmp_path, frame, f"spanning{before_only}.csv"), frame


def test_a_cut_point_every_unit_spans_refuses_and_one_unit_before_it_trains(tmp_path):
    """Fires: every person has a row at or after 2017, so held out whole, no row would train,
    though rows before 2017 exist; silent: one person seen only before it trains, the rest held
    out."""
    t, _ = _spanning(tmp_path, 0)
    with pytest.raises(ValueError, match="Every unit with a readable `cycle` has a row at or"):
        drawn(t, state(CutPoint(column="cycle", at=2017), grouped=True))
    t1, frame = _spanning(tmp_path, 1)
    split = drawn(t1, state(CutPoint(column="cycle", at=2017), grouped=True))
    assert held_rows(split) == set(np.flatnonzero(frame["pid"].to_numpy() != "p0").tolist())


def test_a_cut_point_on_the_outcome_is_refused(tmp_path):
    """P10: a cut-point on an outcome column (the target, or a time to event's follow-up time)
    would choose the held-out rows by their outcome: refused at the answer and at the draw; one on
    a time column is not."""
    t, frame = table(tmp_path)
    body = {"kind": "set_split", "holdout": 0.0, "seed": 0, "fold_order": "random"}
    with t.store() as store:
        ctx = {"columns": list(frame.columns), "target": "y", "store": store,
               "state": ProjectState(target="y", task="regression")}
        with pytest.raises(Refusal) as found:
            validate({**body, "cut_point": {"column": "y", "at": 0.5}}, ctx)
        assert found.value.code == "cut_point_on_outcome"
        assert found.value.exits[0]["decision"]["cut_point"] is None
        survival = ProjectState(target="y", task="time_to_event",
                                follow_up=d.FollowUpSpec(time_column="x"))
        with pytest.raises(Refusal) as found:
            validate({**body, "cut_point": {"column": "x", "at": 0.0}}, {**ctx, "state": survival})
        assert found.value.code == "cut_point_on_outcome"
        validate({**body, "cut_point": {"column": "cycle", "at": 2017}}, ctx)  # silent
        validate({**body, "cut_point": {"column": "x", "at": 0.0}}, ctx)  # x is no outcome here
    with pytest.raises(ValueError, match="outcome"):
        drawn(t, state(CutPoint(column="y", at=0.5)))


# ── what else reads whether rows are held out (P4 to P7) ─────────────────────

CUT = SplitSpec(holdout=0.0, seed=0, folds=5, fold_order="random",
                cut_point=CutPoint(column="cycle", at=2017))
ALONE = SplitSpec(holdout=0.0, seed=0, folds=5)  # cross-validation alone


def _fit(*families: str) -> dict:
    return {"n_holdout": 30, "primary_metric": "rmse", "metric_labels": {"rmse": "RMSE"},
            "models": [{"family": f, "label": f, "cv": {"rmse": {"mean": 1.0 + i / 10}}}
                       for i, f in enumerate(families)]}


def test_under_a_cut_point_opening_the_seal_needs_the_final_model_named():
    """P4: with two families fitted and rows held out by a cut-point, an opening that names no
    final model is refused, one that names it is not, and with one family the opening declares it;
    under cross-validation alone neither acts (no held-out rows wait)."""
    from turbotab.core.models import selection

    for split, acts in ((CUT, True), (ALONE, False)):
        s = ProjectState(target="y", task="regression", purpose="prediction", split=split)
        two = {"state": s, "artifact": lambda stage: _fit("linear", "random_forest")
               if stage == "fit" else None}
        one = {"state": s, "artifact": lambda stage: _fit("linear") if stage == "fit" else None}
        if acts:
            with pytest.raises(Refusal) as found:
                selection._a_final_model_is_declared(d.OpenSeal(), two)
            assert found.value.code == "final_model_needed"
        else:
            selection._a_final_model_is_declared(d.OpenSeal(), two)
        selection._a_final_model_is_declared(d.OpenSeal(family="linear"), two)
        declared = selection._the_only_family_is_the_final_one(d.OpenSeal(), one)
        assert declared.family == ("linear" if acts else None)


def test_under_a_cut_point_the_opening_is_offered_and_a_compared_family_may_be_dropped():
    """P5: rows held out by a cut-point wait to be opened, so the Router's last step applies, and
    a family whose score was shown may be dropped (the held-out rows score the final model);
    under cross-validation alone the step does not apply and dropping it is refused."""
    from turbotab.core import interview

    seen = {"y": ["linear", "random_forest"]}
    drop = d.SelectModels(models=["linear"])
    for split, held in ((CUT, True), (ALONE, False)):
        s = ProjectState(target="y", task="regression", purpose="prediction",
                         models=["linear", "random_forest"], split=split)
        gate = interview._open_seal_gate(s)
        ctx = {"state": s, "seen": seen, "records": []}
        if held:
            assert gate is None
            validate(drop, ctx)
        else:
            assert gate is not None and gate[0] == "not_applicable"
            with pytest.raises(Refusal) as found:
                validate(drop, ctx)
            assert found.value.code == "compared_families_stay"


def test_under_a_cut_point_the_stepwise_room_counts_the_rows_the_split_drew_to_train():
    """P6: a cut-point's training rows are known once it is drawn, so the smallest training fold
    is (K − 1)/K of the split's own training count, by definition; until the split for that
    cut-point is drawn nothing is judged. A share and cross-validation alone keep their arithmetic
    on the cohort's count."""
    import math

    from turbotab.core.models.variable_selection import stepwise_room

    cols = ["g0", "g1", "g2"]
    info = {c: {"dtype": "numeric", "n_unique": 100, "n_missing": 0} for c in cols}
    cohort = {"n_final": 100, "predictors": cols}
    drawn_here = {"n_train": 70, "n_holdout": 30,
                  "chronology": {"drawn": True, "time_column": "cycle", "cut_point": "2017"}}
    elsewhere = {**drawn_here, "chronology": {**drawn_here["chronology"], "cut_point": "2015"}}

    def room(split, artifacts):
        s = ProjectState(target="y", task="regression", purpose="prediction",
                         roles={c: "covariate" for c in cols}, split=split)
        return stepwise_room(s, {"state": s, "column_info": info,
                                 "artifact": lambda stage: artifacts.get(stage)}, "regression")

    assert room(CUT, {"cohort": cohort, "split": drawn_here})[1] == math.floor(70 * 4 / 5)
    assert room(CUT, {"cohort": cohort}) is None  # not drawn yet
    assert room(CUT, {"cohort": cohort, "split": elsewhere}) is None  # drawn for another cut
    assert room(ALONE, {"cohort": cohort, "split": drawn_here})[1] == math.floor(100 * 4 / 5)
    share = SplitSpec(holdout=0.2, seed=0, folds=5)
    assert room(share, {"cohort": cohort})[1] == math.floor(math.floor(100 * 0.8) * 4 / 5)


def test_under_a_cut_point_the_temporal_answer_previews_that_it_draws_nothing(tmp_path):
    """P7: with a cut-point recorded the temporal answer draws no held-out rows (the cut-point
    holds them out), so its preview says so instead of a draw by time; with a share it still
    previews the latest units held out."""
    from turbotab.core import structure_previews  # noqa: F401 - registers the builders
    from turbotab.core.consequences import PreviewContext, plan
    from turbotab.core.datastore import DataStore, ingest

    table(tmp_path)
    ingest(tmp_path / "cycles.csv", tmp_path / "cycles.parquet")
    answer = d.SetTemporal(temporal=True, time_column="cycle")
    captions = {}
    with DataStore(tmp_path / "cycles.parquet", 2 << 30) as store:
        for name, s in (("cut", state(CutPoint(column="cycle", at=2017))),
                        ("share", state(None, holdout=0.2))):
            ctx = PreviewContext(project_id="p", state=s, datastore=store, artifact={}.get,
                                 training_row_ids=None, cohort_row_ids=None, sealed_row_ids=None)
            captions[name] = plan(answer, ctx, basis="b").views[0].caption
    assert captions["cut"] == ("A cut-point at `2017` on `cycle` holds the rows out, so this "
                               "answer draws none.")
    assert captions["share"].startswith("The latest `12` rows by their last `cycle` are held out")


# ── repair round 2: the verifier's problems ──────────────────────────────────

LEFT_OUT = ("are all left out by the answers that choose the analyzed rows, an exclusion or the "
            "missing-values answer among them. A cut-point is drawn only where some analyzed rows "
            "are held out and some before it train.")


def _excluding(tmp_path, rule):
    t, frame = table(tmp_path)
    return t, frame, state(CutPoint(column="cycle", at=2017)).model_copy(update={"exclusions": [rule]})


def test_an_exclusion_that_leaves_no_analyzed_row_at_or_after_the_cut_point_refuses_the_draw(tmp_path):
    """V2: among the rows with the outcome recorded the cut-point holds twenty out, but an
    exclusion of every cycle after 2016 leaves the analysis none of them: the draw refuses, naming
    the answers that leave them out, rather than a split that holds out no analyzed row (which
    every reader of ``decisions.holds_out`` would take for one that does). Counts by pandas."""
    t, frame, s = _excluding(tmp_path, d.ExclusionRule(column="cycle", high=2016, reason="early"))
    later = frame["cycle"] >= 2017
    kept = frame["cycle"] <= 2016
    assert int((later & kept).sum()) == 0 and int(later.sum()) == 20
    with pytest.raises(ValueError) as found:
        drawn(t, s)
    assert str(found.value) == (
        f"No analyzed row would be held out: the `{int(later.sum())}` rows with the outcome "
        f"recorded that the cut-point at `2017` on `cycle` would hold out {LEFT_OUT}")


def test_an_exclusion_that_leaves_no_analyzed_row_before_the_cut_point_refuses_the_draw(tmp_path):
    """V2, the other side: an exclusion of every cycle before 2017 leaves no analyzed row to train
    on, though forty rows with the outcome recorded would."""
    t, frame, s = _excluding(tmp_path, d.ExclusionRule(column="cycle", low=2017, reason="late"))
    earlier = frame["cycle"] < 2017
    assert int((earlier & (frame["cycle"] >= 2017)).sum()) == 0 and int(earlier.sum()) == 40
    with pytest.raises(ValueError) as found:
        drawn(t, s)
    assert str(found.value) == (
        f"No analyzed row with a readable `cycle` would train before `2017`: the "
        f"`{int(earlier.sum())}` rows with the outcome recorded there {LEFT_OUT}")


def test_complete_cases_that_leave_no_analyzed_row_held_out_refuse_the_draw(tmp_path):
    """V2 as NHANES meets it: a predictor the later cycle never recorded, under complete cases,
    drops every row the cut-point holds out."""
    rng = np.random.default_rng(5)
    frame = pd.DataFrame({"cycle": np.repeat(CYCLES, 10), "x": rng.normal(size=60)})
    frame["y"] = frame["x"] + rng.normal(scale=0.5, size=60)
    frame["z"] = np.where(frame["cycle"] >= 2017, np.nan, rng.normal(size=60))
    t = _write(tmp_path, frame)
    s = state(CutPoint(column="cycle", at=2017)).model_copy(update={
        "roles": {"x": "covariate", "z": "covariate"}})
    held = frame["cycle"] >= 2017
    assert int((held & frame["z"].notna()).sum()) == 0
    with pytest.raises(ValueError) as found:
        drawn(t, s)
    assert str(found.value) == (
        f"No analyzed row would be held out: the `{int(held.sum())}` rows with the outcome "
        f"recorded that the cut-point at `2017` on `cycle` would hold out {LEFT_OUT}")


def test_an_exclusion_that_thins_a_side_of_the_cut_point_is_drawn(tmp_path):
    """Silent: an exclusion of the earliest cycle leaves rows on both sides, so the cut-point is
    drawn, and its counts are the analyzed rows' (by pandas)."""
    t, frame, s = _excluding(tmp_path, d.ExclusionRule(column="cycle", low=2013, reason="later"))
    kept = frame["cycle"] >= 2013
    held, train = int((kept & (frame["cycle"] >= 2017)).sum()), int((kept & (frame["cycle"] < 2017)).sum())
    assert (held, train) == (20, 30)
    split = drawn(t, s)
    assert (split.data["n_holdout"], split.data["n_train"]) == (held, train)


# ── repair round 3: the verifier's problems ──────────────────────────────────


def _answered(t: Ingested, frame: pd.DataFrame, at) -> Refusal:
    """The refusal the split's answer gets for a cut-point at ``at`` on ``cycle``."""
    with t.store() as store:
        ctx = {"columns": list(frame.columns), "target": "y", "store": store,
               "state": ProjectState(target="y", task="regression")}
        body = {"kind": "set_split", "holdout": 0.0, "seed": 0, "fold_order": "random",
                "cut_point": {"column": "cycle", "at": at}}
        with pytest.raises(Refusal) as found:
            validate(body, ctx)
    return found.value


def _one_side_blank(cycles, blank) -> pd.DataFrame:
    """Twenty rows at each of ``cycles``, with ``y`` blank on the rows ``blank`` picks."""
    rng = np.random.default_rng(13)
    frame = pd.DataFrame({"cycle": np.repeat(cycles, 20), "x": rng.normal(size=20 * len(cycles))})
    frame["y"] = frame["x"] + rng.normal(scale=0.5, size=len(frame))
    frame.loc[blank(frame), "y"] = np.nan
    return frame


@pytest.mark.parametrize("cycles, blank, words", [
    # y is blank on every row at or after 2017: the table holds 20 such rows, none with y recorded.
    ((2011.0, 2017.0), lambda f: f["cycle"] >= 2017,
     "No row with `y` recorded has `cycle` at or after `2017`, so nothing would be held out."),
    # y is blank on every row before 2017: 20 rows train in the table, none with y recorded.
    ((2011.0, 2017.0), lambda f: f["cycle"] < 2017,
     "Every row with `y` recorded and a readable `cycle` is at or after `2017`, so no row would "
     "train before it."),
    # Every row with y recorded has no cycle; the rows with a cycle have y blank.
    ((np.nan, 2011.0, 2017.0), lambda f: f["cycle"].notna(),
     "No value of `cycle` reads as a time among the rows with `y` recorded, so no row can be held "
     "out at or after `2017`."),
])
def test_the_cut_points_refusals_name_the_rows_with_the_outcome_recorded(tmp_path, cycles, blank,
                                                                         words):
    """V6: the cut-point is judged over the rows with the outcome recorded, at the answer and at
    the draw, so where some row has none its refusal says so, rather than speak of every row of the
    column; pandas shows the table holds the rows today's words said it did not."""
    frame = _one_side_blank(cycles, blank)
    recorded = frame["y"].notna()
    dated = frame.loc[recorded, "cycle"].dropna()
    assert 0 < int(recorded.sum()) < len(frame)
    assert ((dated >= 2017).any(), (dated < 2017).any()) != (True, True)  # the check's own rows
    assert (frame["cycle"] >= 2017).any() and (frame["cycle"] < 2017).any()  # the table's
    t = _write(tmp_path, frame)
    assert _answered(t, frame, 2017).message == words
    with pytest.raises(ValueError) as found:
        drawn(t, state(CutPoint(column="cycle", at=2017)))
    assert str(found.value) == words


def test_with_every_outcome_recorded_the_cut_points_refusals_keep_the_columns_words(tmp_path):
    """V6, silent: where every row has the outcome recorded, the rows the check reads are every row
    of the column, and the refusals say so as before."""
    t, frame = table(tmp_path)
    assert frame["y"].notna().all()
    assert _answered(t, frame, 2030).message == (
        "No row of `cycle` is at or after `2030`, so nothing would be held out.")
    assert _answered(t, frame, 2011).message == (
        "Every row with a readable `cycle` is at or after `2011`, so no row would train before it.")


def test_the_undated_share_is_counted_on_the_rows_with_the_outcome_recorded(tmp_path):
    """V6: 6 of the 50 rows with y recorded have no cycle, more than a tenth; ten more rows have
    neither, and the refusal counts the 50 the check reads, naming them so (pandas counts)."""
    rng = np.random.default_rng(5)
    frame = pd.DataFrame({"cycle": np.repeat([2011.0, 2013.0, 2015.0, 2017.0, 2019.0, np.nan], 10),
                          "x": rng.normal(size=60)})
    frame["y"] = frame["x"] + rng.normal(scale=0.5, size=60)
    frame.loc[50:, "y"] = np.nan
    frame.loc[:5, "cycle"] = np.nan
    recorded = frame["y"].notna()
    undated, rows = int((recorded & frame["cycle"].isna()).sum()), int(recorded.sum())
    assert (undated, rows) == (6, 50) and int(frame["cycle"].isna().sum()) == 16
    words = (f"`{undated}` of `{rows}` rows with `y` recorded have no readable `cycle`: rows with "
             f"no readable time would train")
    t = _write(tmp_path, frame)
    found = _answered(t, frame, 2017)
    assert found.code == "cut_point_undated" and found.message.startswith(words)
    with pytest.raises(ValueError) as drew:
        drawn(t, state(CutPoint(column="cycle", at=2017)))
    assert str(drew.value).startswith(words)



def test_a_cut_point_every_unit_with_the_outcome_spans_names_the_rows_with_it(tmp_path):
    """V6, the draw's unit form: every person with y recorded is seen in 2015 and again in 2019, so
    held out whole none would train; ten people seen only in 2011 have y blank, so the refusal
    speaks of the rows with y recorded (pandas)."""
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({"pid": [f"p{i}" for i in range(20)] * 2 + [f"q{i}" for i in range(10)],
                          "cycle": [2015.0] * 20 + [2019.0] * 20 + [2011.0] * 10,
                          "x": rng.normal(size=50)})
    frame["y"] = frame["x"] + rng.normal(scale=0.5, size=50)
    frame.loc[40:, "y"] = np.nan
    recorded = frame["y"].notna()
    last = frame.loc[recorded].groupby("pid")["cycle"].max()
    assert (last >= 2017).all() and (frame.groupby("pid")["cycle"].max() < 2017).sum() == 10
    t = _write(tmp_path, frame)
    with pytest.raises(ValueError) as found:
        drawn(t, state(CutPoint(column="cycle", at=2017), grouped=True))
    assert str(found.value) == (
        "Every unit with a readable `cycle` among the rows with `y` recorded has a row at or after "
        "`2017`, so held out whole, no row would train before it.")


def _counted(before: int, after: int, undated: int = 0) -> pd.DataFrame:
    """``before`` rows in 2011, ``after`` in 2017 and ``undated`` with no cycle, y on each."""
    rng = np.random.default_rng(17)
    n = before + after + undated
    frame = pd.DataFrame({"cycle": [2011.0] * before + [2017.0] * after + [np.nan] * undated,
                          "x": rng.normal(size=n)})
    frame["y"] = frame["x"] + rng.normal(scale=0.5, size=n)
    return frame


ROWS = "A cut-point holds out every row at or after it, and every other analyzed row trains."


@pytest.mark.parametrize("before, after, undated, words", [
    (40, 1, 1, "`1` row held out and `41` train. {ROWS} `1` row with no readable `cycle` trains "
               "and is never held out."),
    (1, 40, 0, "`40` rows held out and `1` trains. {ROWS}"),
    (40, 2, 2, "`2` rows held out and `42` train. {ROWS} `2` rows with no readable `cycle` train "
               "and are never held out."),
])
def test_the_seal_sentence_agrees_with_its_counts_row_by_row(tmp_path, before, after, undated,
                                                             words):
    """V9: each count of one takes a singular noun and verb, and two or more the plural (the
    counts by construction, checked with pandas)."""
    frame = _counted(before, after, undated)
    held = int((frame["cycle"] >= 2017).sum())
    assert (held, len(frame) - held, int(frame["cycle"].isna().sum())) == (
        after, before + undated, undated)
    split = drawn(_write(tmp_path, frame), state(CutPoint(column="cycle", at=2017)))
    assert split.data["chronology"]["sentence"] == (
        "Held out by a cut-point at `2017` on `cycle`: " + words.format(ROWS=ROWS))


def test_one_undated_row_of_too_few_is_said_in_the_singular(tmp_path):
    """V9: one of eight rows with no cycle is more than a tenth, and the refusal says that one row
    has none; six of fifty (above) have none."""
    frame = _counted(4, 3, 1)
    assert int(frame["cycle"].isna().sum()) == 1 and 1 > 0.1 * len(frame) == 0.8
    found = _answered(_write(tmp_path, frame), frame, 2017)
    assert found.code == "cut_point_undated"
    assert found.message.startswith("`1` of `8` rows has no readable `cycle`: rows with no "
                                    "readable time would train")

UNITS = ("A cut-point holds out every row at or after it, with the earlier rows of any unit it "
         "holds out, and every other analyzed row trains.")


def _people(rows: list[tuple[str, float]]) -> pd.DataFrame:
    rng = np.random.default_rng(19)
    frame = pd.DataFrame(rows, columns=["pid", "cycle"])
    frame["x"] = rng.normal(size=len(frame))
    frame["y"] = frame["x"] + rng.normal(scale=0.5, size=len(frame))
    return frame


TWICE_BEFORE = [(f"q{i}", c) for i in range(20) for c in (2011.0, 2013.0)]  # 20 people, 40 rows
SPANNING = [(f"p{i}", c) for i in range(20) for c in (2015.0, 2019.0)]  # 20 people, both sides


@pytest.mark.parametrize("rows, counts, words", [
    (TWICE_BEFORE + [("p0", 2019.0)], (1, 1, 40, 0),
     "`1` unit whole, `1` row, and `40` rows train. {UNITS}"),
    (TWICE_BEFORE + [("p0", 2015.0), ("p0", 2019.0)], (1, 2, 40, 1),
     "`1` unit whole, `2` rows, and `40` rows train. {UNITS} Whole units are held out, earlier "
     "rows included: `1` of the held-out rows was observed before `2017`."),
    (SPANNING + [("q0", 2011.0)], (20, 40, 1, 20),
     "`20` units whole, `40` rows, and `1` row trains. {UNITS} Whole units are held out, earlier "
     "rows included: `20` of the held-out rows were observed before `2017`."),
    (SPANNING + TWICE_BEFORE + [("u0", np.nan)], (20, 40, 41, 20),
     "`20` units whole, `40` rows, and `41` rows train. {UNITS} Whole units are held out, earlier "
     "rows included: `20` of the held-out rows were observed before `2017`. `1` unit with no "
     "readable `cycle` trains and is never held out."),
])
def test_the_seal_sentence_agrees_with_its_counts_in_whole_units(tmp_path, rows, counts, words):
    """V9, grouped: a unit, a row and a held-out row observed before the cut-point, each counted
    once, take the singular. ``counts`` (units held out, their rows, the rows that train, the
    held-out rows before the cut-point) are pandas' over the table written."""
    frame = _people(rows)
    held = frame.groupby("pid")["cycle"].transform("max") >= 2017
    assert (int(frame.loc[held, "pid"].nunique()), int(held.sum()), int((~held).sum()),
            int((held & (frame["cycle"] < 2017)).sum())) == counts
    split = drawn(_write(tmp_path, frame), state(CutPoint(column="cycle", at=2017), grouped=True))
    assert split.data["chronology"]["sentence"] == (
        "Held out by a cut-point at `2017` on `cycle`: " + words.format(UNITS=UNITS))


LEFT_OUT_ONE = ("is left out by the answers that choose the analyzed rows, an exclusion or the "
                "missing-values answer among them. A cut-point is drawn only where some analyzed "
                "rows are held out and some before it train.")


def test_the_draws_refusal_agrees_with_one_row_left_out(tmp_path):
    """V9: where the one row with the outcome recorded on a side of the cut-point is the one an
    exclusion leaves out, the refusal says that row is left out; the plural is pinned by the
    repair-round-2 tests above (twenty and forty rows)."""
    rule = d.ExclusionRule(column="age", low=18, reason="adults")
    for before, after, side in ((40, 1, "held"), (1, 40, "train")):
        frame = _counted(before, after)
        child = frame["cycle"] >= 2017 if side == "held" else frame["cycle"] < 2017
        frame["age"] = np.where(child, 12, 30)
        assert int(child.sum()) == 1 and int((frame["age"] >= 18).sum()) == 40
        s = state(CutPoint(column="cycle", at=2017)).model_copy(update={
            "roles": {"x": "covariate", "age": "covariate"}, "exclusions": [rule]})
        with pytest.raises(ValueError) as found:
            drawn(_write(tmp_path, frame, f"{side}.csv"), s)
        assert str(found.value) == (
            "No analyzed row would be held out: the `1` row with the outcome recorded that the "
            f"cut-point at `2017` on `cycle` would hold out {LEFT_OUT_ONE}" if side == "held" else
            "No analyzed row with a readable `cycle` would train before `2017`: the `1` row with "
            f"the outcome recorded there {LEFT_OUT_ONE}")

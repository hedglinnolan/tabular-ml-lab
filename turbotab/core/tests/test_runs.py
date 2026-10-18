"""The record of every fit (SWARM_ENGINE_PLAN §2 E1; ``turbotab/core/runs.py``), from construction.

* The movement under Predict is ``|mean Δ| / SD(Δ)`` of the primary metric's fold scores, this
  fit's less the previous one's, computed here with numpy on the fold scores the test writes; it is
  measured only on the same folds, outcome and metric, and each case without one says why.
* The log keeps one record per fit key, in the order written, and never rewrites a line, however
  many threads write at once.
* Two fits share a folds digest exactly when they score the same training rows in the same folds.
"""
from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from turbotab.core import runs

AT = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)


def summary(folds: list[float | None]) -> dict:
    present = [f for f in folds if f is not None]
    sd = float(np.std(present, ddof=1)) if len(present) > 1 else None
    return {"mean": float(np.mean(present)), "sd": sd, "folds": folds}


def record(scores: dict[str, list[float | None]], **fields) -> runs.FitRecord:
    base = dict(id="a" * 32, seq=1, at=AT, purpose="prediction", target="hba1c", fit_key="k1",
                plan_digest=None, decisions=[1], press=None, families=list(scores), seconds=1.0,
                workers=2, primary_metric="rmse", folds="f1", seed=0,
                scores={f: {"rmse": summary(v)} for f, v in scores.items()})
    base.update(fields)
    return runs.FitRecord(**base)


def measured(previous: runs.FitRecord, scores: dict[str, list[float | None]], **fields):
    args = dict(purpose="prediction", target="hba1c", metric="rmse", folds="f1", seed=0)
    args.update(fields)
    return runs.movement(previous, scores={f: {"rmse": summary(v)} for f, v in scores.items()},
                         **args)


def test_the_movement_is_paired_folds_on_the_two_fits_fold_scores():
    before = [0.81, 0.77, 0.90, 0.84, 0.79]
    after = [0.78, 0.76, 0.85, 0.83, 0.75]
    previous = record({"linear": before, "boosted_trees": [0.7, 0.71, 0.69, 0.72, 0.7]})
    found, why = measured(previous, {"linear": after})
    assert why is None and [m.family for m in found] == ["linear"]
    d = np.asarray(after) - np.asarray(before)  # this fit's less the previous one's
    expected = abs(d.mean()) / d.std(ddof=1)
    m = found[0].movement
    assert found[0].against == previous.id and found[0].why is None
    assert m.instrument == "paired_folds" and m.regime == "realized"
    assert m.value == pytest.approx(expected, rel=1e-12)
    assert f"{d.mean():+.4f}" in m.words and f"{expected:.2f}" in m.words
    assert m.words.startswith("For `linear`, what differs from the previous fit changed")


def test_identical_fold_scores_move_by_exactly_zero():
    scores = [0.81, 0.77, 0.90, 0.84, 0.79]
    found, why = measured(record({"linear": scores}), {"linear": list(scores)})
    assert why is None and found[0].movement.value == 0.0 and found[0].movement.band == 0


def test_a_fold_scored_in_only_one_fit_is_left_out_of_the_pairs():
    """Time-ordered folds leave the first fold unscored: only folds both fits scored are paired."""
    before = [None, 0.77, 0.90, 0.84, 0.79]
    after = [None, 0.76, 0.85, 0.80, 0.75]
    found, _ = measured(record({"linear": before}), {"linear": after})
    d = np.asarray(after[1:]) - np.asarray(before[1:])
    assert found[0].movement.value == pytest.approx(abs(d.mean()) / d.std(ddof=1), rel=1e-12)


@pytest.mark.parametrize("fields, why", [
    ({"folds": "f2"}, runs.OTHER_FOLDS),
    ({"folds": "f2", "seed": 1},
     "The previous fit's folds were drawn with seed 0, this one's with seed 1."),
    ({"folds": None}, runs.FOLDS_UNREAD),
    ({"target": "glucose"}, "The previous fit predicts `hba1c`, not `glucose`."),
    ({"metric": "log_loss"}, "The previous fit is scored by `rmse`, this one by `log_loss`."),
    ({"purpose": "inference"}, runs.NOT_PREDICT),
])
def test_no_movement_is_measured_across_different_folds_and_it_says_why(fields, why):
    found, said = measured(record({"linear": [0.8, 0.7, 0.9]}), {"linear": [0.7, 0.7, 0.8]}, **fields)
    assert found == [] and said == why


def test_the_first_fit_and_one_after_an_estimate_carry_no_movement():
    assert measured(None, {"linear": [0.8, 0.7, 0.9]}) == ([], runs.NO_PREVIOUS)
    # under Estimate the reason is the purpose, first or not
    assert measured(None, {}, purpose="inference") == ([], runs.NOT_PREDICT)
    # a fit no record prefix keys: its purpose is not known, nor its folds
    assert measured(record({"linear": [0.8, 0.7, 0.9]}), {}, purpose=None) == (
        [], runs.ANSWERS_UNREAD)
    estimate = record({}, purpose="inference", primary_metric=None)
    assert measured(estimate, {"linear": [0.8, 0.7, 0.9]}) == ([], runs.PREVIOUS_NOT_PREDICT)


def test_a_family_not_fitted_before_or_scored_in_too_few_folds_says_so():
    previous = record({"linear": [0.8, 0.7, 0.9], "elastic_net": [None, None, 0.6]})
    found, why = measured(previous, {"linear": [0.7, 0.7, 0.8], "boosted_trees": [0.6, 0.6, 0.7],
                                     "elastic_net": [None, 0.5, 0.6]})
    by = {m.family: m for m in found}
    assert why is None and by["linear"].movement is not None
    assert by["boosted_trees"].movement is None
    assert by["boosted_trees"].why == "`boosted_trees` was not fitted in the previous fit."
    assert by["elastic_net"].movement is None
    assert by["elastic_net"].why == "Fewer than two folds scored `elastic_net` in both fits."


def test_a_change_with_no_fold_to_fold_spread_is_refused_not_infinite():
    before = [0.5, 0.75, 0.25]
    after = [b + 0.125 for b in before]  # every difference exactly 0.125 (binary fractions)
    found, why = measured(record({"linear": before}), {"linear": after})
    assert why is None and found[0].movement is None
    assert found[0].why == ("Every fold scored `linear` +0.1250 from the previous fit, so the change "
                            "has no fold-to-fold spread and the paired measure is not defined.")


# ── the log ──────────────────────────────────────────────────────────────────


def fields(key: str, **more) -> dict:
    out = record({"linear": [0.8, 0.7]}).model_dump()
    out.pop("id"), out.pop("seq")
    out["fit_key"] = key
    out.update(more)
    return out


def test_the_log_keeps_one_record_per_fit_key_and_never_rewrites_a_line(tmp_path):
    log = runs.RunLog(tmp_path / runs.RUNS_FILE)
    first = log.append(lambda existing: fields("k1"))
    assert first is not None and first.seq == 1
    text = (tmp_path / runs.RUNS_FILE).read_bytes()
    assert log.append(lambda existing: fields("k1", seconds=99.0)) is None  # the key is recorded
    assert log.append(lambda existing: None) is None
    seen: list[list[str]] = []

    def second(existing):
        seen.append([r.fit_key for r in existing])
        return fields("k2")

    two = log.append(second)
    assert seen == [["k1"]] and two.seq == 2 and two.id != first.id
    after = (tmp_path / runs.RUNS_FILE).read_bytes()
    assert after.startswith(text)  # the first line, byte for byte
    assert [r.fit_key for r in log.records()] == ["k1", "k2"]
    assert log.records()[0] == first and log.records()[0].seconds == 1.0


def test_writers_at_once_record_each_key_once(tmp_path):
    path = tmp_path / runs.RUNS_FILE
    keys = [f"k{i % 5}" for i in range(40)]
    barrier = threading.Barrier(8)

    def write(chunk):
        log = runs.RunLog(path)  # a log of its own, as two services would hold
        barrier.wait()
        for key in chunk:
            log.append(lambda existing, key=key: fields(key))

    threads = [threading.Thread(target=write, args=(keys[i::8],)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    found = runs.RunLog(path).records()
    assert sorted(r.fit_key for r in found) == [f"k{i}" for i in range(5)]
    assert [r.seq for r in found] == [1, 2, 3, 4, 5]


def test_a_torn_last_line_is_skipped_and_never_glued_onto(tmp_path):
    path = tmp_path / runs.RUNS_FILE
    log = runs.RunLog(path)
    log.append(lambda existing: fields("k1"))
    with open(path, "ab") as fh:
        fh.write(b'{"id": "torn')
    assert [r.fit_key for r in log.records()] == ["k1"]
    log.append(lambda existing: fields("k2"))
    assert [r.fit_key for r in log.records()] == ["k1", "k2"]


# ── which records keyed a fit ────────────────────────────────────────────────


class Rec:
    def __init__(self, seq: int, at: datetime):
        self.seq, self.at = seq, at


def test_the_records_by_a_time_and_the_prefix_that_keys_a_fit(monkeypatch):
    rows = [Rec(i, AT + timedelta(seconds=i)) for i in (1, 2, 3, 4)]
    assert [r.seq for r in runs.appended_by(rows, AT + timedelta(seconds=2))] == [1, 2]
    assert [r.seq for r in runs.appended_by(rows, None)] == [1, 2, 3, 4]

    from turbotab.core import decisions

    monkeypatch.setattr(decisions, "fold", lambda prefix: prefix)  # a prefix stands for its state
    keyed = runs.keyed_prefix(rows, "k", lambda prefix: "k" if len(prefix) == 3 else "other")
    assert [r.seq for r in keyed] == [1, 2, 3]  # the longest prefix that keys it
    assert runs.keyed_prefix(rows, "k", lambda prefix: "k") == rows
    assert runs.keyed_prefix(rows, "k", lambda prefix: "other") is None


# ── the folds ────────────────────────────────────────────────────────────────


def assignment(folds: list[float], train: list[bool], ids: list[int] | None = None) -> pd.DataFrame:
    ids = ids or list(range(len(folds)))
    return pd.DataFrame({"train": train, "fold": folds}, index=pd.Index(ids, name="row_id"))


def test_two_fits_share_a_folds_digest_exactly_when_they_share_their_folds():
    train = [True, True, True, True, False]
    a = assignment([0, 1, 0, 1, float("nan")], train)
    assert runs.folds_digest(a) == runs.folds_digest(a.copy())
    # a held-out row's fold is not a fold any score reads
    assert runs.folds_digest(a) == runs.folds_digest(assignment([0, 1, 0, 1, 3.0], train))
    assert runs.folds_digest(a) != runs.folds_digest(assignment([1, 1, 0, 1, float("nan")], train))
    assert runs.folds_digest(a) != runs.folds_digest(
        assignment([0, 1, 0, 1, float("nan")], train, ids=[0, 1, 2, 5, 4]))
    assert runs.folds_digest(a) != runs.folds_digest(a, scheme="time_ordered")
    repeated = a.assign(fold_r1=[1, 0, 1, 0, float("nan")])
    assert runs.folds_digest(a) != runs.folds_digest(repeated)

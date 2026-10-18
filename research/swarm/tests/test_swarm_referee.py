"""The referee audit's sampler and prompts (DRIVER_PLAN D5): a seed draws the same sample from the
same corpus, only attempts with a reason are drawn, none twice, and each prompt holds the attempt's
reasons, the engine's numbers and words, and the analyst's later reasons, with every blank filled
and no word of whether the dataset is a control. Nothing here runs a referee."""
from __future__ import annotations

import copy
import json

from research.swarm import referee
from research.swarm.record import validate
from research.swarm.tests.constructed import predict_record


def corpus(n: int):
    """``n`` copies of the Predict record under their own run ids: 2 attempts each, both with a
    reason; and one run whose second attempt has none."""
    out = []
    for i in range(n):
        data = copy.deepcopy(predict_record())
        data["run"]["id"] = f"run-{i:02d}"
        out.append(validate(data))
    quiet = copy.deepcopy(predict_record())
    quiet["run"]["id"] = "run-quiet"
    quiet["attempts"][1]["reasons"] = []
    out.append(validate(quiet))
    return out


def test_the_pool_is_every_attempt_with_a_reason():
    records = corpus(3)
    assert referee.pool(records) == [("run-00", 1), ("run-00", 2), ("run-01", 1), ("run-01", 2),
                                     ("run-02", 1), ("run-02", 2), ("run-quiet", 1)]


def test_a_seed_draws_the_same_sample_and_another_seed_another():
    records = corpus(25)  # a pool of 51
    first = referee.sample(records, 10, seed=7)
    assert first == referee.sample(list(reversed(records)), 10, seed=7)
    assert len(set(first)) == 10 and set(first) <= set(referee.pool(records))
    assert ("run-quiet", 2) not in first
    assert referee.sample(records, 10, seed=8) != first


def test_a_sample_larger_than_the_pool_is_the_whole_pool():
    records = corpus(1)
    assert sorted(referee.sample(records, 50, seed=1)) == referee.pool(records)


def test_a_prompt_shows_the_reason_the_numbers_the_movement_and_the_later_reading():
    data = predict_record("null_decoy")
    data["decisions"][4]["reason"] = "The elastic net scored no better, so I open the seal."
    rec = validate(data)
    text = referee.prompt(rec, 2)
    assert "Record #4, `select_models`" in text and "a reason" in text  # what changed, and why
    assert "cross-validated auc 0.74 (interval 0.69 to 0.79, 5 folds)" in text
    assert "For `linear`, what differs moved the score by 0.4." in text  # the engine's words
    assert "`elastic_net` was not fitted in the previous fit." in text
    assert "The elastic net scored no better, so I open the seal." in text  # the reading
    assert "$" not in text and "decoy" not in text and "control" not in text
    first = referee.prompt(rec, 1)
    assert "The elastic net scored no better" not in first  # recorded after fit 2, not fit 1


def test_write_keeps_each_prompt_and_an_index_of_the_draw(tmp_path):
    records = corpus(4)
    index = referee.write(records, 3, seed=11, out=tmp_path)
    body = json.loads(index.read_text("utf-8"))
    assert body["seed"] == 11 and body["pool"] == 9 and len(body["samples"]) == 3
    drawn = [(s["run"], s["fit"]) for s in body["samples"]]
    assert drawn == referee.sample(records, 3, seed=11)
    for s in body["samples"]:
        assert (tmp_path / s["prompt"]).read_text("utf-8").startswith("# Referee")

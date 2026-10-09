"""The decision log's format, its migrations and its tombstones (V2X_SEAMS, seam guards 1 and 2).

``fixtures/decision_logs/`` is the corpus of logs written by earlier engines; every file there must
keep loading. ``format_0_causal_learners.jsonl`` was written by ``DecisionLog.append`` at
turbotab-next 9df23628, before the log recorded a format: a causal answer with the lane's forest
under its old key ``random_forest``, then one with its boosted trees under ``boosted_trees``. The
numbers it replays to (:data:`BEFORE_THE_RENAME`) were fitted by that engine from the same log.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Literal, get_args

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from turbotab.core import decisions as d
from turbotab.core.models import causal as est

CORPUS = Path(__file__).parent / "fixtures" / "decision_logs"
FORMAT_0 = CORPUS / "format_0_causal_learners.jsonl"
# (learner as recorded, method) -> (estimate, SE), fitted at 9df23628 from FORMAT_0 and table().
BEFORE_THE_RENAME = {
    ("random_forest", "dml_irm"): (-3.9991971766703944, 1.406598046214111),
    ("boosted_trees", "dml_plr"): (-4.160842018566974, 0.7933379957626656),
}


def _copy(tmp_path: Path, source: Path = FORMAT_0) -> Path:
    path = tmp_path / "decisions.jsonl"
    shutil.copyfile(source, path)
    return path


def _causal_learners(records: list[d.DecisionRecord]) -> list[str]:
    return [r.decision.learner for r in records if r.decision.kind == "set_causal"]


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ── format 0 still loads, renamed in memory only ────────────────────────────────


@pytest.mark.parametrize("source", sorted(CORPUS.glob("*.jsonl")), ids=lambda p: p.name)
def test_every_log_in_the_corpus_loads(tmp_path, source):
    assert d.DecisionLog(_copy(tmp_path, source)).records()


def test_a_format_0_log_reads_its_learners_by_their_new_keys_and_its_file_is_untouched(tmp_path):
    path = _copy(tmp_path)
    before = path.read_bytes()
    assert all(d.LOG_FORMAT_KEY not in line for line in _lines(path))  # format 0: no marker
    records = d.DecisionLog(path).records()
    assert _causal_learners(records) == ["nuisance_forest", "untuned_boosted_trees"]
    assert d.fold(records).causal.learner == "untuned_boosted_trees"
    assert path.read_bytes() == before


def test_a_format_0_log_folds_to_the_state_of_the_same_answers_recorded_today(tmp_path):
    old = d.DecisionLog(_copy(tmp_path)).records()
    new = d.DecisionLog(tmp_path / "today.jsonl")
    for record in old:
        new.append(record.decision)
    assert all(line[d.LOG_FORMAT_KEY] == d.LOG_FORMAT for line in _lines(new.path))
    for upto in (len(old) - 1, len(old)):
        assert d.fold(old[:upto]) == d.fold(new.records()[:upto])


def test_a_line_appended_to_a_format_0_log_records_the_format_and_the_old_lines_stay(tmp_path):
    path = _copy(tmp_path)
    before = path.read_bytes()
    log = d.DecisionLog(path)
    log.append(d.SetCausal(exposure="supplement", method="dml_irm", learner="nuisance_forest",
                           folds=3, repetitions=1, assumptions=["positivity"]))
    assert path.read_bytes().startswith(before)
    last = _lines(path)[-1]
    assert last[d.LOG_FORMAT_KEY] == d.LOG_FORMAT and next(iter(last)) == d.LOG_FORMAT_KEY
    assert _causal_learners(d.DecisionLog(path).records()) == [
        "nuisance_forest", "untuned_boosted_trees", "nuisance_forest"]


# ── every fitted number is unchanged ────────────────────────────────────────────


def table(n: int = 400, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.normal(50, 10, n)
    smoker = (rng.random(n) < 0.3).astype(int)
    p = 1 / (1 + np.exp(-(-0.5 + 0.03 * (age - 50) + 0.6 * smoker)))
    supplement = (rng.random(n) < p).astype(int)
    sbp = 120 + 0.5 * (age - 50) + 4 * smoker - 3 * supplement + rng.normal(0, 8, n)
    return pd.DataFrame({"person_id": np.arange(n), "age": age, "smoker": smoker,
                         "supplement": supplement, "sbp": sbp})


def _estimates(run, records: list[d.DecisionRecord]) -> list[tuple[float, float]]:
    out = run.public(run.run(d.fold(records), upto=["causal"]))["causal"]
    assert out["withheld"] is None
    return [(e["estimate"], e["se"]) for e in out["estimates"]]


def test_a_format_0_log_replays_to_the_numbers_the_old_engine_fitted(tmp_path):
    """The causal stage on the migrated log against the pre-rename engine's numbers. The same
    answers recorded today fold to the same state (the test above), so they fit these numbers too."""
    from turbotab.core.tests.graph_runner import GraphRun

    table().to_csv(tmp_path / "t.csv", index=False)
    old = d.DecisionLog(_copy(tmp_path)).records()
    run = GraphRun(tmp_path / "t.csv", tmp_path / "project")
    try:
        for upto, key in ((len(old) - 1, ("random_forest", "dml_irm")),
                          (len(old), ("boosted_trees", "dml_plr"))):
            [replayed] = _estimates(run, old[:upto])
            assert replayed == pytest.approx(BEFORE_THE_RENAME[key], rel=1e-9, abs=0)
    finally:
        run.close()


# ── the rename round trip ───────────────────────────────────────────────────────


def test_a_renamed_value_round_trips_through_the_current_format():
    raw = _lines(FORMAT_0)[-1]
    migrated = d.read_record(raw)
    assert migrated.decision.learner == "untuned_boosted_trees"
    again = d.read_record(json.loads(d.record_line(migrated)))
    assert again == migrated
    assert json.loads(d.record_line(again))["decision"]["learner"] == "untuned_boosted_trees"
    assert raw["decision"]["learner"] == "boosted_trees"  # the migration copied, never mutated


def test_a_locked_plan_keeps_the_learner_it_quoted():
    """A lock's plan quotes the plan as declared, with a SHA-256 that may be registered outside the
    software: the migration renames answers, never a quotation."""
    plan = {"causal": {"exposure": "supplement", "method": "dml_irm", "learner": "random_forest"}}
    raw = {"id": "a", "seq": 1, "at": "2026-10-01T00:00:00Z",
           "decision": {"kind": "lock_plan", "plan": plan, "digest": "f" * 64}}
    assert d.read_record(raw).decision.plan == plan


def test_the_migrations_and_the_tombstones_agree_on_the_format():
    assert d.LOG_FORMAT == len(d.LOG_MIGRATIONS) >= 1
    assert all(1 <= stone.since <= d.LOG_FORMAT for stone in d.TOMBSTONES)
    assert est.LEARNERS == get_args(d.CausalLearner) == tuple(est.LEARNER_WORDS) == tuple(
        est.LEARNER_NAMES)


# ── a newer format is refused ───────────────────────────────────────────────────


def test_a_line_from_a_newer_format_is_refused_and_the_file_is_untouched(tmp_path):
    path = _copy(tmp_path)
    newer = dict(_lines(path)[0], id="next", seq=99, format=d.LOG_FORMAT + 1)
    with open(path, "a") as f:
        f.write(json.dumps(newer) + "\n")
    before = path.read_bytes()
    with pytest.raises(d.LogFormatError, match=f"format {d.LOG_FORMAT + 1}, newer") as refused:
        d.DecisionLog(path).records()
    assert f"{path}:17" in str(refused.value)
    assert path.read_bytes() == before


@pytest.mark.parametrize("marker", ["1", True, -1, 1.0, None])
def test_a_malformed_format_is_refused(marker):
    raw = dict(_lines(FORMAT_0)[0], format=marker)
    with pytest.raises(d.LogFormatError):
        d.read_record(raw)


# ── a retired value is never recorded again ─────────────────────────────────────


@pytest.mark.parametrize("old,new", [("random_forest", "nuisance_forest"),
                                     ("boosted_trees", "untuned_boosted_trees")])
def test_a_retired_learner_is_refused_for_new_records(tmp_path, old, new):
    body = {"kind": "set_causal", "exposure": "supplement", "method": "dml_plr", "learner": old}
    with pytest.raises(ValidationError, match=f"`{old}` is a retired .*now `{new}`"):
        d.parse_decision(body)
    with pytest.raises(ValidationError, match="retired"):
        d.CausalSpec(**{k: v for k, v in body.items() if k != "kind"})
    log = d.DecisionLog(tmp_path / "decisions.jsonl")
    with pytest.raises(ValidationError):
        log.append(body)
    assert log.records() == []
    # A line that claims the current format is never migrated, so it cannot carry the old key.
    current = dict(_lines(FORMAT_0)[-1], format=d.LOG_FORMAT)
    current["decision"] = dict(current["decision"], learner=old)
    with pytest.raises(ValidationError, match="retired"):
        d.read_record(current)
    with pytest.raises(ValueError, match=f"now `{new}`"):
        est.make_learner(old, classifier=False)


def test_a_retired_value_can_never_come_back_to_its_vocabulary(monkeypatch):
    d._tombstones_stay_buried()
    reused = Literal["linear", "lasso", "nuisance_forest", "untuned_boosted_trees",
                     "random_forest"]
    monkeypatch.setitem(d.TOMBSTONE_VOCABULARIES, "causal_learner", reused)
    with pytest.raises(RuntimeError, match="never reused"):
        d._tombstones_stay_buried()
    dropped = Literal["linear", "lasso", "untuned_boosted_trees"]
    monkeypatch.setitem(d.TOMBSTONE_VOCABULARIES, "causal_learner", dropped)
    with pytest.raises(RuntimeError, match="does not hold"):
        d._tombstones_stay_buried()

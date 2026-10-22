"""Replay's rebuilds for the swarm (``core/export/replay.py``; SWARM_PLAN §6–§8): the log truncated
before the opening or the lock, the findings rule on a fresh draw, and the fresh draw's generator
check, on decision logs written here by the log's own appender and on tables whose bytes are
hashed here with hashlib.

* :func:`replay.truncate` keeps every record before the first live ``open_seal`` (under inference
  ``lock_plan``), each kept line the log's own bytes; a withdrawn lock is passed over for the lock
  that holds; a log with no such record comes back whole; the kind is read from the purpose.
* :func:`replay.findings_rule` keeps a record tied to a finding the new bytes raise, and leaves out
  one whose finding they do not raise, with every revert of it; nothing else moves.
* :func:`replay.findings_inputs` keeps the records the findings stage reads and no finding's.
* :func:`replay.fresh_table` refuses a generator that does not write the recorded bytes at the
  recorded seed, the recorded seed given as the new one, and a record with joined files, and
  writes the new draw, whose SHA-256 is hashlib's of the generator's own CSV.

The engine-level rebuilds on the swarm's fixtures are ``research/swarm/tests/test_replay_fixtures.py``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbotab.core.decisions import DecisionLog
from turbotab.core.export import replay


def _log(tmp_path: Path, decisions: list[dict], name: str = "decisions.jsonl") -> bytes:
    log = DecisionLog(tmp_path / name)
    for d in decisions:
        if d.get("kind") == "revert" and isinstance(d.get("decision_id"), int):
            d = {**d, "decision_id": log.records()[d["decision_id"] - 1].id}
        log.append(d)
    return (tmp_path / name).read_bytes()


PREDICT = [
    {"kind": "set_lens", "lenses": ["clinical"]},
    {"kind": "set_target", "column": "y"},
    {"kind": "set_purpose", "purpose": "prediction"},
    {"kind": "select_models", "models": ["linear"]},
    {"kind": "open_seal", "family": "linear"},
    {"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5},
]


def test_truncation_keeps_every_record_before_the_opening_as_its_own_bytes(tmp_path):
    log = _log(tmp_path, PREDICT)
    lines = log.split(b"\n")
    cut, where = replay.truncate(log)
    assert where == replay.Cut(kind="open_seal", seq=5, kept=4, left=2)
    assert cut == b"".join(line + b"\n" for line in lines[:4])
    assert [json.loads(x)["decision"]["kind"] for x in cut.splitlines()] == [
        "set_lens", "set_target", "set_purpose", "select_models"]


def test_a_log_with_no_opening_comes_back_whole(tmp_path):
    log = _log(tmp_path, [d for d in PREDICT if d["kind"] != "open_seal"])
    cut, where = replay.truncate(log)
    assert cut == log and where == replay.Cut(kind="open_seal", seq=None, kept=5, left=0)


def test_under_inference_the_cut_is_before_the_lock_that_holds_not_a_withdrawn_one(tmp_path):
    log = _log(tmp_path, [
        {"kind": "set_target", "column": "y"},
        {"kind": "set_purpose", "purpose": "inference"},
        {"kind": "lock_plan"},
        {"kind": "revert", "decision_id": 3},  # the server withdrew the lock: no estimate seen
        {"kind": "select_models", "models": ["linear"]},
        {"kind": "lock_plan"},
        {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5},
    ])
    cut, where = replay.truncate(log)
    assert where == replay.Cut(kind="lock_plan", seq=6, kept=5, left=2)
    assert [json.loads(x)["decision"]["kind"] for x in cut.splitlines()] == [
        "set_target", "set_purpose", "lock_plan", "revert", "select_models"]
    # Named, the other kind cuts where it first holds; a kind that is neither is refused.
    assert replay.truncate(log, "open_seal")[1].seq is None
    with pytest.raises(replay.Refused, match="not 'set_split'"):
        replay.truncate(log, "set_split")


def _repair_log(tmp_path: Path) -> list:
    log = _log(tmp_path, [
        {"kind": "set_lens", "lenses": ["clinical"]},
        {"kind": "apply_repair", "finding_id": "binary_text__sex", "option": "level",
         "params": {"column": "sex", "one": "f", "zero": "m", "relabels": True}},
        {"kind": "dismiss_finding", "finding_id": "wide_shape"},
        {"kind": "revert", "decision_id": 3},
        {"kind": "set_target", "column": "y"},
        {"kind": "select_models", "models": ["linear"]},
    ])
    return replay.log_lines(log)


def test_the_findings_rule_keeps_a_record_whose_finding_fires(tmp_path):
    lines = _repair_log(tmp_path)
    kept, left = replay.findings_rule(lines, {"binary_text__sex", "wide_shape"})
    assert left == [] and kept == lines


def test_the_findings_rule_leaves_out_a_record_whose_finding_does_not_fire_and_its_revert(
        tmp_path):
    lines = _repair_log(tmp_path)
    kept, left = replay.findings_rule(lines, {"binary_text__sex"})
    assert [(o.seq, o.kind, o.finding_id) for o in left] == [(3, "dismiss_finding", "wide_shape"),
                                                             (4, "revert", None)]
    assert "`wide_shape` does not fire on this draw" in left[0].why
    assert [r.seq for r, _ in kept] == [1, 2, 5, 6]
    recorded = {r.id: line for r, line in lines}
    assert all(line == recorded[r.id] for r, line in kept)  # each kept line as recorded
    kept, left = replay.findings_rule(lines, set())
    assert [o.seq for o in left] == [2, 3, 4] and [r.seq for r, _ in kept] == [1, 5, 6]


def test_the_findings_are_read_under_the_records_their_stages_read_and_no_findings(tmp_path):
    lines = _repair_log(tmp_path)
    kept = replay.findings_inputs(lines, {"lens", "target"})
    assert [r.decision.kind for r, _ in kept] == ["set_lens", "set_target"]


# ── the fresh draw's table ────────────────────────────────────────────────────


def draw(seed: int) -> pd.DataFrame:
    """A small generator: two columns and an outcome from ``seed``."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"x": rng.normal(size=20).round(3), "y": rng.integers(0, 2, 20)})


def other(seed: int) -> pd.DataFrame:
    return draw(seed + 1)


def _prov(tmp_path: Path, seed: int = 7, **more) -> dict:
    path = tmp_path / "table.csv"
    draw(seed).to_csv(path, index=False)
    table = {"role": "table", "name": "table.csv", "path": str(path),
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return {"inputs": [table, *more.get("extra", [])]}


def test_a_fresh_draw_is_the_recorded_generator_at_another_seed(tmp_path):
    check = replay.fresh_table(_prov(tmp_path), replay.FreshDraw(draw, 7, 8), tmp_path / "d")
    expected = draw(8).to_csv(index=False).encode()
    assert Path(check.path).read_bytes() == expected
    assert check.sha256 == hashlib.sha256(expected).hexdigest() != check.recorded_sha256
    assert check.regenerated_sha256 == check.recorded_sha256 and check.rows == 20
    by_name = replay.fresh_table(_prov(tmp_path), replay.FreshDraw(
        "turbotab.core.tests.test_replay_rebuild:draw", 7, 8), tmp_path / "e")
    assert by_name.sha256 == check.sha256


def test_a_fresh_draw_is_refused_from_another_generator_the_same_seed_or_a_joined_file(tmp_path):
    prov = _prov(tmp_path)
    with pytest.raises(replay.Refused, match="does not write the recorded table at seed 7"):
        replay.fresh_table(prov, replay.FreshDraw(other, 7, 8), tmp_path / "a")
    with pytest.raises(replay.Refused, match="does not write the recorded table at seed 6"):
        replay.fresh_table(prov, replay.FreshDraw(draw, 6, 8), tmp_path / "b")
    with pytest.raises(replay.Refused, match="Seed 7 is the recorded table's own"):
        replay.fresh_table(prov, replay.FreshDraw(draw, 7, 7), tmp_path / "c")
    joined = _prov(tmp_path, extra=[{"role": "joined file", "name": "labs.csv"}])
    with pytest.raises(replay.Refused, match="joined file"):
        replay.fresh_table(joined, replay.FreshDraw(draw, 7, 8), tmp_path / "d")
    with pytest.raises(replay.Refused, match="MODULE:FUNCTION"):
        replay.fresh_table(prov, replay.FreshDraw("no_colon", 7, 8), tmp_path / "e")
    with pytest.raises(replay.Refused, match="cannot be loaded"):
        replay.fresh_table(prov, replay.FreshDraw("turbotab.core.tests.test_replay_rebuild:nope",
                                                  7, 8), tmp_path / "f")


def test_the_command_takes_the_flags_and_a_refused_draw_exits_2(tmp_path, capsys):
    """The CLI's fresh draw on a bundle whose generator does not write its table: refused, exit 2,
    nothing replayed (no home kept)."""
    import zipfile

    prov = _prov(tmp_path)
    log = _log(tmp_path, PREDICT)
    prov["decisions"] = {"sha256": hashlib.sha256(log).hexdigest()}
    bundle = tmp_path / "bundle.zip"
    with zipfile.ZipFile(bundle, "w") as z:
        z.writestr("turbotab-export/provenance.json", json.dumps(prov))
        z.writestr("turbotab-export/decisions.jsonl", log)
    code = replay.main([str(bundle), "--fresh-draw",
                        "turbotab.core.tests.test_replay_rebuild:other", "--recorded-seed", "7",
                        "--seed", "8", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert code == 2 and "does not write the recorded table" in out["refused"]
    assert out["home"] is None and out["project"] is None
    with pytest.raises(SystemExit):
        replay.main([str(bundle), "--fresh-draw", "x:y"])  # no seeds: a usage error

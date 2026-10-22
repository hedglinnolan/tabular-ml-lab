"""Replay's rebuilds for the swarm (``core/export/replay.py``; SWARM_PLAN §6–§8): the log truncated
before the opening or the lock, the findings rule on a fresh draw, and the fresh draw's generator
check, on decision logs written here by the log's own appender and on tables whose bytes are
hashed here with hashlib.

* :func:`replay.truncate` keeps every record before the first live ``open_seal`` (under inference
  ``lock_plan``), each kept line the log's own bytes; a withdrawn lock is passed over for the lock
  that holds; a log with no such record comes back whole; the kind is read from the purpose.
* :func:`replay.findings_rule` keeps a record tied to a finding the new bytes raise, and leaves out
  one whose finding they do not raise, with every revert of it; nothing else moves. A standing
  repair whose settings the finding offers (or a part of them) is kept as its own bytes; one whose
  settings it does not offer is replayed with the one offered form that has its choices (its text,
  yes-or-no and empty settings), its sentence authored again from the new settings
  (``qc_drift.rlsc_sentence`` on the settings written here); with no such form, or several, it is
  unmatched; a repair that does not stand (withdrawn, or followed by its finding's next record)
  is kept as recorded.
* :func:`replay.findings_inputs` keeps the records that write a slot the findings stage reads: a
  read-numbers repair (``numbers_read``) among them, any other finding's record not.
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


SEX = {"column": "sex", "one": "f", "zero": "m", "relabels": True}


def finding(fid: str, *forms: tuple[str, dict]) -> dict:
    """A finding as the findings stage serves it: its id and each offered repair's key and
    settings."""
    return {"id": fid, "repairs": [{"key": key, "label": key, "sentence": f"{key} was applied.",
                                    "decision": {"kind": "apply_repair", "finding_id": fid,
                                                 "option": key, "params": params}}
                                   for key, params in forms]}


SEX_FINDING = finding("binary_text__sex", ("level", SEX))


def test_the_findings_rule_keeps_a_record_whose_finding_fires_and_offers_its_settings(tmp_path):
    lines = _repair_log(tmp_path)
    ruled = replay.findings_rule(lines, [SEX_FINDING, finding("wide_shape")])
    assert ruled.left_out == [] and ruled.redrawn == [] and ruled.unmatched == []
    assert ruled.kept == lines  # every line its own bytes


def test_a_part_of_what_the_new_draw_offers_is_the_records_own_choice(tmp_path):
    """The record marked 999 missing; the new draw's finding offers 999 and -1: the record is
    replayed as recorded, -1 left as the analyst left it (``repairs.admits``)."""
    lines = replay.log_lines(_log(tmp_path, [
        {"kind": "apply_repair", "finding_id": "sentinel_missing__hb", "option": "as_missing",
         "params": {"values": {"hb": [999.0]}}}]))
    wider = finding("sentinel_missing__hb", ("as_missing", {"values": {"hb": [999.0, -1.0]}}))
    ruled = replay.findings_rule(lines, [wider])
    assert ruled.kept == lines and ruled.redrawn == [] and ruled.unmatched == []
    other = finding("sentinel_missing__hb", ("as_missing", {"values": {"hb": [-1.0]}}))
    ruled = replay.findings_rule(lines, [other])  # its one form has the record's choices
    assert [d.settings for d in ruled.redrawn] == [["values"]]
    assert json.loads(ruled.kept[0][1])["decision"]["params"] == {"values": {"hb": [-1.0]}}


def test_the_findings_rule_leaves_out_a_record_whose_finding_does_not_fire_and_its_revert(
        tmp_path):
    lines = _repair_log(tmp_path)
    ruled = replay.findings_rule(lines, [SEX_FINDING])
    left, kept = ruled.left_out, ruled.kept
    assert [(o.seq, o.kind, o.finding_id) for o in left] == [(3, "dismiss_finding", "wide_shape"),
                                                             (4, "revert", None)]
    assert "`wide_shape` does not fire on this draw" in left[0].why
    assert [r.seq for r, _ in kept] == [1, 2, 5, 6]
    recorded = {r.id: line for r, line in lines}
    assert all(line == recorded[r.id] for r, line in kept)  # each kept line as recorded
    ruled = replay.findings_rule(lines, [])
    assert [o.seq for o in ruled.left_out] == [2, 3, 4]
    assert [r.seq for r, _ in ruled.kept] == [1, 5, 6]


# ── the settings rule: a repair's settings against the new draw's offer ──────

QC = "pack::metabolomics::pooled_qc"
FEATURES = [f"mz_{j:04d}" for j in range(1, 151)]


def qc_params(batch: str | None, dropped: dict, kept: int) -> dict:
    """QC-RLSC's settings as ``qc_drift.rlsc_offer`` writes them: the columns it reads (text, or
    empty for the whole run as one batch) and the features its filters drop (lists and counts)."""
    return {"column": "sample_type", "qc_levels": ["pooled_qc"], "order_column": "run_order",
            "batch_column": batch, "features": FEATURES, "n_qc": 28, "platform": "lc_ms",
            "d_ratio_max": None, "dropped": dropped, "kept": kept, "batches": 4 if batch else 1}


FIRST = {"detection": ["mz_0004", "mz_0021"], "uncorrectable": ["mz_0026"]}
SECOND = {"detection": ["mz_0004"], "uncorrectable": ["mz_0001", "mz_0021", "mz_0026"]}


def _qc_log(tmp_path: Path, *more: dict) -> list:
    return replay.log_lines(_log(tmp_path, [
        {"kind": "set_lens", "lenses": ["metabolomics"]},
        {"kind": "apply_repair", "finding_id": QC, "option": "qc_rlsc_lc",
         "params": qc_params("batch", FIRST, 147)},
        {"kind": "apply_repair", "finding_id": "binary_text__sex", "option": "level",
         "params": SEX},
        *more]))


def test_a_repair_whose_settings_the_new_draw_does_not_offer_takes_the_draws_own(tmp_path):
    """The per-batch QC-RLSC form drops other features on the new draw: the record is replayed
    with that form's settings (the one form whose batch column is the record's), named with the
    settings that differ, and its sentence counts the new draw's features; the sex repair, whose
    settings the draw offers, is its own bytes."""
    from turbotab.core.methods.qc_drift import rlsc_sentence

    lines = _qc_log(tmp_path)
    new = qc_params("batch", SECOND, 146)
    raised = [finding(QC, ("qc_rlsc_lc", new), ("qc_rlsc_lc", qc_params(None, SECOND, 120))),
              SEX_FINDING]
    ruled = replay.findings_rule(lines, raised)
    assert ruled.left_out == [] and ruled.unmatched == []
    assert ruled.redrawn == [replay.Redrawn(seq=2, finding_id=QC, option="qc_rlsc_lc",
                                            settings=["dropped", "kept"])]
    (r0, l0), (r1, l1), (r2, l2) = ruled.kept
    assert l0 == lines[0][1] and l2 == lines[2][1]
    again = json.loads(l1)
    assert again["decision"]["params"] == new
    assert again["id"] == r1.id and again["seq"] == 2 and again["at"] == json.loads(lines[1][1])["at"]
    assert again["sentence"] == rlsc_sentence(new)
    assert "146 of 150 remain" in again["sentence"]
    assert "147 of 150" not in again["sentence"]


def test_a_repair_with_no_offered_form_of_its_choices_is_unmatched(tmp_path):
    """The option is not offered; only the whole-run form is (another batch column); two forms
    have the record's choices: each is unmatched, named, and its line left as recorded."""
    lines = _qc_log(tmp_path)
    for raised, why in [
            ([finding(QC, ("aside", {"column": "sample_type"}))], "does not offer `qc_rlsc_lc`"),
            ([finding(QC, ("qc_rlsc_lc", qc_params(None, SECOND, 120)))],
             "only with other choices than the record's"),
            ([finding(QC, ("qc_rlsc_lc", qc_params("batch", SECOND, 146)),
                      ("qc_rlsc_lc", qc_params("batch", FIRST, 140)))],
             "offers 2 forms of `qc_rlsc_lc`")]:
        ruled = replay.findings_rule(lines, [*raised, SEX_FINDING])
        [unmatched] = ruled.unmatched
        assert (unmatched.seq, unmatched.finding_id) == (2, QC) and why in unmatched.why
        assert ruled.redrawn == [] and [line for _, line in ruled.kept] == [x for _, x in lines]


def test_a_repair_that_does_not_stand_is_kept_as_recorded_whatever_the_draw_offers(tmp_path):
    """Withdrawn by a revert, or followed by its finding's next record: it changes nothing, so its
    settings are not read against the new draw's offer."""
    offer = [finding(QC, ("qc_rlsc_lc", qc_params("batch", SECOND, 146))), SEX_FINDING]
    withdrawn = _qc_log(tmp_path, {"kind": "revert", "decision_id": 2})
    ruled = replay.findings_rule(withdrawn, offer)
    assert ruled.redrawn == [] and ruled.unmatched == [] and ruled.kept == withdrawn
    followed = _qc_log(tmp_path / "b", {"kind": "dismiss_finding", "finding_id": QC})
    ruled = replay.findings_rule(followed, offer)
    assert ruled.redrawn == [] and ruled.unmatched == [] and ruled.kept == followed


# ── the records the findings are read under ──────────────────────────────────


def test_the_findings_are_read_under_the_records_their_stages_read(tmp_path):
    """Of the records tied to a finding, a read-numbers repair writes ``numbers_read`` (which the
    findings stage reads) and is among them with its revert; the sentinel repair and the sex
    repair write only the dispositions, and are not."""
    lines = replay.log_lines(_log(tmp_path, [
        {"kind": "set_lens", "lenses": ["clinical"]},
        {"kind": "apply_repair", "finding_id": "text_numbers__hb", "option": "read",
         "params": {"columns": {"hb": {"decimal": ".", "thousands": False}}}},
        {"kind": "apply_repair", "finding_id": "binary_text__sex", "option": "level",
         "params": SEX},
        {"kind": "apply_repair", "finding_id": "sentinel_missing__hb", "option": "as_missing",
         "params": {"values": {"hb": [999.0]}}},
        {"kind": "revert", "decision_id": 2},
        {"kind": "set_target", "column": "y"},
        {"kind": "select_models", "models": ["linear"]},
    ]))
    kept = replay.findings_inputs(lines, {"lens", "target", "numbers_read"})
    assert [r.seq for r, _ in kept] == [1, 2, 5, 6]
    assert [r.seq for r, _ in replay.findings_inputs(lines, {"lens", "target"})] == [1, 6]


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


def test_the_summary_says_each_record_left_out_and_each_repair_redrawn():
    report = replay.ReplayReport(
        bundle="b.zip", reproduced=False, finished=True, estimates_again={},
        left_out=[replay.LeftOut(seq=3, kind="apply_repair", finding_id="binary_text__sex",
                                 why="its finding `binary_text__sex` does not fire on this draw")],
        redrawn=[replay.Redrawn(seq=2, finding_id=QC, option="qc_rlsc_lc",
                                settings=["dropped", "kept"])])
    text = replay.summary(report)
    assert ("Record 3 (apply_repair) left out of the fresh draw: its finding `binary_text__sex` "
            "does not fire on this draw.") in text
    assert (f"Record 2 (apply_repair) replayed with the fresh draw's settings for `qc_rlsc_lc` of "
            f"`{QC}`; these differ from the record's: dropped, kept.") in text


def test_the_summary_and_the_help_name_the_live_record_the_log_was_cut_before(tmp_path, capsys):
    """A lock withdrawn by a revert (Fit pressed, then cancelled) stays in the rebuilt log, so the
    summary says the cut came before the first *live* lock, and the help says the same of the
    flag; with no live one it says nothing was cut."""
    log = _log(tmp_path, [
        {"kind": "set_target", "column": "y"},
        {"kind": "set_purpose", "purpose": "inference"},
        {"kind": "lock_plan"},
        {"kind": "revert", "decision_id": 3},
        {"kind": "select_models", "models": ["linear"]},
        {"kind": "lock_plan"},
    ])
    _, where = replay.truncate(log)
    assert where == replay.Cut(kind="lock_plan", seq=6, kept=5, left=1)
    text = replay.summary(replay.ReplayReport(bundle="b.zip", reproduced=False, finished=True,
                                              estimates_again={}, cut=where))
    assert ("The log cut before its first live `lock_plan` (record 6): 5 records kept, 1 left "
            "out.") in text
    _, nowhere = replay.truncate(log, "open_seal")
    text = replay.summary(replay.ReplayReport(bundle="b.zip", reproduced=False, finished=True,
                                              estimates_again={}, cut=nowhere))
    assert "The log holds no live `open_seal`: nothing was cut." in text
    with pytest.raises(SystemExit):
        replay.main(["--help"])
    assert "before its first live open_seal" in " ".join(capsys.readouterr().out.split())

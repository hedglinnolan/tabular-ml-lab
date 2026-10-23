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
* A fresh draw of a log holding a live ``open_seal`` is refused before anything is built, the
  record and ``--truncate open_seal`` named; the log cut before it, and an opening a revert
  withdrew, hold none. A form's cut points and knot places are not in its record (a record of
  quintiles, a spline or a data-derived cut with cut points is refused) but learned by the form
  step on the rows it is given (numpy's quantiles).
* A spline whose k the server set by Harrell's rule, on a fresh draw (:func:`replay.rule_verdict`,
  k by the rule as RMS states it): under prediction another size at the record's k is replayed
  with that size, another k (or no size read) refused; under inference any other size refused.
  The later records that write what the rows are read under (:func:`replay.later_records`, read
  on the state folded up to the record before each: an exclusion, a revert of one recorded before
  the spline, the outcome's task or event answered, but not the models answer, a withdrawn
  record nor a task answered for another column) are named only where their own step changed
  the rule's size (:func:`replay.later_changes`, each size pandas' on the rows named): an
  exclusion of the rows under 50 is, a reading confirmed in a table with no blank cell is not;
  where the sizes are not the engine's (over every row, or every row not held out) or a later
  record rewrites the table read (the grain, a reading of a column the flow reads) or writes the
  task or the outcome, each is named as one that can change them. Where a later record changed
  the outcome, its scale, task or event from the record to the log's end
  (:func:`replay.input_changes`), another size is refused under either purpose, the record and
  the change named; a task answered as the detected one, or changed and changed back, changes
  none. A refilled record (:func:`replay.refilled_line`) holds the decision and the
  sentence the log's own appender writes for the spline declared on the new size, every other
  field as recorded; one whose sentence the log would not author now is refused.

The engine-level rebuilds on the swarm's fixtures are ``research/swarm/tests/test_replay_fixtures.py``.
"""
from __future__ import annotations

import dataclasses
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


def _bundle_with(tmp_path: Path, log: bytes, seed: int = 7) -> Path:
    """A bundle of ``log`` whose table is :func:`draw` at ``seed`` (its provenance hashed here)."""
    import zipfile

    prov = _prov(tmp_path, seed)
    prov["decisions"] = {"sha256": hashlib.sha256(log).hexdigest()}
    bundle = tmp_path / "bundle.zip"
    with zipfile.ZipFile(bundle, "w") as z:
        z.writestr("turbotab-export/provenance.json", json.dumps(prov))
        z.writestr("turbotab-export/decisions.jsonl", log)
    return bundle


OPENING = {"kind": "open_seal", "family": "linear", "target": "y", "n_holdout": 80,
           "metric": "log_loss", "scores": {"linear": {"log_loss": 0.6089974194457624,
                                                      "auc": 0.6559877955758963}}}


def test_a_fresh_draw_of_a_log_that_opened_the_held_out_rows_is_refused_naming_the_cut(tmp_path):
    """The opening keeps the recorded draw's held-out scores as the plan's result, and a fresh
    draw would replay it as written: refused before any project is built, the record, its score
    and ``--truncate open_seal`` named. The generator writes the recorded table (checked first),
    so nothing else refuses it."""
    log = _log(tmp_path, [*PREDICT[:4], OPENING, PREDICT[5]])
    report = replay.run(_bundle_with(tmp_path, log), fresh=replay.FreshDraw(draw, 7, 8))
    assert report.refused is not None
    assert report.refused.startswith("Record 5 (open_seal) holds the recorded draw's held-out "
                                     "scores as the plan's reported result (`linear`'s held-out "
                                     "`log_loss` 0.609 on 80 rows)")
    assert "--truncate open_seal" in report.refused
    assert report.home is None and report.project is None and report.finished is None
    assert report.fresh.regenerated_sha256 == report.fresh.recorded_sha256


def test_a_cut_log_and_a_withdrawn_opening_hold_no_live_opening(tmp_path):
    """The check reads what the replay writes: a log cut before the opening holds none, and an
    opening a revert withdrew is not live, as :func:`replay.truncate` reads it."""
    log = _log(tmp_path, [*PREDICT[:4], OPENING, PREDICT[5]])
    assert replay.live_record(replay.log_lines(log), "open_seal").seq == 5
    cut, _ = replay.truncate(log)
    assert replay.live_record(replay.log_lines(cut), "open_seal") is None
    replay.refuse_live_opening(cut)  # silent
    withdrawn = _log(tmp_path / "w", [*PREDICT[:4], OPENING, {"kind": "revert", "decision_id": 5},
                                      PREDICT[5]])
    assert replay.live_record(replay.log_lines(withdrawn), "open_seal") is None
    replay.refuse_live_opening(withdrawn)  # silent
    with pytest.raises(replay.Refused, match=r"^Record 5 \(open_seal\)"):
        replay.refuse_live_opening(log)


def test_a_forms_cut_points_and_knot_places_are_not_in_its_record_but_learned_on_the_rows():
    """Why a fresh draw needs no rule for them: a record of quintiles, a spline or a data-derived
    cut cannot carry cut points (refused), and the form step learns quintiles' cut points and a
    spline's knots from the values it is given (numpy's type-7 quantiles, Harrell's percentiles
    for 5 knots), so each draw's rows give their own. Declared categories' cut points are the
    analyst's, from outside the data."""
    from turbotab.core import decisions
    from turbotab.core.methods import exposure_form as ef

    for form in ("quintiles", "spline", "optimal"):
        with pytest.raises(decisions.Refusal, match=f"the {form} form has none"):
            decisions.validate({"kind": "set_exposure_form", "column": "x", "form": form,
                                "cuts": [1.0]})
    for seed in (1, 2):
        x = np.random.default_rng(seed).normal(size=1000)
        assert np.allclose(ef.place({"form": "quintiles"}, x)["cuts"],
                           np.quantile(x, [0.2, 0.4, 0.6, 0.8]), rtol=0, atol=1e-12)
        assert np.allclose(ef.place({"form": "spline", "knots": 5}, x)["knots"],
                           np.quantile(x, [0.05, 0.275, 0.5, 0.725, 0.95]), rtol=0, atol=1e-12)


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


# ── a spline's k by Harrell's rule on a fresh draw ───────────────────────────


def harrell(n: float) -> int:
    """Harrell's rule as RMS §2.4.6 states it: 3 knots below an effective sample size of 30, 5
    from 100, else 4."""
    return 3 if n < 30 else 5 if n >= 100 else 4


def rule_read(said: float | None, read: float | None, *, knots: int | None = None,
              later: tuple = (), inputs: tuple = ()) -> "replay.RuleRead":
    """The age spline of record 4 as :func:`replay.rule_forms` reads it: the record's k (the
    rule's on ``said``, else the one given) and size, and the rule's k and size on this draw."""
    return replay.RuleRead(column="age", seq=4, kind="set_exposure_form",
                           knots=knots if knots is not None else harrell(said), said=said,
                           k=harrell(read) if read is not None else None, read=read, later=later,
                           inputs=inputs)


def test_under_prediction_a_new_size_with_the_records_k_is_replayed_and_another_k_refused():
    """Under prediction the engine keeps the record's k whatever the rows (the form question is
    skipped there), so a size that gives the same k by the rule is replayed with that size; one
    that gives another k, or no size read, is refused; the same size is replayed as recorded."""
    assert replay.rule_verdict(rule_read(117.0, 117.0), "prediction") == ("kept", None)
    verdict, said = replay.rule_verdict(rule_read(117.0, 114.0), "prediction")
    assert harrell(114) == harrell(117) == 5 and verdict == "refilled"
    assert said == (
        "Record 4 (set_exposure_form) gives `age` a spline whose k is Harrell's rule on the "
        "analyzed rows: k = 5 on an effective sample size of 117 on the recorded draw, and on "
        "this draw the rule reads 114 (k = 5) on the rows as the log ends. Under prediction the "
        "record's k is replayed as recorded, and it is the rule's on this draw too, so the record "
        "is replayed stating this draw's size, 114, in place of 117.")
    verdict, said = replay.rule_verdict(rule_read(117.0, 95.0), "prediction")
    assert harrell(95) == 4 and verdict == "refused"
    assert said == (
        "Record 4 (set_exposure_form) gives `age` a spline whose k is Harrell's rule on the "
        "analyzed rows: k = 5 on an effective sample size of 117 on the recorded draw, and on "
        "this draw the rule reads 95 (k = 4) on the rows as the log ends. Under prediction a "
        "fresh draw replays the record's k as recorded, not the rule's on this draw, so the "
        "rebuild is refused.")
    verdict, said = replay.rule_verdict(rule_read(117.0, None), "prediction")
    assert verdict == "refused" and "the rule reads no effective sample size" in said
    # A record of the rule's default k, no size read when it was declared: the new draw's size
    # where the rule gives that k, refused where it gives another.
    verdict, said = replay.rule_verdict(rule_read(None, 85.0, knots=4), "prediction")
    assert verdict == "refilled" and "k = 4 with no effective sample size read" in said
    assert said.endswith("so the record is replayed stating this draw's size, 85, where it "
                         "stated none.")
    assert replay.rule_verdict(rule_read(None, 120.0, knots=4), "prediction")[0] == "refused"
    assert replay.rule_verdict(rule_read(None, None, knots=4), "prediction") == ("kept", None)


def test_under_inference_another_size_is_refused_whatever_k_it_gives():
    """Under inference the engine asks the form question again where the analyzed rows' size is
    not the record's (``exposure_form.rule_stale``), so another size is refused even with the same
    k; the same size, to within 1e-9 of its size, is replayed as recorded."""
    assert replay.rule_verdict(rule_read(400.0, 400.0), "inference") == ("kept", None)
    assert replay.rule_verdict(rule_read(400.0, 400.0 + 1e-8), "inference") == ("kept", None)
    verdict, said = replay.rule_verdict(rule_read(400.0, 380.0), "inference")
    assert harrell(380) == harrell(400) and verdict == "refused"
    assert said == (
        "Record 4 (set_exposure_form) gives `age` a spline whose k is Harrell's rule on the "
        "analyzed rows: k = 5 on an effective sample size of 400 on the recorded draw, and on "
        "this draw the rule reads 380 (k = 5) on the rows as the log ends. A fresh draw replays "
        "the record's k and the size it states as recorded, so the rebuild is refused.")
    assert replay.rule_verdict(rule_read(400.0, None), "inference")[0] == "refused"


def test_a_refusal_names_each_later_record_that_changed_the_size_the_rule_reads():
    """A later record whose own step changed the rule's size on this draw is named as one that
    changed it; where the sizes could not be read step by step (``traced`` False), each later
    record that writes what the rows are read under is named as one that can change them."""
    _, one = replay.rule_verdict(rule_read(117.0, 95.0, later=((16, "set_exclusions"),)),
                                 "prediction")
    assert ("the rule reads 95 (k = 4) on the rows as the log ends, after record 16 "
            "(set_exclusions), which changed the size it reads. Under prediction") in one
    _, two = replay.rule_verdict(rule_read(400.0, 380.0, later=((16, "set_exclusions"),
                                                               (18, "set_missing"))), "inference")
    assert ("on the rows as the log ends, after records 16 (set_exclusions) and 18 "
            "(set_missing), which changed the size it reads. A fresh draw") in two
    loose = dataclasses.replace(rule_read(117.0, 95.0, later=((16, "set_exclusions"),
                                                              (17, "confirm_readings"))),
                                traced=False)
    _, three = replay.rule_verdict(loose, "prediction")
    assert ("on the rows as the log ends, after records 16 (set_exclusions) and 17 "
            "(confirm_readings), which can change them. Under prediction") in three
    _, none = replay.rule_verdict(rule_read(117.0, 95.0), "prediction")
    assert "on the rows as the log ends. Under prediction" in none and "after" not in none


def test_a_size_read_under_what_a_later_record_changed_is_refused_naming_it():
    """AR-V1: the record's size and sentence are read under the outcome, its scale, task and event
    at its record, so where a later record changed one of them (``inputs``, from
    :func:`replay.input_changes`) another size is refused under prediction even at the record's
    k, as under inference, naming the record and what it changed; with nothing changed, the same
    sizes are refilled (above)."""
    assert harrell(400) == harrell(117)
    task = rule_read(117.0, 400.0, later=((15, "set_task"),),
                     inputs=("the task from binary to regression",))
    for purpose in ("prediction", "inference"):
        verdict, said = replay.rule_verdict(task, purpose)
        assert verdict == "refused"
        assert said == (
            "Record 4 (set_exposure_form) gives `age` a spline whose k is Harrell's rule on the "
            "analyzed rows: k = 5 on an effective sample size of 117 on the recorded draw, and on "
            "this draw the rule reads 400 (k = 5) on the rows as the log ends, after record 15 "
            "(set_task), which changed what the size is read under: the task from binary to "
            "regression. The record's size and sentence are read under the outcome, its scale, "
            "task and event as they stood when it was recorded, so a size read under later ones "
            "is not replayed in it, and the rebuild is refused.")
    two = rule_read(117.0, 400.0, later=((15, "set_outcome_scale"), (16, "set_event")),
                    inputs=("the outcome from `y` to `ln_y`", "the outcome's scale from none to log"))
    assert ("on the rows as the log ends, after records 15 (set_outcome_scale) and 16 "
            "(set_event), which changed what the size is read under: the outcome from `y` to "
            "`ln_y` and the outcome's scale from none to log. The record's size"
            ) in replay.rule_verdict(two, "prediction")[1]
    # The same sizes with nothing changed: refilled, as above; the same size, kept.
    assert replay.rule_verdict(rule_read(117.0, 114.0), "prediction")[0] == "refilled"
    assert replay.rule_verdict(rule_read(117.0, 117.0, inputs=task.inputs), "prediction") == (
        "kept", None)


# A yes/no outcome beside ``age`` and ``bmi``, no cell blank, for the rule's size step by step
# (:func:`replay.later_changes`); every size below is pandas' smaller of the events and the
# non-events on the rows named.
FIFTY = 50


@pytest.fixture(scope="module")
def ages(tmp_path_factory):
    from turbotab.core.datastore import DataStore, ingest

    rng = np.random.default_rng(11)
    n = 400
    frame = pd.DataFrame({"y": (rng.random(n) < 0.3).astype(int),
                          "age": rng.integers(20, 81, n), "bmi": rng.normal(27, 4, n).round(1)})
    assert not frame.isna().any().any()
    folder = tmp_path_factory.mktemp("ages")
    frame.to_csv(folder / "ages.csv", index=False)
    ingest(folder / "ages.csv", folder / "ages.parquet")
    with DataStore(folder / "ages.parquet", 2 << 30) as store:
        yield frame, store


def events_or_not(frame: pd.DataFrame) -> float:
    return float(min(frame["y"].sum(), len(frame) - frame["y"].sum()))


SPLINE_THEN = [
    {"kind": "set_lens", "lenses": ["clinical"]},
    {"kind": "set_target", "column": "y"},
    {"kind": "set_purpose", "purpose": "prediction"},
    {"kind": "set_exposure_form", "column": "age", "form": "spline"},
    {"kind": "confirm_readings", "items": [{"reading": "code_or_count", "column": "bmi",
                                            "value": "amount"}]},
]
FIFTY_AND_OVER = {"kind": "set_exclusions", "rules": [{"kind": "range", "column": "age",
                                                       "low": FIFTY, "reason": "fifty and over"}]}


def _slots():
    """What the rule's size is read on (as :func:`replay.rule_forms` reads it), and what the
    working table is built from."""
    from turbotab.core.stages import build_graph

    graph = build_graph()
    reads = {s for name in graph.upstream("forms") for s in graph[name].reads}
    table = {s for name in graph.upstream("working") | {"working"} for s in graph[name].reads}
    return reads | {"target", "task", "event"}, table


def test_the_rule_reads_its_size_on_the_rows_each_states_cohort_keeps(ages, tmp_path):
    from turbotab.core.decisions import fold

    frame, store = ages
    records = [r for r, _ in replay.log_lines(_log(tmp_path, [*SPLINE_THEN, FIFTY_AND_OVER]))]
    states = [fold(records[:4]), fold(records)]
    old, kept = events_or_not(frame), events_or_not(frame[frame["age"] >= FIFTY])
    assert old != kept
    assert replay.rule_sizes(store, states, "binary") == [old, kept]
    # Over the same rows, every row but those held out.
    held = np.arange(0, len(frame), 5)
    pool = frame.drop(index=held)
    assert replay.rule_sizes(store, states, "binary", np.setdiff1d(np.arange(len(frame)), held)
                             ) == [events_or_not(pool), events_or_not(pool[pool["age"] >= FIFTY])]


def test_only_a_later_record_whose_step_changed_the_rules_size_is_named(ages, tmp_path):
    """After the spline (record 4): confirming `bmi`'s reading in a table with no blank cell
    changes no row, and the exclusion of the rows under 50 does (pandas): the exclusion alone is
    named, as one that changed the size. With no exclusion, nothing is named."""
    frame, store = ages
    reads, table = _slots()
    kept = events_or_not(frame[frame["age"] >= FIFTY])
    records = [r for r, _ in replay.log_lines(_log(tmp_path, [
        *SPLINE_THEN, FIFTY_AND_OVER, {"kind": "select_models", "models": ["linear"]}]))]
    assert replay.later_records(records, 4, reads) == ((5, "confirm_readings"),
                                                      (6, "set_exclusions"))
    assert replay.later_changes(records, 4, reads, store, "binary", kept, table=table) == (
        ((6, "set_exclusions"),), True)
    plain = [r for r, _ in replay.log_lines(_log(tmp_path, SPLINE_THEN, "plain.jsonl"))]
    assert replay.later_changes(plain, 4, reads, store, "binary", events_or_not(frame),
                                table=table) == ((), True)


def test_the_steps_are_read_on_the_rows_whose_size_is_the_engines(ages, tmp_path):
    """The sizes are read over every row, else over every row not held out, whichever gives the
    size the engine reads as the log ends; where neither does, or a later record rewrites the
    table the walk reads (the grain; a confirmed reading of `age`, which the exclusion reads,
    where one of `bmi` alone is traced above), each later record that writes what the rows are
    read under is named, not traced."""
    frame, store = ages
    reads, table = _slots()
    records = [r for r, _ in replay.log_lines(_log(tmp_path, [*SPLINE_THEN, FIFTY_AND_OVER]))]
    held = np.arange(0, len(frame), 5)
    pool = frame.drop(index=held)
    training = events_or_not(pool[pool["age"] >= FIFTY])
    assert training != events_or_not(frame[frame["age"] >= FIFTY])
    assert replay.later_changes(records, 4, reads, store, "binary", training, table=table,
                                sealed=held) == (((6, "set_exclusions"),), True)
    later = ((5, "confirm_readings"), (6, "set_exclusions"))
    assert replay.later_changes(records, 4, reads, store, "binary", training + 1, table=table,
                                sealed=held) == (later, False)
    assert "grain" in table and "grain" in reads
    grained = [r for r, _ in replay.log_lines(_log(tmp_path, [
        *SPLINE_THEN, FIFTY_AND_OVER, {"kind": "set_grain", "grain": "one_row_per_unit"}],
        "grain.jsonl"))]
    assert replay.later_changes(grained, 4, reads, store, "binary",
                                events_or_not(frame[frame["age"] >= FIFTY]), table=table) == (
        (*later, (7, "set_grain")), False)
    aged = [r for r, _ in replay.log_lines(_log(tmp_path, [
        *SPLINE_THEN, FIFTY_AND_OVER, {"kind": "confirm_readings", "items": [
            {"reading": "code_or_count", "column": "age", "value": "amount"}]}], "aged.jsonl"))]
    assert replay.later_changes(aged, 4, reads, store, "binary",
                                events_or_not(frame[frame["age"] >= FIFTY]), table=table) == (
        (*later, (7, "confirm_readings")), False)


def test_the_later_records_are_those_that_write_what_the_rows_are_read_under(tmp_path):
    """After the spline (record 4): an exclusion writes a slot the rows are read under and is
    named; the models answer is not; a missing-values answer a revert withdrew is not, nor its
    revert; a revert that withdraws an exclusion recorded before the spline is."""
    from turbotab.core.stages import build_graph

    graph = build_graph()
    reads = {s for name in graph.upstream("forms") for s in graph[name].reads}
    assert {"exclusions", "missing", "target"} <= reads and "models" not in reads
    lines = replay.log_lines(_log(tmp_path, [
        {"kind": "set_target", "column": "y"},
        {"kind": "set_exclusions", "rules": [{"kind": "range", "column": "age", "low": 18,
                                              "reason": "adults"}]},
        {"kind": "set_purpose", "purpose": "prediction"},
        {"kind": "set_exposure_form", "column": "age", "form": "spline"},
        {"kind": "select_models", "models": ["linear"]},
        {"kind": "set_missing", "strategy": "complete_case"},
        {"kind": "revert", "decision_id": 6},
        {"kind": "revert", "decision_id": 2},
        {"kind": "set_exclusions", "rules": [{"kind": "range", "column": "age", "low": 50,
                                              "reason": "fifty and over"}]},
    ]))
    records = [r for r, _ in lines]
    assert replay.later_records(records, 4, reads) == ((8, "revert"), (9, "set_exclusions"))
    assert replay.later_records(records, 9, reads) == ()


def _after_the_spline(tmp_path: Path, name: str, *later: dict) -> list:
    """The records of :data:`SPLINE_THEN` (the spline at record 4, `y`'s readings at 5) and then
    ``later``, written by the log's own appender."""
    return [r for r, _ in replay.log_lines(_log(tmp_path, [*SPLINE_THEN, *later], name))]


TASK = {"kind": "set_task", "column": "y", "task": "regression"}


def test_a_task_or_an_event_answered_after_the_spline_is_a_later_record(tmp_path):
    """AR-V2: what a later record writes is read on the state folded up to the record before it,
    so a task or an event answered for the outcome after the spline is a later record, as an
    exclusion is; a task answered for another column writes nothing while `y` is the outcome
    (``SetTask`` stands only while its column is the target), and is not."""
    reads, _ = _slots()
    records = _after_the_spline(tmp_path, "a.jsonl", TASK,
                                {"kind": "set_event", "column": "y", "level": "1"},
                                {"kind": "set_task", "column": "bmi", "task": "regression"})
    assert replay.later_records(records, 4, reads) == (
        (5, "confirm_readings"), (6, "set_task"), (7, "set_event"))


def test_a_later_record_that_changed_what_the_size_is_read_under_is_named_with_the_change(
        tmp_path):
    """AR-V1: the outcome, its scale, its task (answered, else the one detected, here binary for
    `y`) and its event as the log ends against those at the spline's record (4): each later record
    whose step changed one is named, with the change from the record to the log's end; none where
    they are the same at both, a task answered as the detected one and a task changed and changed
    back among them."""
    reads, _ = _slots()
    detected = ("y", "binary")

    def changes(name: str, *later: dict):
        return replay.input_changes(_after_the_spline(tmp_path, name, *later), 4, reads, detected)

    assert changes("task.jsonl", TASK) == (((6, "set_task"),), ("the task from binary to regression",))
    assert changes("event.jsonl", {"kind": "set_event", "column": "y", "level": "1"}) == (
        ((6, "set_event"),), ("the event from none to `1`",))
    assert changes("scale.jsonl", {"kind": "set_outcome_scale", "column": "y", "scale": "log"}) == (
        ((6, "set_outcome_scale"),),
        ("the outcome from `y` to `ln_y`", "the outcome's scale from none to log"))
    assert changes("twice.jsonl", TASK, {"kind": "set_task", "column": "y", "task": "multiclass"}
                   ) == (((6, "set_task"), (7, "set_task")), ("the task from binary to multiclass",))
    # The same at both ends: nothing is named.
    assert changes("plain.jsonl", FIFTY_AND_OVER) == ((), ())
    assert changes("detected.jsonl", {**TASK, "task": "binary"}) == ((), ())
    assert changes("back.jsonl", TASK, {**TASK, "task": "binary"}) == ((), ())
    assert changes("other.jsonl", {"kind": "set_task", "column": "bmi", "task": "regression"}
                   ) == ((), ())


def test_a_later_record_that_writes_the_task_or_the_outcome_leaves_the_sizes_untraced(
        ages, tmp_path):
    """The guard: the sizes step by step read a state with no answered task by the task as the
    log ends, so where a later record writes the task or the outcome (here the detected task
    answered, and the outcome moved to `bmi` and back, the outcome and its task the same at both
    ends), each later record is named as one that can change them, not traced. With neither, the
    exclusion alone is named, traced (above)."""
    frame, store = ages
    reads, table = _slots()
    kept = events_or_not(frame[frame["age"] >= FIFTY])
    answered = _after_the_spline(tmp_path, "answered.jsonl", {**TASK, "task": "binary"},
                                 FIFTY_AND_OVER)
    assert replay.later_changes(answered, 4, reads, store, "binary", kept, table=table) == (
        ((5, "confirm_readings"), (6, "set_task"), (7, "set_exclusions")), False)
    moved = _after_the_spline(tmp_path, "moved.jsonl", {"kind": "set_target", "column": "bmi"},
                              FIFTY_AND_OVER, {"kind": "set_target", "column": "y"})
    assert replay.later_changes(moved, 4, reads, store, "binary", kept, table=table) == (
        ((5, "confirm_readings"), (6, "set_target"), (7, "set_exclusions"), (8, "set_target")),
        False)
    plain = _after_the_spline(tmp_path, "plain.jsonl", FIFTY_AND_OVER)
    assert replay.later_changes(plain, 4, reads, store, "binary", kept, table=table) == (
        ((6, "set_exclusions"),), True)


def _spline_log(tmp_path: Path, spline: dict, name: str = "decisions.jsonl") -> list:
    """A Predict log whose last record declares ``spline``, each sentence authored by the log's
    own appender (``voice.sentence_for`` on the state before it, the outcome detected yes/no)."""
    from turbotab.core import voice

    log = DecisionLog(tmp_path / name)
    for d in [{"kind": "set_lens", "lenses": ["clinical"]},
              {"kind": "set_target", "column": "y"},
              {"kind": "set_purpose", "purpose": "prediction"},
              spline]:
        log.append(d, sentence=lambda d, before: voice.sentence_for(d, before,
                                                                    {"detected_task": "binary"}))
    return replay.log_lines((tmp_path / name).read_bytes())


def test_a_refilled_record_states_the_new_size_in_the_sentence_the_log_authors_for_it(tmp_path):
    """The record replayed with this draw's size: every field as recorded but the size, and its
    sentence the one the log's own appender writes for a spline declared on that size (a second
    log, appended here); a ``set_forms`` record's other column is as recorded."""
    facts = {"detected_task": "binary"}
    for kind, spline, again in [
            ("set_exposure_form",
             {"kind": "set_exposure_form", "column": "age", "form": "spline", "knots": 5,
              "knots_rule": "harrell", "n_effective": 117.0},
             {"kind": "set_exposure_form", "column": "age", "form": "spline", "knots": 5,
              "knots_rule": "harrell", "n_effective": 114.0}),
            ("set_forms",
             {"kind": "set_forms", "forms": {
                 "age": {"form": "spline", "knots": 5, "knots_rule": "harrell",
                         "n_effective": 117.0},
                 "bmi": {"form": "linear"}}},
             {"kind": "set_forms", "forms": {
                 "age": {"form": "spline", "knots": 5, "knots_rule": "harrell",
                         "n_effective": 114.0},
                 "bmi": {"form": "linear"}}})]:
        lines = _spline_log(tmp_path / kind, spline)
        record, line = lines[-1]
        assert "here 117, the smaller of the events and the non-events" in record.sentence
        new = json.loads(replay.refilled_line(record, {"age": 114.0},
                                              [r for r, _ in lines[:-1]], facts))
        expected = json.loads(_spline_log(tmp_path / kind, again, "b.jsonl")[-1][1])
        old = json.loads(line)
        assert {k: v for k, v in new.items() if k not in ("decision", "sentence")} == {
            k: v for k, v in old.items() if k not in ("decision", "sentence")}
        assert new["decision"] == expected["decision"]
        assert new["sentence"] == expected["sentence"] != old["sentence"]
        assert ("k = 5 by Harrell's rule (3 knots below an effective sample size of 30, 5 from "
                "100, else 4; here 114, the smaller of the events and the non-events)"
                ) in new["sentence"] and "117" not in new["sentence"]


def test_a_record_whose_sentence_the_log_would_not_author_now_is_not_refilled(tmp_path):
    """Right or refused: the sentence is authored again only where the log's sentence function
    gives the record's own sentence on its own size, so the size is all that changes; a record
    whose sentence differs (another version's words) refuses the rebuild, naming it."""
    lines = _spline_log(tmp_path, {"kind": "set_exposure_form", "column": "age",
                                   "form": "spline", "knots": 5, "knots_rule": "harrell",
                                   "n_effective": 117.0})
    record, _ = lines[-1]
    before = [r for r, _ in lines[:-1]]
    replay.refilled_line(record, {"age": 114.0}, before, {"detected_task": "binary"})  # silent
    other = record.model_copy(update={"sentence": record.sentence.replace("restricted", "natural")})
    assert other.sentence != record.sentence
    with pytest.raises(replay.Refused, match=r"^Record 4 \(set_exposure_form\)'s sentence is not "
                                             r"the one"):
        replay.refilled_line(other, {"age": 114.0}, before, {"detected_task": "binary"})

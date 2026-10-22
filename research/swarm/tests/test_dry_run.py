"""The driver's dry run (DRIVER_PLAN D6; ``research/swarm/dry_run.py``): its expectations, its data
files, and its committed records, whose checks must still give what each run's construction
implies. The runs themselves are ``python -m research.swarm.dry_run``, a heavy run, never a test."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.swarm import checks as C
from research.swarm import dry_run as D
from research.swarm.record import validate

DRYRUN = D.OUT


def test_each_run_expects_a_result_with_its_reason_for_every_check():
    specs = D.specs()
    assert set(D.RUNS) == set(specs) == set(D.EXPECTED) == set(D.WHY)
    for run in D.RUNS:
        assert set(D.EXPECTED[run]) == set(D.WHY[run]) == set(D.CHECKS), run
        assert set(D.EXPECTED[run].values()) <= {"pass", "fail", "not_checked"}, run
    # One run explores, the others pass once: the single-pass check is seen both ways.
    assert {run: D.EXPECTED[run]["single_pass"] for run in D.RUNS} == {
        run: "pass" if specs[run].explore else "fail" for run in D.RUNS}


@pytest.mark.parametrize("run", D.RUNS)
def test_each_runs_table_is_rebuilt_to_its_manifests_sha256(tmp_path, run, monkeypatch):
    """A fixture copied, the decoy generated from its seed: each file's SHA-256 is the one its
    intake manifest records (hashlib here, the manifest's own number). A manifest whose digest
    is not the file's is refused."""
    dataset = D.specs()[run].dataset
    manifest = json.loads((D.DATASETS / dataset / "manifest.json").read_text("utf-8"))
    path = D.table_of(dataset, tmp_path / "data")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["table"]["sha256"]
    wrong = tmp_path / "datasets" / dataset
    wrong.mkdir(parents=True)
    (wrong / "manifest.json").write_text(json.dumps(
        {**manifest, "table": {**manifest["table"], "sha256": "0" * 64}}), "utf-8")
    monkeypatch.setattr(D, "DATASETS", tmp_path / "datasets")
    with pytest.raises(RuntimeError, match="is not the manifest's table"):
        D.table_of(dataset, tmp_path / "again")


def test_the_committed_dry_run_records_still_give_what_their_construction_implies():
    """Each run's committed record validates, and every check run on it today gives the run's
    expected result (``dry_run.EXPECTED``): a change to a check, the record or the schema that
    moves one fails here, and the dry run is run again (a heavy run), never re-expected."""
    table = json.loads((DRYRUN / "comparison.json").read_text("utf-8"))
    assert [r["run"] for r in table] == list(D.RUNS)
    for row in table:
        assert row["valid"] and row["error"] is None, row["run"]
        rec = validate(json.loads((D.REPO / row["record"]).read_text("utf-8")))
        assert rec.run.id == row["run_id"] and rec.dataset.id == row["dataset"]
        got = {o.check: o.outcome for o in C.run_checks([rec])}
        assert got == D.EXPECTED[row["run"]], row["run"]
        assert all(c["agrees"] for c in row["checks"]), row["run"]
        assert not row["refused"] or "not_for_the_analyst" not in row["refused"]


def test_the_docstring_counts_the_exploring_runs_fits_as_its_record_holds_them():
    """DRV-FIX (h): the exploring run presses Fit three times, and its third press, for answers
    already fitted, makes no new fit, so its record holds two; the module docstring says two."""
    words = {1: "one", 2: "two", 3: "three", 4: "four"}
    table = json.loads((DRYRUN / "comparison.json").read_text("utf-8"))
    explores = [r for r in table if D.specs()[r["run"]].explore]
    assert len(explores) == 1
    rec = validate(json.loads((D.REPO / explores[0]["record"]).read_text("utf-8")))
    text = " ".join(D.__doc__.split())
    assert f"{words[rec.run.fits]} fits)" in text
    assert [f"{w} fits)" in text for w in words.values()].count(True) == 1


def test_each_scripted_answer_is_tagged_with_a_marker_of_the_vocabulary():
    """Record version 2: the scripted analyst's tag on each answer, by its question, is a marker
    the vocabulary holds (the driver refuses any other)."""
    from research.swarm import vocabulary as V

    assert set(D.MARKER_OF.values()) | {D.DEFAULT_MARKER} <= set(V.NAMES)
    from turbotab.core.interview import QUESTION_KEYS

    assert set(D.MARKER_OF) <= set(QUESTION_KEYS)


def test_no_run_is_started_with_a_label_of_its_own():
    """DRV-FIX (c): the dry run's names (``decoy_single``) never reach its analyst: each committed
    record's run id is one ``start`` made itself, ``run_`` and a counter, and names no run of the
    dry run, no dataset and no control kind."""
    from research.swarm import driver

    table = json.loads((DRYRUN / "comparison.json").read_text("utf-8"))
    for row in table:
        label = row["run_id"].rsplit("-", 1)[0]
        assert driver.RUN_ID.fullmatch(row["run_id"]) and label.startswith("run_"), row["run_id"]
        assert label[4:].isdigit()

"""The shuffled control's corpus (SWARM_PLAN §9 step 5; ``shuffle.py``), on a corpus built here: three
constructed records on three datasets, each with markers of its own (its rows, 101, 202 or 303),
so where each record's markers came from is known by construction. Each guard both ways: no record
keeps its dataset's markers, the corpus is never written, the planted records are in the copy
only and listed in the sealed file outside it, and what cannot be shuffled is refused."""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from research.swarm import shuffle as S
from research.swarm.record import validate
from research.swarm.tests.constructed import (estimate_record, marker, predict_record,
                                              revert_record)

ROWS = {"d1": 101, "d2": 202, "d3": 303}


def corpus(folder: Path, datasets: tuple[str, ...] = ("d1", "d2", "d3")) -> Path:
    folder.mkdir(parents=True)
    makers = [predict_record, estimate_record, revert_record]
    for i, dataset in enumerate(datasets):
        rec = makers[i % len(makers)]()
        rec["run"]["id"] = f"run_{i + 1:02d}"
        rec["dataset"]["id"] = dataset
        rec["regime"]["n_rows"] = ROWS[dataset]
        rec["regime"]["markers"] = [marker("rows", ROWS[dataset]),
                                    marker("rows_analyzed", ROWS[dataset] - 1)]
        rec["dataset"]["declared"] = [{"name": "instrument", "status": "declared",
                                       "value": f"device {dataset}", "quote": f"by {dataset}",
                                       "source": f"{dataset} methods"}]
        for a in rec["attempts"]:
            a["markers"] = [marker("rows", ROWS[dataset])]
        (folder / f"{rec['run']['id']}.json").write_text(json.dumps(rec), "utf-8")
    (folder / "failed").mkdir()
    (folder / "failed" / "run_99.json").write_text("{}", "utf-8")  # a failed run: not a record
    return folder


def by_dataset(folder: Path) -> dict[str, dict]:
    return {json.loads(p.read_text())["dataset"]["id"]: json.loads(p.read_text())
            for p in folder.glob("*.json")}


def test_every_record_takes_another_datasets_markers_whole(tmp_path):
    src = corpus(tmp_path / "corpus")
    before = {p.name: p.read_bytes() for p in src.rglob("*.json")}
    body = S.write(src, tmp_path / "copy", tmp_path / "sealed" / "s.json", seed=3, k=0)
    mapping = body["derangement"]
    assert sorted(mapping) == sorted(mapping.values()) == ["d1", "d2", "d3"]
    assert all(a != b for a, b in mapping.items())  # a derangement: no dataset keeps its own
    got = by_dataset(tmp_path / "copy")
    assert set(got) == {"d1", "d2", "d3"}
    for dataset, rec in got.items():
        donor = mapping[dataset]
        assert rec["regime"]["n_rows"] == ROWS[donor]
        assert [m["value"] for m in rec["regime"]["markers"]] == [ROWS[donor], ROWS[donor] - 1]
        assert rec["dataset"]["declared"][0]["value"] == f"device {donor}"
        assert all(a["markers"][0]["value"] == ROWS[donor] for a in rec["attempts"])
        validate(rec)
    # The rest of each record is its own: its run, its decisions, its result.
    original = by_dataset(src)
    for dataset, rec in got.items():
        assert rec["run"] == original[dataset]["run"]
        assert rec["decisions"] == original[dataset]["decisions"]
    assert {p.name: p.read_bytes() for p in src.rglob("*.json")} == before  # never written


def test_the_same_seed_gives_the_same_copy_and_another_seed_may_not(tmp_path):
    src = corpus(tmp_path / "corpus")
    a = S.write(src, tmp_path / "a", tmp_path / "sa.json", seed=11, k=0)["derangement"]
    b = S.write(src, tmp_path / "b", tmp_path / "sb.json", seed=11, k=0)["derangement"]
    assert a == b
    seen = {json.dumps(S.derangement(["d1", "d2", "d3"], random.Random(s)), sort_keys=True)
            for s in range(40)}
    assert len(seen) == 2  # both 3-cycles are drawn, and nothing else: each moves every dataset


def test_one_dataset_cannot_be_shuffled(tmp_path):
    src = corpus(tmp_path / "corpus", datasets=("d1",))
    with pytest.raises(S.ShuffleRefused, match="at least two"):
        S.write(src, tmp_path / "copy", tmp_path / "s.json", seed=1, k=0)


def test_the_planted_records_are_in_the_copy_only_and_listed_in_the_sealed_file(tmp_path):
    src = corpus(tmp_path / "corpus")
    sealed = tmp_path / "sealed" / "s.json"
    body = S.write(src, tmp_path / "copy", sealed, seed=5, k=6)
    planted = body["planted"]["records"]
    assert len(planted) == 6 and json.loads(sealed.read_text())["planted"]["records"] == planted
    copy = {p.stem for p in (tmp_path / "copy").glob("*.json")}
    assert set(planted) <= copy and len(copy) == 3 + 6
    assert not set(planted) & {p.stem for p in src.rglob("*.json")}
    effect = body["planted"]["effect"]
    for run_id in planted:
        rec = validate(json.loads((tmp_path / "copy" / f"{run_id}.json").read_text()))
        assert rec.run.id.startswith("run_") and not rec.dataset.id.startswith(("d1", "d2", "d3"))


def test_the_planted_effect_is_one_true_effect_by_construction(tmp_path):
    """Below the threshold the decision moved the score materially, above it by nothing: the
    planted records split exactly so."""
    src = corpus(tmp_path / "corpus")
    body = S.write(src, tmp_path / "copy", tmp_path / "s.json", seed=8, k=6)
    effect = body["planted"]["effect"]
    below, above = [], []
    for run_id in body["planted"]["records"]:
        rec = json.loads((tmp_path / "copy" / f"{run_id}.json").read_text())
        value = next(m["value"] for m in rec["regime"]["markers"] if m["name"] == effect["marker"])
        moved = next(d for d in rec["decisions"] if d["id"] == effect["decision_id"]
                     and d["materiality"]["source"] == "fits")
        size = moved["materiality"]["movement"][0]["movement"]["value"]
        change = rec["attempts"][-1]["change"][0]
        (below if value < effect["threshold"] else above).append((size, change["excludes_zero"]))
    assert len(below) == len(above) == 3
    assert all(s == S.MATERIAL and ex for s, ex in below)
    assert all(s == S.NONE and not ex for s, ex in above)


@pytest.mark.parametrize("where", ["inside", "corpus", "existing", "sealed_exists"])
def test_what_would_leak_or_overwrite_is_refused(tmp_path, where):
    src = corpus(tmp_path / "corpus")
    out, sealed = tmp_path / "copy", tmp_path / "s.json"
    if where == "inside":
        sealed = out / "s.json"
    if where == "corpus":
        out = src
    if where == "existing":
        out.mkdir()
        (out / "x.json").write_text("{}")
    if where == "sealed_exists":
        sealed.write_text("{}")
    with pytest.raises(S.ShuffleRefused):
        S.write(src, out, sealed, seed=1, k=2)


def test_the_command_line_writes_the_copy_and_says_where(tmp_path, capsys):
    src = corpus(tmp_path / "corpus")
    assert S.main([str(src), "--out", str(tmp_path / "copy"), "--sealed",
                   str(tmp_path / "s.json"), "--seed", "2", "--planted", "4"]) == 0
    said = json.loads(capsys.readouterr().out)
    assert said == {"copy": str(tmp_path / "copy"), "records": 3, "planted": 4,
                    "sealed": str(tmp_path / "s.json")}
    assert S.main([str(src), "--out", str(tmp_path / "copy"), "--sealed",
                   str(tmp_path / "t.json")]) == 1  # the copy is written once

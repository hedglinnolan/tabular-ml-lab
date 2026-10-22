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
        if makers[i % len(makers)] is predict_record:
            planted_scores(rec)
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


# 100 training rows in five random folds of 20 (``signed.test_share``: 1/4).
FIVE = {"n_train": 100, "folds": 5, "repeats": 1, "fold_scheme": "random", "validation": "kfold",
        "grouped_by": None}


def planted_scores(rec: dict) -> None:
    """The constructed Predict record's two fits (fit2's whole change is the analyst's record #4)
    with log loss fold scores of their own, and fit2's movement and signed change as the engine and
    the assembler compute them from those scores."""
    from turbotab.core import runs

    from research.swarm.assemble import signed_change

    first, second = rec["attempts"]
    for a, folds in ((first, [0.52, 0.55, 0.50, 0.58, 0.54]),
                     (second, [0.50, 0.52, 0.49, 0.55, 0.53])):
        a["primary_metric"] = "log_loss"
        m = sum(folds) / 5
        a["scores"] = {"linear": {"log_loss": {
            "estimate": m, "estimator": "fold_mean", "mean": m, "folds": folds, "repeats": 1,
            "sd": (sum((x - m) ** 2 for x in folds) / 4) ** 0.5, "se": 0.02, "ci_low": m - 0.04,
            "ci_high": m + 0.04, "repeat_sd": None}}}
    moved, why = runs.movement(
        runs.FitRecord.model_validate({f: first.get(f) for f in runs.FitRecord.model_fields
                                       if f in first}),
        purpose="prediction", target=second["target"], metric="log_loss", folds=second["folds"],
        seed=second["seed"], scores=second["scores"])
    second["movement"], second["movement_why"] = [m.model_dump(mode="json") for m in moved], why
    second["change"], second["change_why"] = signed_change(second, first, {"folds": FIVE})
    rec["decisions"][3]["materiality"] = {"source": "fits", "attempt": "fit2",
                                          "movement": second["movement"]}
    rec["result"]["metric"] = "log_loss"


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
        # A run id as ``start`` draws one, on one of the copy's datasets (the verifier's P6).
        assert rec.run.id.startswith("run_") and rec.dataset.id in {"d1", "d2", "d3"}


def test_the_planted_effect_is_one_true_effect_by_construction(tmp_path):
    """Below the threshold the decision moved the score materially (a better score, its paired
    interval excluding zero), above it by little (an interval holding zero): the planted records
    split exactly so, and the sealed file says each one's side."""
    src = corpus(tmp_path / "corpus")
    body = S.write(src, tmp_path / "copy", tmp_path / "s.json", seed=8, k=6)
    effect = body["planted"]["effect"]
    below, above = [], []
    sides = {r["run"]: r["below"] for r in effect["records"]}
    for run_id in body["planted"]["records"]:
        rec = json.loads((tmp_path / "copy" / f"{run_id}.json").read_text())
        value = next(m["value"] for m in rec["regime"]["markers"] if m["name"] == effect["marker"])
        moved = next(d for d in rec["decisions"] if d["id"] == effect["decision_id"])
        assert moved["materiality"]["source"] == "fits"
        size = next(m for m in moved["materiality"]["movement"]
                    if m["family"] == effect["family"])["movement"]["value"]
        change = next(c for c in rec["attempts"][-1]["change"] if c["family"] == effect["family"])
        side = below if value < effect["threshold"] else above
        assert sides[run_id] == (side is below)
        side.append((size, change["excludes_zero"], change["better"]))
    assert len(below) == len(above) == 3
    assert all(ex and better is True for _, ex, better in below)
    assert all(not ex for _, ex, _ in above)
    assert min(s for s, _, _ in below) > max(s for s, _, _ in above)


def _form(obj):
    """What a record's JSON holds, by key and value type, its lists by their items' forms."""
    if isinstance(obj, dict):
        return {k: _form(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_form(v) for v in obj]
    return "number" if isinstance(obj, float) else type(obj).__name__


def test_no_planted_record_differs_in_form_from_a_real_one_beyond_the_planted_effect(tmp_path):
    """The verifier's P6: a planted record's ids are drawn as a real record's are, its dataset is
    one of the copy's, each value keeps its type, each change is against the fit before with
    ``better`` by the metric's direction, and its movement, materiality and signed change are what
    the engine and the assembler compute from its own fold scores."""
    from turbotab.core import runs
    from turbotab.core.models.metrics import higher_is_better

    from research.swarm import signed
    from research.swarm.driver import RUN_ID

    src = corpus(tmp_path / "corpus")
    body = S.write(src, tmp_path / "copy", tmp_path / "s.json", seed=8, k=6)
    effect = body["planted"]["effect"]
    copy = {json.loads(p.read_text())["run"]["id"]: json.loads(p.read_text())
            for p in (tmp_path / "copy").glob("*.json")}
    real = {i: r for i, r in copy.items() if i not in body["planted"]["records"]}
    template = real[effect["template"]]
    labels = {i.rsplit("-", 1)[0] for i in real}
    datasets = {r["dataset"]["id"] for r in real.values()}
    for run_id in body["planted"]["records"]:
        rec = copy[run_id]
        validate(rec)
        assert RUN_ID.fullmatch(run_id) and run_id.rsplit("-", 1)[0] in labels
        assert rec["dataset"] == template["dataset"] and rec["dataset"]["id"] in datasets
        assert _form(rec) == _form(template)  # every key, and each value's type, the template's
        for m in rec["regime"]["markers"]:
            if m["name"] == effect["marker"]:
                assert type(m["value"]) is type(effect["threshold"])
                assert not m["name"].endswith("_share") or 0 <= m["value"] <= 1
        # No time of the template's is kept, and the record's own order of times is.
        times = [t for t in signed_times(rec)]
        assert not set(times) & set(signed_times(template))
        attempts = rec["attempts"]
        for i, a in enumerate(attempts):
            for c in a["change"]:
                assert i and c["against"] == attempts[i - 1]["id"] != a["id"]
                if c["difference"] is not None and c["difference"] != 0:
                    assert c["better"] == ((c["difference"] > 0) == higher_is_better(c["metric"]))
        last, previous = attempts[-1], attempts[-2]
        metric = last["primary_metric"]
        moved, _ = runs.movement(
            runs.FitRecord.model_validate({f: previous.get(f) for f in runs.FitRecord.model_fields
                                           if f in previous}),
            purpose="prediction", target=last["target"], metric=metric, folds=last["folds"],
            seed=last["seed"], scores=last["scores"])
        assert last["movement"] == [m.model_dump(mode="json") for m in moved]
        mine = next(d for d in rec["decisions"] if d["id"] == effect["decision_id"])
        assert mine["materiality"] == {"source": "fits", "attempt": last["id"],
                                       "movement": last["movement"]}
        family = effect["family"]
        planted_change = next(c for c in last["change"] if c["family"] == family)
        recomputed = signed.change(last["scores"][family][metric],
                                   previous["scores"][family][metric], same_folds=True,
                                   higher_is_better=higher_is_better(metric), metric=metric,
                                   after_split=FIVE)
        assert {k: planted_change[k] for k in recomputed} == pytest.approx(recomputed)
        score = last["scores"][family][metric]
        assert score["estimate"] == pytest.approx(sum(score["folds"]) / len(score["folds"]))


def signed_times(rec):
    from research.swarm.record import STAMP

    return [m.group() for m in STAMP.finditer(json.dumps({k: v for k, v in rec.items()
                                                          if k != "dataset"}))]


def test_a_corpus_with_no_record_to_plant_in_is_refused(tmp_path):
    src = corpus(tmp_path / "corpus")
    for path in src.glob("*.json"):
        rec = json.loads(path.read_text())
        rec["attempts"] = rec["attempts"][:1]  # one fit: no fit before to change from
        rec["attempts"][0]["change"], rec["attempts"][0]["change_why"] = [], "first fit"
        for d in rec["decisions"]:
            if d["materiality"].get("source") == "fits":
                d["materiality"] = {"source": None, "why": "constructed: none measured"}
        rec["result"]["attempt"] = rec["attempts"][0]["id"] if rec["result"].get("attempt") else None
        path.write_text(json.dumps(rec))
    with pytest.raises(S.ShuffleRefused, match="No record of the copy"):
        S.write(src, tmp_path / "copy", tmp_path / "s.json", seed=1, k=2)


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


@pytest.mark.parametrize("out, sealed", [
    ("corpus/shuffled", "s.json"),      # the copy inside the corpus: its records would be read
    ("corpus/a/b", "s.json"),           # deeper inside it
    ("copy", "corpus/sealed.json"),     # the sealed file inside the corpus the theorist reads
    ("copy", "corpus/x/sealed.json"),
    ("copy", "copy/x/sealed.json"),     # the sealed file deeper inside the copy
    (".", "s.json"),                    # the copy above the corpus
])
def test_a_copy_or_a_sealed_file_at_or_under_the_corpus_or_a_sealed_file_in_the_copy_is_refused(
        tmp_path, out, sealed):
    """The verifier's P7: nothing the shuffle writes lands where the corpus is read, and the
    sealed file never lands where the copy is read; the corpus is left as it was."""
    src = corpus(tmp_path / "corpus")
    before = sorted(p.relative_to(src) for p in src.rglob("*"))
    with pytest.raises(S.ShuffleRefused):
        S.write(src, tmp_path / out, tmp_path / sealed, seed=1, k=2)
    assert sorted(p.relative_to(src) for p in src.rglob("*")) == before


@pytest.mark.parametrize("out, sealed", [
    ("copy", "s.json"),
    ("elsewhere/copy", "sealed/s.json"),
    ("corpus-copy", "corpus-sealed/s.json"),  # a sibling whose name begins as the corpus's does
])
def test_a_copy_and_a_sealed_file_apart_from_the_corpus_and_each_other_are_written(
        tmp_path, out, sealed):
    src = corpus(tmp_path / "corpus")
    S.write(src, tmp_path / out, tmp_path / sealed, seed=1, k=2)
    assert len(list((tmp_path / out).glob("*.json"))) == 3 + 2 and (tmp_path / sealed).is_file()


def test_the_command_line_writes_the_copy_and_says_where(tmp_path, capsys):
    src = corpus(tmp_path / "corpus")
    assert S.main([str(src), "--out", str(tmp_path / "copy"), "--sealed",
                   str(tmp_path / "s.json"), "--seed", "2", "--planted", "4"]) == 0
    said = json.loads(capsys.readouterr().out)
    assert said == {"copy": str(tmp_path / "copy"), "records": 3, "planted": 4,
                    "sealed": str(tmp_path / "s.json")}
    assert S.main([str(src), "--out", str(tmp_path / "copy"), "--sealed",
                   str(tmp_path / "t.json")]) == 1  # the copy is written once

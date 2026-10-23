"""The shuffled control's corpus (SWARM_PLAN §9 step 5; ``shuffle.py``), on a corpus built here: three
constructed records on three datasets, each with markers of its own (its rows, 101, 202 or 303),
so where each record's markers came from is known by construction. Each guard both ways: no record
keeps its dataset's markers, the corpus is never written, the planted records are in the copy
only and listed in the sealed file outside it, the sealed file names every marker that separates
the planted records as M does and no other (against a brute force over the copy's files), and
what cannot be shuffled is refused."""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from research.swarm import shuffle as S
from research.swarm.record import neutral_id, validate
from research.swarm.tests.constructed import (estimate_record, marker, predict_record,
                                              revert_record)

ROWS = {"d1": 101, "d2": 202, "d3": 303}


PREDICTORS = 6  # each record's predictors, so its predictors per row is this over its rows analyzed


def corpus(folder: Path, datasets: tuple[str, ...] = ("d1", "d2", "d3"), per: int = 1) -> Path:
    """One record per dataset (``per`` 1: its rows ``ROWS``, its rows analyzed one fewer); or
    ``per`` records per dataset, each of the dataset's own kind, its rows the dataset's and its rows
    analyzed nine fewer for each record before it, its predictors per row and its count of rows
    analyzed read from them, as the engine reads a real record's: a marker that varies across a
    dataset's records beside one that does not."""
    folder.mkdir(parents=True)
    makers = [predict_record, estimate_record, revert_record]
    for i, dataset in enumerate(datasets):
        for r in range(per):
            rec = makers[i % len(makers)]()
            if makers[i % len(makers)] is predict_record:
                planted_scores(rec)
            n = i * per + r + 1
            rec["run"]["id"] = f"run_{i + 1:02d}" if per == 1 else f"run_{n:02d}-{n:032x}"
            rec["dataset"]["id"] = dataset
            rec["regime"]["n_rows"] = ROWS[dataset]
            analyzed = ROWS[dataset] - 1 - 9 * r
            rec["regime"]["markers"] = [marker("rows", ROWS[dataset]),
                                        marker("rows_analyzed", analyzed)]
            fits = [marker("rows", ROWS[dataset])]
            if per > 1:
                rec["regime"]["n_analyzed"] = analyzed
                ratio = marker("predictors_per_row", PREDICTORS / analyzed)
                rec["regime"]["markers"].append(ratio)
                fits = [*rec["regime"]["markers"]]
            rec["dataset"]["declared"] = [{"name": "instrument", "status": "declared",
                                           "value": f"device {dataset}", "quote": f"by {dataset}",
                                           "source": f"{dataset} methods"}]
            for a in rec["attempts"]:
                a["markers"] = json.loads(json.dumps(fits))
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


# ── the purpose and its loss: the record's own ──────────────────────────────

DRY_RUN = Path(__file__).resolve().parents[1] / "dryrun"


def purposed(folder: Path) -> Path:
    """Each record of ``folder`` given the live markers of the purpose and its loss as the driver
    reads them (``markers.py``): its purpose; under Predict its primary score, the metric its newest
    fit names, and no estimand; under Estimate an estimand on a measure of its own and no primary
    score. At each fit, the same, the primary score the metric of that fit."""
    from research.swarm.markers import marker as read

    for i, path in enumerate(sorted(folder.glob("*.json"))):
        rec = json.loads(path.read_text())
        purpose = rec["question"]["purpose"]

        def axis(metric):
            if purpose == "prediction":
                return [read("purpose", purpose, "constructed"),
                        read("primary_score", {"metric": metric}, "constructed"),
                        read("estimand", why="No estimand is declared.")]
            return [read("purpose", purpose, "constructed"),
                    read("primary_score", why="Under Estimate the result is the estimate."),
                    read("estimand", {"exposure": f"x{i}", "measure": "odds_ratio"}, "constructed")]

        named = [a["primary_metric"] for a in rec["attempts"] if a["primary_metric"]]
        rec["regime"]["markers"] = [*rec["regime"]["markers"],
                                    *axis(named[-1] if named else "auc")]
        for a in rec["attempts"]:
            a["markers"] = [*(a["markers"] or []), *axis(a["primary_metric"] or "auc")]
        validate(rec)
        path.write_text(json.dumps(rec), "utf-8")
    return folder


AXIS = ("purpose", "primary_score", "estimand")


def disagreements(v: dict) -> list[tuple[str, str]]:
    """Where a view's live markers of the purpose and its loss contradict its own question or
    fits: a purpose not its question's; a primary score under Estimate, or one naming a metric no
    fit of its scored; an estimand under Predict."""
    purpose = v["question"]["purpose"]
    metrics = {a["primary_metric"] for a in v["attempts"] if a["primary_metric"]}
    out = []
    places = [("regime", v["regime"]["markers"])] + [(f"fit {a['seq']}", a["markers"] or [])
                                                     for a in v["attempts"]]
    for where, markers in places:
        held = {m["name"]: m["value"] for m in markers}
        if held.get("purpose") is not None and held["purpose"] != purpose:
            out.append((where, "purpose"))
        score = held.get("primary_score")
        if score is not None and (purpose != "prediction" or score.get("metric") not in metrics):
            out.append((where, "primary_score"))
        if held.get("estimand") is not None and purpose != "inference":
            out.append((where, "estimand"))
    return out


def test_no_view_of_the_copy_holds_a_purpose_its_own_question_and_fits_contradict(tmp_path):
    """The verifier's V6D-1: the copy permutes the markers across datasets, and the purpose and its
    loss restate a record's own question and fits, which the copy keeps its own. On the dry run's
    three records (two Predict, one Estimate) and on the constructed corpus, under every seed's
    derangement, no record's view holds a purpose, a primary score or an estimand its own question
    or fits contradict. The construction: in the corpus itself none does, and the copy permutes
    the datasets."""
    from research.swarm.claim_cycle import view as W

    dry = tmp_path / "dry"
    dry.mkdir()
    for path in sorted(DRY_RUN.glob("run_0*.json")):
        (dry / path.name).write_bytes(path.read_bytes())
    built = purposed(corpus(tmp_path / "built"))
    for src in (dry, built):
        purposes = {v["question"]["purpose"] for v in W.read(src)}
        assert purposes == {"prediction", "inference"}  # a corpus of both purposes
        assert not [d for v in W.read(src) for d in disagreements(v)]
        mappings = set()
        for seed in range(8):
            out = tmp_path / f"{src.name}-{seed}"
            body = S.write(src, out, tmp_path / f"{src.name}-{seed}.json", seed=seed, k=0)
            mappings.add(json.dumps(body["derangement"], sort_keys=True))
            found = [(v["run"]["id"], d) for v in W.read(out) for d in disagreements(v)]
            assert found == [], (src.name, seed)
        assert len(mappings) == 2  # both of the three datasets' derangements were drawn


def test_the_copy_keeps_a_records_own_purpose_markers_and_takes_the_rest_from_its_donor(tmp_path):
    """The rule itself, by construction: in the copy each record's live markers of the purpose and
    its loss, at the regime and at each fit, are its own in the corpus, and every other marker is
    its donor's; and a regime marker the copy permutes still moves (each record's rows are its
    donor's)."""
    src = purposed(corpus(tmp_path / "corpus"))
    body = S.write(src, tmp_path / "copy", tmp_path / "s.json", seed=3, k=0)
    mine, copied = by_dataset(src), by_dataset(tmp_path / "copy")
    own = lambda ms: [m for m in ms or [] if m["name"] in AXIS]  # noqa: E731
    rest = lambda ms: [m for m in ms or [] if m["name"] not in AXIS]  # noqa: E731
    for dataset, rec in copied.items():
        donor = mine[body["derangement"][dataset]]
        assert own(rec["regime"]["markers"]) == own(mine[dataset]["regime"]["markers"])
        assert rest(rec["regime"]["markers"]) == rest(donor["regime"]["markers"])
        assert rec["regime"]["n_rows"] == ROWS[body["derangement"][dataset]] != ROWS[dataset]
        fits = donor["attempts"]
        for j, a in enumerate(rec["attempts"]):
            assert own(a["markers"]) == own(mine[dataset]["attempts"][j]["markers"])
            assert rest(a["markers"]) == rest(fits[j % len(fits)]["markers"])
        validate(rec)


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
    src = corpus(tmp_path / "corpus", per=3)
    sealed = tmp_path / "sealed" / "s.json"
    body = S.write(src, tmp_path / "copy", sealed, seed=5, k=6)
    planted = body["planted"]["records"]
    assert len(planted) == 6 and json.loads(sealed.read_text())["planted"]["records"] == planted
    copy = {p.stem for p in (tmp_path / "copy").glob("*.json")}
    assert set(planted) <= copy and len(copy) == 9 + 6
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
    src = corpus(tmp_path / "corpus", per=3)
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
    the engine and the assembler compute from its own fold scores. Its regime, and each fit's live
    markers, are a real record's of its dataset in the copy (the verifier's V6D-2), so their form
    is that record's, and the rest is its template's."""
    from turbotab.core import runs
    from turbotab.core.models.metrics import higher_is_better

    from research.swarm import signed
    from research.swarm.driver import RUN_ID

    src = corpus(tmp_path / "corpus", per=3)
    body = S.write(src, tmp_path / "copy", tmp_path / "s.json", seed=8, k=6)
    effect = body["planted"]["effect"]
    sources = {r["run"]: r["markers_from"] for r in effect["records"]}
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
        mine = {k: v for k, v in rec.items() if k != "regime"}
        assert _form(mine) == _form({k: v for k, v in template.items() if k != "regime"})
        assert _form(rec["regime"]) == _form(real[sources[run_id]]["regime"])
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


def _markers_whole(regime: dict) -> str:
    return json.dumps(regime, sort_keys=True)


def test_each_planted_record_holds_a_real_records_markers_whole(tmp_path):
    """The verifier's V6D-2. A planted record's regime, readings and markers, is that of a real
    record of its dataset in the copy, the one the sealed file names (``markers_from``), on the side
    of the threshold the sealed file says; M is a marker whose value varies across the dataset's
    real records, never one constant across them (`rows`), and t is one of their values, so no
    planted record holds a value, or a pair of values (more rows analyzed than rows; predictors per
    row not its predictors over its rows analyzed), no real record of its dataset holds. Both ways:
    in the view, each planted record's markers are a real record's of its dataset, and each real
    record's are its own."""
    from research.swarm.claim_cycle import view as W

    src = corpus(tmp_path / "corpus", per=3)
    markers = set()
    for seed in range(40):
        out, sealed = tmp_path / f"copy{seed}", tmp_path / f"s{seed}.json"
        body = S.write(src, out, sealed, seed=seed, k=6)
        effect = body["planted"]["effect"]
        markers.add(effect["marker"])
        planted = set(body["planted"]["records"])
        copied = {json.loads(p.read_text())["run"]["id"]: json.loads(p.read_text())
                  for p in out.glob("*.json")}
        dataset = copied[effect["template"]]["dataset"]["id"]
        real = {i: r for i, r in copied.items()
                if i not in planted and r["dataset"]["id"] == dataset}
        values = {r["regime"]["markers"][[m["name"] for m in r["regime"]["markers"]].index(
            effect["marker"])]["value"] for r in real.values()}
        assert len(values) > 1 and effect["threshold"] in values  # M varies; t is a real value
        for entry in effect["records"]:
            rec, source = copied[entry["run"]], real[entry["markers_from"]]
            assert rec["regime"] == source["regime"]
            assert [a["markers"] for a in rec["attempts"]] == [
                source["attempts"][j % len(source["attempts"])]["markers"]
                for j in range(len(rec["attempts"]))]
            held = {m["name"]: m["value"] for m in rec["regime"]["markers"]}
            assert (held[effect["marker"]] < effect["threshold"]) == entry["below"]
            assert held[effect["marker"]] == entry["value"]
            assert held["rows_analyzed"] <= held["rows"]
            assert held["predictors_per_row"] == PREDICTORS / held["rows_analyzed"]
        views = W.read(out)
        whole = {}
        for v in views:
            if v["run"]["id"] not in planted:
                whole.setdefault(v["dataset"]["id"], set()).add(_markers_whole(v["regime"]))
        for v in views:
            assert _markers_whole(v["regime"]) in whole[v["dataset"]["id"]], (seed, v["run"]["id"])
    assert markers == {"rows_analyzed", "predictors_per_row"}  # never `rows`, constant per dataset


def _planted_sides(out: Path, effect: dict, at: str) -> dict[str, dict[bool, list]]:
    """Each live marker's values across the planted records in the copy at ``out``, by side
    (material below M's threshold, True; none at or above it, False), read from the records' own
    files at the regime or at the last fit, and only where every planted record holds a number,
    a word or true or false for it (a region bounds nothing else)."""
    by: dict[str, dict[bool, list]] = {}
    held: dict[str, int] = {}
    for entry in effect["records"]:
        rec = json.loads((out / f"{entry['run']}.json").read_text())
        markers = rec["regime"]["markers"] if at == "regime" else rec["attempts"][-1]["markers"]
        for m in markers or []:
            v = m.get("value")
            if m.get("timing") == "live" and (isinstance(v, (str, bool, int, float))):
                by.setdefault(m["name"], {True: [], False: []})[entry["below"]].append(v)
                held[m["name"]] = held.get(m["name"], 0) + 1
    return {n: s for n, s in by.items() if held[n] == len(effect["records"])}


def _numbers(values: list) -> bool:
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)


def _cut(sides: dict[bool, list]) -> tuple[str, float] | None:
    """By brute force over every candidate value: a threshold that splits the sides of a numeric
    marker, with the side the material records lie on, or None."""
    material, other = sides[True], sides[False]
    if not _numbers(material + other):
        return None
    for c in sorted(set(material + other)):
        if all(b < c for b in material) and all(a >= c for a in other):
            return "below", c
        if all(b > c for b in material) and all(a <= c for a in other):
            return "above", c
    return None


def _separates(sides: dict[bool, list]) -> bool:
    """By brute force: a numeric marker separates the planted records where some threshold among
    its values splits them (:func:`_cut`), as M's threshold cuts the effect; a marker of words or
    true or false where no value is held on both sides."""
    material, other = sides[True], sides[False]
    if _numbers(material + other):
        return _cut(sides) is not None
    return not any(json.dumps(a) == json.dumps(b) for a in material for b in other)


def _check_separating(out: Path, effect: dict) -> dict[str, set[str]]:
    """The sealed file's ``separating`` against the brute force, at the regime and at the last
    fit: the markers it names are those that separate the planted records' two sides, each with
    the values on each side, and with a threshold where one splits them on the side it says."""
    found = {}
    for at in ("regime", "last fit"):
        sides = _planted_sides(out, effect, at)
        named = {e["marker"]: e for e in effect["separating"] if e["at"] == at}
        assert set(named) == {n for n, s in sides.items() if _separates(s)}, at
        for name, e in named.items():
            s = sides[name]
            assert sorted(map(json.dumps, set(s[True]))) == sorted(map(json.dumps, e["material"]))
            assert sorted(map(json.dumps, set(s[False]))) == sorted(
                map(json.dumps, e["not_material"]))
            cut = _cut(s)
            if cut is None:  # words or true or false: no value on both sides
                assert e["threshold"] is None and not _numbers(s[True] + s[False])
                continue
            t = e["threshold"]
            if cut[0] == "below":
                assert all(b < t for b in s[True]) and all(a >= t for a in s[False])
                assert e["direction"] == "material below the threshold, none at or above it"
            else:
                assert all(b > t for b in s[True]) and all(a <= t for a in s[False])
                assert e["direction"] == "material above the threshold, none at or below it"
        found[at] = set(named)
    return found


def test_the_sealed_file_names_every_marker_that_separates_the_planted_records(tmp_path):
    """The verifier's V6D-6. Each planted record holds a real record's markers whole, and on this
    corpus a record's predictors per row is its predictors over its rows analyzed, so by
    construction the two move together across a dataset's records: whichever is M, the other
    splits the planted records' two sides just as M does (the other way round), and the rows, one
    value across a dataset, split nothing. The sealed file names both, at the regime and at the
    last fit, M first at its threshold t, and each as the brute force over the copy's files finds
    it."""
    src = corpus(tmp_path / "corpus", per=3)
    for seed in range(10):
        out = tmp_path / f"copy{seed}"
        effect = S.write(src, out, tmp_path / f"s{seed}.json", seed=seed, k=6)["planted"]["effect"]
        found = _check_separating(out, effect)
        assert found == {"regime": {"rows_analyzed", "predictors_per_row"},
                         "last fit": {"rows_analyzed", "predictors_per_row"}}, seed
        first = effect["separating"][0]
        assert (first["marker"], first["at"], first["threshold"], first["direction"]) == (
            effect["marker"], "regime", effect["threshold"], effect["direction"])
        other = next(e for e in effect["separating"] if e["at"] == "regime"
                     and e["marker"] != effect["marker"])
        assert other["direction"] == "material above the threshold, none at or below it"


def _without(folder: Path, name: str) -> Path:
    """Every record of ``folder`` with the marker ``name`` taken out, at the regime and each fit."""
    for path in folder.glob("*.json"):
        rec = json.loads(path.read_text())
        rec["regime"]["markers"] = [m for m in rec["regime"]["markers"] if m["name"] != name]
        for a in rec["attempts"]:
            a["markers"] = [m for m in a["markers"] or [] if m["name"] != name]
        path.write_text(json.dumps(rec), "utf-8")
    return folder


def test_where_only_m_separates_the_planted_records_the_sealed_file_names_m_alone(tmp_path):
    """The guard's silent case: with the predictors per row taken out, the rows analyzed is the
    one marker that varies across a dataset's records, so it is M, and nothing else separates the
    planted records: the rows, one value across the dataset, are never named."""
    src = _without(corpus(tmp_path / "corpus", per=3), "predictors_per_row")
    for seed in range(6):
        out = tmp_path / f"copy{seed}"
        effect = S.write(src, out, tmp_path / f"s{seed}.json", seed=seed, k=6)["planted"]["effect"]
        assert effect["marker"] == "rows_analyzed"
        assert _check_separating(out, effect) == {"regime": {"rows_analyzed"},
                                                  "last fit": {"rows_analyzed"}}
        assert [e["marker"] for e in effect["separating"]] == ["rows_analyzed", "rows_analyzed"]


def test_a_sibling_marker_is_named_only_in_the_draws_where_it_separates_the_planted_records(
        tmp_path):
    """Both ways on one corpus: each record's predictors per row set apart from its rows analyzed
    (0.06, 0.02, 0.11 for a dataset's first, second and third records, whose rows analyzed fall),
    so whether the other marker splits the planted records depends on M, t and the records drawn;
    over the seeds the sealed file names it in some draws and not in others, each time as the
    brute force finds."""
    src = corpus(tmp_path / "corpus", per=3)
    for path in src.glob("*.json"):
        rec = json.loads(path.read_text())
        r = (int(rec["run"]["id"][4:6]) - 1) % 3
        for markers in [rec["regime"]["markers"], *(a["markers"] for a in rec["attempts"])]:
            for m in markers:
                if m["name"] == "predictors_per_row":
                    m["value"] = (0.06, 0.02, 0.11)[r]
        path.write_text(json.dumps(rec), "utf-8")
    named, alone = 0, 0
    for seed in range(24):
        out = tmp_path / f"copy{seed}"
        effect = S.write(src, out, tmp_path / f"s{seed}.json", seed=seed, k=6)["planted"]["effect"]
        found = _check_separating(out, effect)["regime"]
        assert effect["marker"] in found
        if len(found) == 2:
            named += 1
        else:
            alone += 1
    assert named and alone, (named, alone)


def test_a_marker_of_words_is_named_where_the_sides_share_no_value_and_one_with_parts_never(
        tmp_path):
    """Both ways for a marker no threshold cuts: each record's outcome type a word (binary for a
    dataset's first record, continuous for the others) is named, with each side's words and no
    threshold, in the draws where the planted records' two sides hold no word in common, and not
    where they do; a design held as a value with parts (one per record) is never named, since a
    region bounds only a number, a word or true or false."""
    src = corpus(tmp_path / "corpus", per=3)
    for path in src.glob("*.json"):
        rec = json.loads(path.read_text())
        r = (int(rec["run"]["id"][4:6]) - 1) % 3
        added = [marker("outcome_type", ("binary", "continuous", "continuous")[r]),
                 marker("design", {"survey": None, "grain": f"g{r}"})]
        rec["regime"]["markers"] += added
        for a in rec["attempts"]:
            a["markers"] = [*a["markers"], *json.loads(json.dumps(added))]
        path.write_text(json.dumps(rec), "utf-8")
    words = {True: 0, False: 0}
    for seed in range(24):
        out = tmp_path / f"copy{seed}"
        effect = S.write(src, out, tmp_path / f"s{seed}.json", seed=seed, k=6)["planted"]["effect"]
        found = _check_separating(out, effect)["regime"]
        assert "design" not in found
        words["outcome_type" in found] += 1
        for e in effect["separating"]:
            if e["marker"] == "outcome_type":
                assert e["threshold"] is None and e["direction"] == S.AT_VALUES
                assert not set(e["material"]) & set(e["not_material"])
    assert words[True] and words[False], words


def test_a_template_whose_dataset_holds_no_marker_that_varies_is_refused(tmp_path):
    """The guard both ways: where every record of each dataset holds the same markers, no value of
    a marker lies on either side of a threshold among them, and the planted effect is refused; the
    same corpus with its rows analyzed varying is planted in."""
    flat = corpus(tmp_path / "flat", per=3)
    for path in flat.glob("*.json"):
        rec = json.loads(path.read_text())
        rows = rec["regime"]["n_rows"]
        rec["regime"]["n_analyzed"] = rows - 1
        rec["regime"]["markers"] = [marker("rows", rows), marker("rows_analyzed", rows - 1),
                                    marker("predictors_per_row", PREDICTORS / (rows - 1))]
        for a in rec["attempts"]:
            a["markers"] = json.loads(json.dumps(rec["regime"]["markers"]))
        path.write_text(json.dumps(rec), "utf-8")
    with pytest.raises(S.ShuffleRefused, match="varies across the records of its dataset"):
        S.write(flat, tmp_path / "copy", tmp_path / "s.json", seed=1, k=2)
    S.write(corpus(tmp_path / "varied", per=3), tmp_path / "copy2", tmp_path / "s2.json",
            seed=1, k=2)
    # One record a dataset, as each dataset of the default corpus holds: nothing varies there.
    with pytest.raises(S.ShuffleRefused, match="varies across the records of its dataset"):
        S.write(corpus(tmp_path / "one"), tmp_path / "copy3", tmp_path / "s3.json", seed=1, k=2)


def test_a_planted_record_is_its_templates_clone_and_the_sealed_file_says_so(tmp_path):
    """The planted records' decisions, and their fits but the last one's scores, movement and
    change, are their template's (the verifier's V6D-2 (c), accepted): a reader of the view who
    compares records finds them with their template, and the sealed file's ``kept`` says so."""
    from research.swarm.claim_cycle import view as W

    src = corpus(tmp_path / "corpus", per=3)
    body = S.write(src, tmp_path / "copy", tmp_path / "s.json", seed=5, k=6)
    effect = body["planted"]["effect"]
    views = {v["run"]["id"]: v for v in W.read(tmp_path / "copy")}  # by the ids they serve
    template = views[neutral_id(effect["template"])]
    for run_id in body["planted"]["records"]:
        v = views[neutral_id(run_id)]
        assert [(d["seq"], d["kind"], d["source"], d["phase"]) for d in v["decisions"]] == [
            (d["seq"], d["kind"], d["source"], d["phase"]) for d in template["decisions"]]
        assert v["attempts"][0]["movement"] == template["attempts"][0]["movement"]
    assert any("decisions" in k and "earlier fits" in k for k in effect["kept"])


def signed_times(rec):
    from research.swarm.record import STAMP

    return [m.group() for m in STAMP.finditer(json.dumps({k: v for k, v in rec.items()
                                                          if k != "dataset"}))]


def _template_deciding(kind: str) -> dict:
    """The constructed Predict record (fit2's whole change is the analyst's record #4), with
    record #4 of decision kind ``kind``."""
    rec = predict_record()
    planted_scores(rec)
    rec["decisions"][3]["kind"] = kind
    return rec


def test_the_planted_decision_is_one_a_theorist_can_propose_about():
    """The planted effect's claim must be one the theorist can propose (SWARM_PLAN §9: "the planted
    effect's claim proposed and routed"): a template's decision is one the validator takes and the
    theorist's prompt does not set aside (`set_selection`), and every other kind the engine
    records, one of the protocol's among them, is no template's."""
    from research.swarm.claim_cycle import proposal as P
    from research.swarm.claim_cycle.tests.constructed import proposal

    kinds = sorted(P.decision_kinds())
    plantable = {k for k in kinds if S._plantable(_template_deciding(k)) is not None}
    assert "select_models" in plantable and not {"revert", "open_seal", P.SELECTION} & plantable
    assert plantable == set(kinds) - set(P.PROTOCOL) - {P.SELECTION}
    for kind in plantable:
        assert P.validate(proposal(decision=kind)).decision == kind
    for kind in set(kinds) - plantable - {P.SELECTION}:
        with pytest.raises(P.ProposalRefused, match="records the protocol"):
            P.validate(proposal(decision=kind))


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
    src = corpus(tmp_path / "corpus", per=3)
    S.write(src, tmp_path / out, tmp_path / sealed, seed=1, k=2)
    assert len(list((tmp_path / out).glob("*.json"))) == 9 + 2 and (tmp_path / sealed).is_file()


def test_the_command_line_writes_the_copy_and_says_where(tmp_path, capsys):
    src = corpus(tmp_path / "corpus", per=3)
    assert S.main([str(src), "--out", str(tmp_path / "copy"), "--sealed",
                   str(tmp_path / "s.json"), "--seed", "2", "--planted", "4"]) == 0
    said = json.loads(capsys.readouterr().out)
    assert said == {"copy": str(tmp_path / "copy"), "records": 9, "planted": 4,
                    "sealed": str(tmp_path / "s.json")}
    assert S.main([str(src), "--out", str(tmp_path / "copy"), "--sealed",
                   str(tmp_path / "t.json")]) == 1  # the copy is written once

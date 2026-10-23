"""The reading restriction (SWARM_PLAN §9; ``claim_cycle/view.py`` and ``tools.py``): the theorist's and
the experimentalist's tools read a record only through the view's fields, and never the shuffled
copy's sealed file.

Each withheld field is tested both ways on a constructed record: a value planted there, distinct
from every other in the record, never reaches the view or the tools' output, and asking for the
field by name is refused with why; a value planted in a served field reaches both. The sealed file
is the one ``shuffle.write`` writes for a constructed corpus, planted records and all."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from research.swarm import record as R
from research.swarm import shuffle as S
from research.swarm.claim_cycle import proposal as P
from research.swarm.claim_cycle import tools as T
from research.swarm.claim_cycle import view as W
from research.swarm.claim_cycle.tests.constructed import proposal
from research.swarm.tests import test_shuffle as TS
from research.swarm.tests.constructed import marker, predict_record

LATER = "2031-07-09T10:00:00+00:00"


def each(rec: dict, section: str, **values: Any) -> None:
    for item in rec[section]:
        item.update(values)


def renamed_fits(rec: dict) -> dict:
    """``rec`` with its fits' ids renamed everywhere it names them."""
    text = json.dumps(rec).replace('"fit1"', '"sentinel-fit-1"').replace('"fit2"', '"sentinel-fit-2"')
    return json.loads(text)


def _tag(rec):
    for d in rec["decisions"]:
        if d["source"] == "analyst":
            d["tag"] = {"marker": "rows", "rejected": {"id": "sentinel_rejected"}}


def _examined(rec):
    rec["decisions"][3]["examined"][0]["reason"] = "sentinel preview reason"


def _option(rec):
    rec["decisions"][0]["option"]["column"] = "sentinel_column"
    rec["decisions"][0]["answers"][0]["value"] = {"column": "sentinel_column"}


def _mattered(rec):
    spec = {"key": "k", "picks": {"adjust:sentinel_col": "out"}, "primary": False,
            "estimate": 0.35, "low": 0.2, "high": 0.5, "n": 7654321, "estimator": "sentinel-est"}
    rec["decisions"][3]["materiality"] = {
        "source": "mattered", "primary": {**spec, "primary": True, "estimate": 0.3},
        "departures": [{"choice": "adjust:sentinel_col", "label": "sentinel label",
                        "option": "out", "spec": spec, "not_estimated": None}]}


def _markers_why(rec):
    rec["attempts"][0].update(markers=None, markers_why="sentinel markers why")


def _movement_words(rec):
    for a in rec["attempts"]:
        for m in a["movement"]:
            if m["movement"]:
                m["movement"].update(words="sentinel words", label="sentinel label")


def _refusal(rec):
    rec["surfacings"]["refusals"] = [{"at": None, "command": "answer", "code": "allowed_code",
                                      "message": "sentinel refusal message on 1,234 rows",
                                      "exits": ["sentinel exit"], "then": "ask", "by": "server",
                                      "stage": None}]


def _failed_fit(rec):
    rec["surfacings"]["failed_fits"] = [{"at": None, "command": "fit", "stage": "modeling",
                                         "key": "sentinel-key", "error": "sentinel error words",
                                         "kind": "refused",
                                         "refusal": {"code": "nothing_kept",
                                                     "message": "sentinel refusal words",
                                                     "exits": []}}]


def _account(rec):
    rec["account"] = {"at": LATER, "text": "sentinel account", "after_results": True,
                      "after_seal": True}


def _want(rec):
    rec["wants"] = [{"at": LATER, "kind": "check", "what": "sentinel want",
                     "reason": "sentinel want reason", "marker": "rows"}]


# A withheld field: what plants a value there, and the value's text, which no other field holds.
WITHHELD: dict[str, tuple[Callable[[dict], Any], str]] = {
    "run.analyst_model": (lambda r: r["run"].update(analyst_model="claude-sentinel-model"),
                          "sentinel-model"),
    "run.persona": (lambda r: r["run"].update(persona="skeptical"), "skeptical"),
    "run.effort": (lambda r: r["run"].update(effort="effort-sentinel"), "effort-sentinel"),
    "run.prompt_version": (lambda r: r["run"].update(prompt_version="prompt-sentinel"),
                           "prompt-sentinel"),
    "run.spend": (lambda r: r["run"].update(spend={"fresh": 987654321, "cached": 1,
                                                   "messages": 1}), "987654321"),
    "run.started": (lambda r: r["run"].update(started=LATER, ended=None), "2031"),
    "run.server": (lambda r: r["run"].update(server={"pid": 424242, "port": 31337}), "424242"),
    "run.help": (lambda r: r["run"].update(help=["sentinel help"]), "sentinel help"),
    "dataset.id": (lambda r: r["dataset"].update(id="nhanes_sentinel"), "nhanes_sentinel"),
    "dataset.control": (lambda r: r["dataset"].update(control="null_decoy"), "null_decoy"),
    "dataset.fixture": (lambda r: r["dataset"].update(fixture="sentinel_fixture.csv"),
                        "sentinel_fixture"),
    "dataset.known": (lambda r: r["dataset"].update(known={"sentinel_known": 0.5}),
                      "sentinel_known"),
    "dataset.files": (lambda r: r["dataset"]["files"][0].update(name="sentinel_table.csv"),
                      "sentinel_table"),
    "dataset.codebook": (lambda r: r["dataset"].update(codebook="sentinel-codebook"),
                         "sentinel-codebook"),
    "dataset.card": (lambda r: r["dataset"].update(card={"question": "sentinel card"}),
                     "sentinel card"),
    "dataset.declared": (lambda r: r["dataset"].update(declared=[
        {"name": "instrument", "status": "declared", "value": "sentinel-device",
         "quote": "q", "source": "s"}]), "sentinel-device"),
    "question.target": (lambda r: r["question"].update(target="sentinel_target"),
                        "sentinel_target"),
    "question.stated": (lambda r: r["question"].update(stated="sentinel stated"),
                        "sentinel stated"),
    "question.estimand": (lambda r: r["question"].update(estimand={"exposure": "sentinel_x"}),
                          "sentinel_x"),
    "regime.n_rows": (lambda r: r["regime"].update(n_rows=1234567), "1234567"),
    "regime.missing": (lambda r: r["regime"].update(missing={"sentinel_col": 3}),
                       "sentinel_col"),
    "regime.markers.basis": (lambda r: [m.update(basis="sentinel-basis") for m in
                                        r["regime"]["markers"] if m["value"] is not None],
                             "sentinel-basis"),
    "regime.markers.why": (lambda r: r["regime"]["markers"][2].update(why="sentinel marker why"),
                           "sentinel marker why"),
    "decisions.id": (lambda r: [d.update(id=f"sentinel-id-{d['seq']}") for d in r["decisions"]],
                     "sentinel-id"),
    "decisions.at": (lambda r: each(r, "decisions", at=LATER), "2031"),
    "decisions.option": (_option, "sentinel_column"),
    "decisions.answers": (_option, "sentinel_column"),
    "decisions.reason": (lambda r: each(r, "decisions", reason="sentinel reason"),
                         "sentinel reason"),
    "decisions.tag": (_tag, "sentinel_rejected"),
    "decisions.examined": (_examined, "sentinel preview reason"),
    "decisions.materiality.departures.choice": (_mattered, "sentinel_col"),
    "decisions.materiality.departures.label": (_mattered, "sentinel label"),
    "decisions.materiality.departures.spec.n": (_mattered, "7654321"),
    "decisions.materiality.primary.picks": (_mattered, "sentinel_col"),
    "attempts.id": (renamed_fits, "sentinel-fit"),
    "attempts.at": (lambda r: each(r, "attempts", at=LATER), "2031"),
    "attempts.fit_key": (lambda r: each(r, "attempts", fit_key="sentinel-fitkey"),
                         "sentinel-fitkey"),
    "attempts.plan_digest": (lambda r: each(r, "attempts", plan_digest="sentinel-digest"),
                             "sentinel-digest"),
    "attempts.reasons": (lambda r: [a.update(reasons=[{"seq": x["seq"], "note": "sentinel note"}
                                                      for x in a["reasons"]])
                                    for a in r["attempts"]], "sentinel note"),
    "attempts.scores": (lambda r: r["attempts"][0]["scores"]["linear"]["auc"].update(
        estimate=0.4242424242), "0.4242424242"),
    "attempts.at_opening": (lambda r: r["attempts"][1]["at_opening"].update(n_holdout=8675309),
                            "8675309"),
    "attempts.folds": (lambda r: each(r, "attempts", folds="sentinel folds"), "sentinel folds"),
    "attempts.held_out": (lambda r: each(r, "attempts", held_out="sentinel held"),
                          "sentinel held"),
    "attempts.seed": (lambda r: each(r, "attempts", seed=91919191), "91919191"),
    "attempts.markers_why": (_markers_why, "sentinel markers why"),
    "attempts.movement_why": (lambda r: r["attempts"][0].update(movement_why="sentinel moved"),
                              "sentinel moved"),
    "attempts.change_why": (lambda r: each(r, "attempts", change_why="sentinel change why"),
                            "sentinel change why"),
    "attempts.movement.words": (_movement_words, "sentinel words"),
    "result": (lambda r: r["result"].update(n_holdout=8675309), "8675309"),
    "claims": (lambda r: r["claims"][0].update(id="sentinel.claim"), "sentinel.claim"),
    "surfacings.refusals.message": (_refusal, "sentinel refusal message"),
    "surfacings.refusals.exits": (_refusal, "sentinel exit"),
    "surfacings.failed_fits.error": (_failed_fit, "sentinel error words"),
    "surfacings.failed_fits.refusal.message": (_failed_fit, "sentinel refusal words"),
    "surfacings.failed_fits.key": (_failed_fit, "sentinel-key"),
    "surfacings.concerns": (lambda r: r["surfacings"].update(concerns=[{"w": "sentinel concern"}]),
                            "sentinel concern"),
    "export": (lambda r: r["export"].update(methods="sentinel methods"), "sentinel methods"),
    "wants": (_want, "sentinel want"),
    "account": (_account, "sentinel account"),
    "replay": (lambda r: r.update(replay={"scorer": "sentinel-scorer", "generator": "g",
                                          "draws": 1, "why": "w"}), "sentinel-scorer"),
    "replication": (lambda r: r.update(replication={"paper": "sentinel paper", "stated": [],
                                                    "unstated": []}), "sentinel paper"),
}


def planted(fn: Callable[[dict], Any]) -> dict:
    rec = copy.deepcopy(predict_record())  # its fits share the constructed scores' dicts
    rec["run"]["id"] = "run_01-0123"
    got = fn(rec)
    rec = got if isinstance(got, dict) and "version" in got else rec
    R.validate(rec)
    return rec


def written(folder: Path, *recs: dict) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for rec in recs:
        (folder / f"{rec['run']['id']}.json").write_text(json.dumps(rec), "utf-8")
    return folder


def printed(argv: list[str]) -> tuple[int, str]:
    out: list[str] = []
    code = T.main(argv, out=out.append)
    return code, "\n".join(out)


@pytest.mark.parametrize("field", list(WITHHELD))
def test_a_value_in_a_withheld_field_never_reaches_the_view_or_the_tools(field, tmp_path):
    fn, text = WITHHELD[field]
    rec = planted(fn)
    assert text in json.dumps(rec)  # the construction: the record holds it
    assert text not in json.dumps(W.views([rec]))
    folder = written(tmp_path / "corpus", rec)
    for argv in (["records", str(folder)], ["index", str(folder)]):
        code, said = printed(argv)
        assert code == 0 and text not in said, argv


@pytest.mark.parametrize("field", list(W.WITHHELD))
def test_asking_for_a_withheld_field_is_refused_with_why(field, tmp_path):
    with pytest.raises(W.ReadRefused) as exc:
        W.check_fields([field])
    assert W.WITHHELD[field] in str(exc.value)
    folder = written(tmp_path / "corpus", planted(lambda r: None))
    code, said = printed(["records", str(folder), "--fields", field])
    assert code == 1 and W.WITHHELD[field] in json.loads(said)["refused"]


def test_asking_for_a_field_the_view_does_not_know_is_refused():
    with pytest.raises(W.ReadRefused, match="not a field of the view"):
        W.check_fields(["regime.markers.sentinel"])


# A served field: what plants a value there, and the value's text, which the view must carry.
SERVED: dict[str, tuple[Callable[[dict], Any], str]] = {
    "run.id": (lambda r: r["run"].update(id="run_07-sentinel"), "run_07-sentinel"),
    "question.purpose": (lambda r: None, '"prediction"'),
    "regime.markers": (lambda r: r["regime"]["markers"][0].update(value=4815162342),
                       "4815162342"),
    "decisions.kind": (lambda r: None, '"select_models"'),
    "decisions.materiality": (lambda r: r["decisions"][3]["materiality"]["movement"][0][
        "movement"].update(value=0.987654), "0.987654"),
    "attempts.families": (lambda r: None, '"elastic_net"'),
    "attempts.primary_metric": (lambda r: None, '"auc"'),
    "attempts.markers": (lambda r: r["attempts"][0]["markers"][0].update(value=3141592),
                         "3141592"),
    "attempts.change": (lambda r: r["attempts"][1].update(change=[{
        "family": "linear", "metric": "auc", "against": "fit1", "paired": True,
        "difference": 0.0123456, "better": True, "se": 0.004, "df": 4.0, "ci_low": 0.001,
        "ci_high": 0.02, "excludes_zero": True, "folds": [0, 1, 2, 3, 4], "half_width": 0.05,
        "test_share": [0.25, 0.25], "why": None}], change_why=None), "0.0123456"),
    "surfacings.refusals": (_refusal, '"allowed_code"'),
    "surfacings.failed_fits": (_failed_fit, '"nothing_kept"'),
    "decisions.materiality.departures": (_mattered, "0.35"),
}


@pytest.mark.parametrize("field", list(SERVED))
def test_a_value_in_a_served_field_reaches_the_view_and_the_tools(field, tmp_path):
    fn, text = SERVED[field]
    rec = planted(fn)
    assert text in json.dumps(W.views([rec]))
    folder = written(tmp_path / "corpus", rec)
    code, said = printed(["records", str(folder)])
    assert code == 0 and text in said
    if field in W.FIELDS:
        code, said = printed(["records", str(folder), "--fields", field])
        assert code == 0 and text in said


def leaves(obj: Any, prefix: str = "") -> set[str]:
    """The view's field paths, to the depth ``FIELDS`` names them: a list's items read as one."""
    if prefix in W.FIELDS:
        return {prefix}
    if isinstance(obj, list):
        return set().union(*(leaves(v, prefix) for v in obj)) if obj else set()
    if isinstance(obj, dict):
        return set().union(*(leaves(v, f"{prefix}.{k}" if prefix else k) for k, v in obj.items()))
    return {prefix}


def test_the_view_serves_exactly_its_fields():
    rec = planted(_failed_fit)
    _refusal(rec)
    SERVED["attempts.change"][0](rec)
    assert leaves(W.views([rec])[0]) == set(W.FIELDS)
    assert not set(W.FIELDS) & set(W.WITHHELD)


def _models() -> dict[str, type]:
    return {"run": R.Run, "dataset": R.Dataset, "question": R.Question, "regime": R.Regime,
            "regime.markers": R.Marker, "decisions": R.Decision, "attempts": R.Attempt,
            "surfacings": R.Surfacings, "surfacings.refusals": R.Refusal,
            "surfacings.failed_fits": R.FailedFit}


def test_every_field_of_the_record_is_served_or_withheld_with_why():
    """A field the record gains later is withheld until someone serves it on purpose: this test
    names it until it is classed."""
    def classed(path: str) -> bool:
        parts = path.split(".")
        prefixes = {".".join(parts[:i]) for i in range(1, len(parts) + 1)}
        served = any(f == path or f.startswith(path + ".") or path.startswith(f + ".")
                     for f in W.FIELDS)
        return served or bool(prefixes & set(W.WITHHELD))

    paths = [k for k in R.Record.model_fields]
    for where, model in _models().items():
        paths += [f"{where}.{k}" for k in model.model_fields]
    stray = [p for p in paths if not classed(p) and p not in _models()]
    assert stray == []


def test_datasets_are_labeled_in_order_of_first_appearance_and_never_named(tmp_path):
    recs = []
    for i, dataset in enumerate(["zeta_study", "alpha_study", "zeta_study"], start=1):
        rec = predict_record()
        rec["run"]["id"] = f"run_0{i}-ab"
        rec["dataset"]["id"] = dataset
        recs.append(rec)
    got = W.views(list(reversed(recs)))
    assert [(v["run"]["id"], v["dataset"]["id"]) for v in got] == [
        ("run_01-ab", "dataset_1"), ("run_02-ab", "dataset_2"), ("run_03-ab", "dataset_1")]
    assert "_study" not in json.dumps(got)


def test_a_fits_reference_reads_as_its_seq():
    v = W.views([predict_record()])[0]
    moved = v["decisions"][3]["materiality"]
    assert moved["attempt"] == 2 and moved["movement"][0]["against"] == 1
    assert v["attempts"][1]["movement"][0]["against"] == 1


def test_the_tools_filter_by_run_and_dataset(tmp_path):
    a, b = predict_record(), predict_record()
    a["run"]["id"], b["run"]["id"] = "run_01-aa", "run_02-bb"
    b["dataset"]["id"] = "other"
    folder = written(tmp_path / "corpus", a, b)
    code, said = printed(["records", str(folder), "--run", "run_02-bb"])
    assert code == 0 and [r["run"]["id"] for r in json.loads(said)["records"]] == ["run_02-bb"]
    code, said = printed(["records", str(folder), "--dataset", "dataset_1"])
    assert [r["run"]["id"] for r in json.loads(said)["records"]] == ["run_01-aa"]
    code, said = printed(["index", str(folder)])
    assert json.loads(said)["records"][1] == {"run": "run_02-bb", "dataset": "dataset_2",
                                              "purpose": "prediction", "decisions": 5,
                                              "attempts": 2}


# ── the sealed file ──────────────────────────────────────────────────────────


@pytest.fixture
def shuffled(tmp_path) -> tuple[Path, Path, dict]:
    """A shuffled copy of a constructed corpus with two planted records, and its sealed file."""
    src = TS.corpus(tmp_path / "corpus")
    sealed = tmp_path / "sealed" / "s.json"
    body = S.write(src, tmp_path / "copy", sealed, seed=5, k=2)
    return tmp_path / "copy", sealed, body


def test_the_shuffled_copy_reads_whole_and_its_planted_records_read_as_any_other(shuffled):
    copy, _, body = shuffled
    got = W.read(copy)
    assert len(got) == body["records"] + 2
    assert set(body["planted"]["records"]) <= {v["run"]["id"] for v in got}


def test_the_sealed_file_is_never_read(shuffled, tmp_path):
    copy, sealed, body = shuffled
    planted_ids = body["planted"]["records"]
    secrets = [*planted_ids, json.dumps(body["derangement"])]
    beside = tmp_path / "beside"
    beside.mkdir()
    (beside / "s.json").write_bytes(sealed.read_bytes())
    inside = written(tmp_path / "with-sealed", *[json.loads(p.read_text()) for p in
                                                 sorted(copy.glob("*.json"))][:1])
    (inside / "sealed.json").write_bytes(sealed.read_bytes())
    for where in (sealed, beside, inside):
        with pytest.raises(W.ReadRefused, match="sealed file") as exc:
            W.read(where)
        assert not any(s in str(exc.value) for s in secrets)
        asks = [["records", str(where)], ["index", str(where)]]
        for argv in asks + ([["proposals", str(where)]] if where == sealed else []):
            code, said = printed(argv)
            assert code == 1 and "sealed file" in json.loads(said)["refused"], argv
            assert not any(s in said for s in secrets)
    before = sealed.read_bytes()
    code, said = printed(["propose", "--out", str(sealed), json.dumps(proposal())])
    assert code == 1 and "sealed file" in json.loads(said)["refused"]
    assert sealed.read_bytes() == before


def test_a_folder_holding_a_file_that_is_not_a_record_is_refused_without_echoing_it(tmp_path):
    folder = written(tmp_path / "corpus", predict_record())
    (folder / "notes.json").write_text(json.dumps({"secret": "s3cr3t"}), "utf-8")
    with pytest.raises(W.ReadRefused, match="not a record") as exc:
        W.read(folder)
    assert "s3cr3t" not in str(exc.value)


def test_a_failed_run_is_not_a_record_and_is_not_read(tmp_path):
    folder = written(tmp_path / "corpus", predict_record())
    (folder / "failed").mkdir()
    (folder / "failed" / "run_99.json").write_text('{"secret": "s3cr3t"}', "utf-8")
    assert len(W.read(folder)) == 1


def test_an_empty_folder_and_a_missing_one_are_refused(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(W.ReadRefused, match="no record"):
        W.read(tmp_path / "empty")
    with pytest.raises(W.ReadRefused, match="not a folder"):
        W.read(tmp_path / "missing")


# ── proposing and reading proposals through the tools ────────────────────────


def test_propose_validates_and_appends_and_proposals_reads_them_back(tmp_path):
    out = tmp_path / "proposals.jsonl"
    code, said = printed(["propose", "--out", str(out), json.dumps(proposal())])
    assert code == 0 and json.loads(said)["proposed"]["status"] == "proposed"
    code, said = printed(["propose", "--out", str(out), json.dumps(proposal(id="p2", region=[]))])
    assert code == 1 and "not a surfacing claim" in json.loads(said)["refused"]
    code, said = printed(["proposals", str(out)])
    assert code == 0 and [p["id"] for p in json.loads(said)["proposals"]] == ["p1"]
    assert [p.id for p in P.read_proposals(out)] == ["p1"]


def test_propose_refuses_what_is_not_json(tmp_path):
    code, said = printed(["propose", "--out", str(tmp_path / "p.jsonl"), "{not json"])
    assert code == 1 and "JSON" in json.loads(said)["refused"]

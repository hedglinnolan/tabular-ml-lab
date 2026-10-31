"""The assembler's version 2 (SWARM_PLAN §4; ``assemble.py``), on constructed run directories: each
field from the driver's own lines (a preview's reason, an answer's tag, a want, the account, each
failed fit), the live markers at each fit (the driver's snapshots by the fit's key, the answers
the fit read and, under Predict, the primary metric its own record names), each family's signed
change from the fit before (``signed.py``), and the engine the record names (the export's, else
the launch's stamp, else none). Every number is constructed; each guard both ways."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from research.swarm import assemble as A
from research.swarm import markers as M
from research.swarm import vocabulary as V
from research.swarm.record import FailedFit, Marker, SignedChange, validate
from research.swarm.tests.test_swarm_assemble import META, FakeClient, _run_dir, _served_routes

T = "2026-10-22T10:00:"


def t(sec: float) -> str:
    whole = int(sec)
    return f"{T}{whole:02d}.{int(round((sec - whole) * 1e6)):06d}+00:00"


# ── a preview's reason and an answer's tag ───────────────────────────────────


def preview_command(start: float, end: float, kind: str, reason: str) -> dict[str, Any]:
    return {"command": "preview", "began": t(start), "returned": t(end), "at": t(start)[:19] + "+00:00",
            "done": t(end)[:19] + "+00:00", "args": {"reason": reason},
            "summary": {"exit": 0, "previewed": {"kind": kind}}}


def test_a_preview_takes_the_reason_of_the_command_that_made_it():
    commands = [preview_command(1.0, 2.0, "set_missing", "does a fill keep rows?"),
                preview_command(5.0, 6.0, "set_split", "a fifth held out?")]
    previews = [{"at": t(1.5), "decision": {"kind": "set_missing"}, "refused": None},
                {"at": t(5.5), "decision": {"kind": "set_split"}, "refused": None},
                {"at": t(3.0), "decision": {"kind": "set_missing"}, "refused": None},  # no command
                {"at": t(5.7), "decision": {"kind": "set_missing"}, "refused": None}]  # other kind
    got = [p["reason"] for p in A.preview_reasons(previews, commands)]
    assert got == ["does a fill keep rows?", "a fifth held out?", None, None]


def test_a_preview_two_commands_could_have_made_is_given_no_reason():
    commands = [preview_command(1.0, 3.0, "set_missing", "one"),
                preview_command(2.0, 4.0, "set_missing", "two")]
    got = A.preview_reasons([{"at": t(2.5), "decision": {"kind": "set_missing"}}], commands)
    assert got[0]["reason"] is None


def test_a_command_kept_to_the_second_is_read_to_the_end_of_its_second():
    command = {"command": "preview", "at": f"{T}01+00:00", "done": f"{T}01+00:00",
               "args": {"reason": "r"}, "summary": {"previewed": {"kind": "set_missing"}}}
    got = A.preview_reasons([{"at": t(1.9), "decision": {"kind": "set_missing"}}], [command])
    assert got[0]["reason"] == "r"


def answer_command(record_id: str, marker: str, exit_: int = 0, rejected: Any = None) -> dict[str, Any]:
    return {"command": "answer", "at": f"{T}09+00:00",
            "summary": {"exit": exit_, "recorded": {"id": record_id},
                        "tag": {"marker": marker, "rejected": rejected}}}


def test_an_answers_tag_is_kept_beside_the_record_it_posted_and_only_there():
    rejected = {"id": "complete_case", "decision": {"kind": "set_missing", "strategy": "complete_case"}}
    tags = A.tags_of([answer_command("r1", "missingness", rejected=rejected),
                      answer_command("r2", "rows", exit_=1),
                      {"command": "revisit", "summary": {"exit": 0, "recorded": {"id": "r3"}}}])
    assert tags == {"r1": {"marker": "missingness", "rejected": rejected, "command": "answer",
                           "at": f"{T}09+00:00"}}


# ── the live markers at a fit ────────────────────────────────────────────────


def record(seq: int, sec: float, decision: dict[str, Any]) -> dict[str, Any]:
    return {"id": f"r{seq}", "seq": seq, "at": t(sec), "decision": decision}


LOG = [record(1, 1.0, {"kind": "set_purpose", "purpose": "prediction"}),
       record(2, 3.0, {"kind": "set_intended_use", "use": "decision_support"})]


def mk(name: str, value: Any = None, why: str | None = None) -> dict[str, Any]:
    from research.swarm.markers import marker

    return marker(name, value, "constructed" if value is not None else None, why)


def test_a_fits_markers_are_the_drivers_at_its_key_and_its_answers_are_those_it_read():
    """The fit read record #1 only (its intended use came after it): a snapshot at its key taken
    after the intended use was answered says decision support, but the fit's own answers declare
    none. The rows come from the first snapshot holding them, the rows analyzed from the second."""
    fit = {"id": "f1", "fit_key": "K", "at": t(2.0), "decisions": [1]}
    snapshots = [{"fit_key": "K", "markers": [mk("rows", 480), mk("rows_analyzed", why="computing"),
                                              mk("intended_use", {"use": "decision_support"})]},
                 {"fit_key": "K", "markers": [mk("rows", 999), mk("rows_analyzed", 384)]},
                 {"fit_key": "other", "markers": [mk("rows", 1)]}]
    got, why = A.fit_markers(fit, snapshots, A.Prefixes(LOG))
    by = {m["name"]: m for m in got}
    assert why is None and [m["name"] for m in got] == [n for n in V.LIVE if n in by]
    assert by["rows"]["value"] == 480 and by["rows_analyzed"]["value"] == 384
    assert by["intended_use"]["value"] is None and by["intended_use"]["why"]
    assert by["purpose"]["value"] == "prediction"


def test_a_fit_no_snapshot_was_taken_at_has_no_markers_and_says_so():
    fit = {"id": "f1", "fit_key": "K", "at": t(2.0), "decisions": [1]}
    got, why = A.fit_markers(fit, [{"fit_key": "other", "markers": []}], A.Prefixes(LOG))
    assert got is None and why == A.NO_SNAPSHOT


def test_a_fit_whose_answers_are_not_known_says_so_of_the_markers_read_from_them():
    fit = {"id": "f1", "fit_key": "K", "at": t(2.0), "decisions": [7]}  # no prefix holds #7
    got, _ = A.fit_markers(fit, [{"fit_key": "K", "markers": [mk("rows", 480)]}], A.Prefixes(LOG))
    by = {m["name"]: m for m in got}
    assert by["rows"]["value"] == 480
    assert by["purpose"]["value"] is None and "not known" in by["purpose"]["why"]


# ── a fit's primary score ────────────────────────────────────────────────────


def tasks(purpose: str = "prediction") -> list[dict[str, Any]]:
    """The outcome answered yes/no (#3), then numeric (#4): its primary metric changes."""
    return [record(1, 1.0, {"kind": "set_purpose", "purpose": purpose}),
            record(2, 1.5, {"kind": "set_target", "column": "y"}),
            record(3, 2.0, {"kind": "set_task", "column": "y", "task": "binary"}),
            record(4, 5.0, {"kind": "set_task", "column": "y", "task": "regression"})]


def fit_record(seq: int, key: str, metric: str, decisions: list[int], sec: float,
               purpose: str = "prediction") -> dict[str, Any]:
    return {"id": f"f{seq}", "seq": seq, "at": t(sec), "purpose": purpose, "target": "y",
            "fit_key": key, "decisions": decisions, "families": ["linear"],
            "primary_metric": metric, "folds": "D", "scores": {"linear": {}}, "effects": {},
            "movement": [], "movement_why": None}


# K1 read the yes/no answer and names log loss; K2, fitted after the numeric one, mean squared error.
K1 = fit_record(1, "K1", "log_loss", [1, 2, 3], 3.0)
K2 = fit_record(2, "K2", "mse", [1, 2, 4], 9.0)


def snapshot(key: str, served: list[dict[str, Any]], purpose: str = "prediction"
             ) -> dict[str, Any]:
    """The driver's snapshot (``driver.keep_markers``) while ``key`` is the current fit's and
    ``GET …/fits`` serves the records ``served``: every live marker as ``markers.live`` reads it."""
    view = {"state": {"purpose": purpose, "target": "y"}, "stages": {"fit": {"key": key}}}
    return {"fit_key": M.fit_key(view),
            "markers": M.live(view, lambda path: {"records": served} if path == "fits" else None)}


def primary_score(markers: list[dict[str, Any]]) -> dict[str, Any]:
    return next(m for m in markers if m["name"] == "primary_score")


def test_a_fits_primary_score_is_its_own_records_though_kept_while_it_ran():
    """K2's only snapshot was kept while K2 ran, GET …/fits serving K1's record alone, so it names
    K1's log loss; K2's attempt names its own record's metric, and every other marker is as the
    snapshot holds it (or, read from the answers alone, as the answers K2 read give it)."""
    snaps = [snapshot("K1", [K1]), snapshot("K2", [K1])]
    assert primary_score(snaps[1]["markers"])["value"] == {"metric": "log_loss"}
    first, second = A.attempts_of([K1, K2], tasks(), snaps)
    assert primary_score(first["markers"])["value"] == {"metric": "log_loss"}
    got = primary_score(second["markers"])
    assert got["value"] == {"metric": "mse"} and got["why"] is None
    assert got["basis"] == A.OWN_METRIC
    Marker.model_validate(got)
    assert [m["name"] for m in second["markers"]] == list(V.LIVE)
    kept = {m["name"]: m for m in snaps[1]["markers"]
            if m["name"] not in (*M.STATE_ONLY, "primary_score")}
    assert {m["name"]: m for m in second["markers"] if m["name"] in kept} == kept
    by = {m["name"]: m for m in second["markers"]}
    assert by["purpose"]["value"] == "prediction"


def test_a_fits_primary_score_kept_after_its_record_was_served_stays_its_metric():
    snaps = [snapshot("K1", [K1]), snapshot("K2", [K1, K2])]
    assert primary_score(snaps[1]["markers"])["value"] == {"metric": "mse"}
    _, second = A.attempts_of([K1, K2], tasks(), snaps)
    assert primary_score(second["markers"])["value"] == {"metric": "mse"}


@pytest.mark.parametrize("later", [False, True], ids=["before-only", "and-after"])
def test_a_first_fit_snapshotted_before_any_record_ends_with_its_own_records_metric(later):
    """A snapshot kept while the run's first fit ran holds no primary score (no record named one);
    with or without a later snapshot after its record was served, the fit names its record's."""
    snaps = [snapshot("K1", []), *([snapshot("K1", [K1])] if later else [])]
    assert primary_score(snaps[0]["markers"])["value"] is None
    (got,) = A.attempts_of([K1], tasks(), snaps)
    assert primary_score(got["markers"])["value"] == {"metric": "log_loss"}
    assert primary_score(got["markers"])["why"] is None


def test_an_estimate_fits_primary_score_holds_none_though_its_record_names_a_metric():
    """Under Estimate the fit record names the fit's primary metric too (the engine's fit artifact
    always does), but the marker holds none, as the snapshot read it: the result is the estimate."""
    est = {**K2, "purpose": "inference"}
    snaps = [snapshot("K2", [est], purpose="inference")]
    held = primary_score(snaps[0]["markers"])
    assert held["value"] is None and "Estimate" in held["why"]
    (got,) = A.attempts_of([est], tasks("inference"), snaps)
    assert primary_score(got["markers"]) == held


def test_a_predict_fit_no_snapshot_was_taken_at_still_has_no_markers():
    (got,) = A.attempts_of([K1], tasks(), [snapshot("K9", [K1])])
    assert got["markers"] is None and got["markers_why"] == A.NO_SNAPSHOT


# ── the signed change ────────────────────────────────────────────────────────


def fit(fid: str, folds: str, auc: list[float] | None, purpose: str = "prediction",
        metric: str = "auc", family: str = "linear") -> dict[str, Any]:
    scores = {family: {metric: {"folds": auc, "repeats": 1, "estimator": "fold_mean",
                                "ci_low": 0.6, "ci_high": 0.8}}} if auc is not None else {}
    return {"id": fid, "purpose": purpose, "primary_metric": metric if purpose == "prediction"
            else None, "folds": folds, "scores": scores}


# 100 training rows in five random folds of 20: n_test/n_train is 20/80 = 1/4 in each.
FIVE = {"n_train": 100, "folds": 5, "repeats": 1, "fold_scheme": "random", "validation": "kfold",
        "grouped_by": None}


def test_a_fit_on_the_same_folds_is_changed_paired_and_on_others_unpaired():
    before = fit("f1", "D", [0.70, 0.70, 0.70, 0.70, 0.70])
    after = fit("f2", "D", [0.71, 0.72, 0.73, 0.74, 0.75])
    (paired,), why = A.signed_change(after, before, {"D": FIVE, "E": FIVE})
    se = math.sqrt((1 / 5 + 1 / 4) * 2.5e-4)  # differences .01 to .05: s² = 2.5e-4
    assert why is None and paired["paired"] is True and paired["against"] == "f1"
    assert paired["difference"] == pytest.approx(0.03) and paired["se"] == pytest.approx(se)
    assert paired["better"] is True and paired["half_width"] == pytest.approx(0.1)
    assert paired["test_share"] == pytest.approx([0.25, 0.25])
    (unpaired,), _ = A.signed_change({**after, "folds": "E"}, before, {"D": FIVE, "E": FIVE})
    assert unpaired["paired"] is False and unpaired["difference"] == pytest.approx(0.03)


def test_a_fit_whose_folds_no_snapshot_read_has_no_change_and_says_why():
    before = fit("f1", "D", [0.70] * 5)
    after = fit("f2", "D", [0.71, 0.72, 0.73, 0.74, 0.75])
    (got,), why = A.signed_change(after, before, {})
    assert why is None and got["difference"] is None and "not read" in got["why"]
    SignedChange.model_validate(got)


def test_each_digest_takes_the_split_read_at_a_fit_scored_on_it():
    fits = [{"fit_key": "k1", "folds": "D"}, {"fit_key": "k2", "folds": "D"},
            {"fit_key": "k3", "folds": "E"}, {"fit_key": "k4", "folds": "G"}]
    other = {**FIVE, "n_train": 103}
    snaps = [{"fit_key": "k1", "split": FIVE}, {"fit_key": "k3", "split": FIVE},
             {"fit_key": "k3", "split": other},  # E read two ways: not known
             {"fit_key": "k4", "split": None}, {"fit_key": "k9", "split": FIVE}]
    assert A.splits_by_folds(fits, snaps) == {"D": FIVE, "E": None}


@pytest.mark.parametrize("after,before,why", [
    (fit("f2", "D", [0.7] * 5), None, A.NO_FIT_BEFORE),
    (fit("f2", "D", None, purpose="inference"), fit("f1", "D", None, purpose="inference"),
     A.NOT_PREDICT),
    (fit("f2", "D", [0.7] * 5, metric="log_loss"), fit("f1", "D", [0.7] * 5), "different"),
    (fit("f2", "D", [0.7] * 5, family="elastic_net"), fit("f1", "D", [0.7] * 5), "both fits"),
])
def test_a_fit_with_nothing_to_change_from_says_why(after, before, why):
    got, said = A.signed_change(after, before)
    assert got == [] and why in said


# ── the failed fits, the wants and the account ───────────────────────────────

REFUSAL = {"stage": "fit", "code": "nothing_kept", "message": "kept no predictor",
           "exits": [{"id": "x-1", "label": "No selection",
                      "decision": {"kind": "set_selection", "method": "none"}}]}


def test_each_failed_fit_is_classed_by_its_status_refusal_once():
    commands = [
        {"command": "fit", "at": f"{T}10+00:00", "summary": {"exit": 1, "fit": {
            "status": "error", "key": "k1", "error": "ValueError: kept no predictor",
            "refusal": REFUSAL}}},
        {"command": "ask", "at": f"{T}11+00:00", "summary": {"exit": 0, "fit_failed": [
            {"stage": "fit", "key": "k1", "error": "ValueError: kept no predictor",
             "kind": "refused", "refusal": REFUSAL}]}},  # the same failure: once
        {"command": "fit", "at": f"{T}12+00:00", "summary": {"exit": 0, "fit": {
            "status": "error", "key": "k2", "error": "ValueError: at least one array"}}},
        {"command": "fit", "at": f"{T}13+00:00", "summary": {"exit": 0, "fit": {
            "status": "fresh", "key": "k3"}}},
    ]
    got = A.failed_fits_of(commands)
    assert [(f["kind"], f["key"]) for f in got] == [("refused", "k1"), ("failed", "k2")]
    assert got[0]["refusal"] == {"code": "nothing_kept", "message": "kept no predictor",
                                 "exits": [{"label": "No selection", "decision": {
                                     "kind": "set_selection", "method": "none"}}]}
    for f in got:
        FailedFit.model_validate(f)


def test_a_refusal_says_who_refused_and_at_which_stage():
    commands = [{"command": "fit", "at": f"{T}10+00:00", "summary": {"exit": 1, "refused": {
        "by": "engine", "stage": "design", "code": "no_rows", "message": "no rows", "exits": []}}}]
    (refusal,) = A.surfacings_of(commands, A.Read(200, {"rows": []}))["refusals"]
    assert (refusal["by"], refusal["stage"], refusal["code"]) == ("engine", "design", "no_rows")


def test_the_wants_and_the_account_are_the_drivers_own_lines():
    wanted = {"at": t(20.0), "kind": "check", "what": "a drift check", "reason": "QC drifts",
              "marker": "run_order"}
    account = {"at": t(30.0), "text": "I kept the linear model.", "after_results": True,
               "after_seal": False}
    commands = [{"command": "want", "summary": {"exit": 0, "wanted": wanted}},
                {"command": "want", "summary": {"exit": 1, "refused": {"code": "x"}}},
                {"command": "account", "summary": {"exit": 0, "account": account}}]
    assert A.wants_of(commands) == [wanted]
    assert A.account_of(commands) == {**account, "source": "driver"}
    assert A.account_of(commands[:2]) is None


# ── a whole record ───────────────────────────────────────────────────────────


def test_a_run_with_markers_wants_and_an_account_assembles_to_a_valid_version_2_record(tmp_path):
    run = _run_dir(tmp_path / "runs" / "r1", started="2026-10-22T10:00:00+00:00",
                   vocabulary=V.DIGEST, persona="skeptical")
    wanted = {"at": t(20.0), "kind": "option", "what": "drift correction", "reason": "QC drifts",
              "marker": "run_order"}
    account = {"at": t(30.0), "text": "Nothing was fitted.", "after_results": False,
               "after_seal": False}
    lines = [{"command": "want", "at": f"{T}20+00:00", "summary": {"exit": 0, "wanted": wanted}},
             {"command": "account", "at": f"{T}30+00:00", "summary": {"exit": 0,
                                                                      "account": account}}]
    (run / A.COMMANDS_FILE).write_text("".join(json.dumps(x) + "\n" for x in lines))
    (run / A.MARKERS_FILE).write_text(json.dumps({"fit_key": "K", "markers": []}) + "\n")
    path, ok = A.assemble(run, meta={**META, "persona": "skeptical"},
                          client=FakeClient(_served_routes()), corpus=tmp_path / "corpus",
                          mattered_wait=0)
    assert ok, json.loads(path.read_text())["error"]
    rec = validate(json.loads(path.read_text("utf-8")))
    assert rec.version == 2 and rec.run.persona == "skeptical" and rec.run.vocabulary == V.DIGEST
    assert [m.name for m in rec.regime.markers] == [*V.LIVE, *V.POST_HOC]
    assert rec.wants[0].what == "drift correction" and rec.account.text == "Nothing was fitted."
    assert "replay" not in json.loads(path.read_text("utf-8"))  # the replays are score_pilot's
    assert rec.surfacings.failed_fits == []
    # The routes the markers read: the stages that are no estimate's, the readings, the fits.
    assert A.MARKER_STAGES == ("ingest", "target_info", "cohort", "structure", "seal_plan",
                               "shelf", "findings")


def test_the_run_dir_holds_the_drivers_markers(tmp_path):
    run = _run_dir(tmp_path / "r1")
    assert A.read_run_dir(run).markers == []
    (run / A.MARKERS_FILE).write_text(json.dumps({"fit_key": "K", "markers": []}) + "\n\n")
    assert A.read_run_dir(run).markers == [{"fit_key": "K", "markers": []}]
    assert Path(run / A.MARKERS_FILE).exists()


# ── the engine a run started on ──────────────────────────────────────────────

LAUNCHED = {"commit": "c0", "engine_tree": "t0", "engine_modified": False}  # launcher.engine_now's


def _bundle(engine: dict[str, Any]) -> bytes:
    """An export bundle as ``GET …/export`` serves it, its provenance naming ``engine`` (the
    export's own keys: ``commit``, ``modified``)."""
    import io
    import zipfile

    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("bundle/provenance.json", json.dumps({"engine": engine, "inputs": []}))
    return out.getvalue()


def _engine(tmp_path: Path, meta: dict[str, Any], export: bytes | None = None,
            **driver: Any) -> tuple[str | None, bool | None]:
    """The record's ``run.engine_commit`` and ``engine_modified`` for a run given ``meta``, its
    export served as ``export`` (refused where None) and driver.json holding ``driver``."""
    run = _run_dir(tmp_path / "runs" / "r1", started="2026-10-29T22:18:59+00:00", **driver)
    routes = _served_routes() | ({"/export": export} if export is not None else {})
    path, ok = A.assemble(run, meta=meta, client=FakeClient(routes), corpus=tmp_path / "corpus",
                          mattered_wait=0)
    assert ok, json.loads(path.read_text())["error"]
    rec = validate(json.loads(path.read_text("utf-8")))
    return rec.run.engine_commit, rec.run.engine_modified


def test_a_run_that_never_exported_names_the_engine_its_launch_stamped(tmp_path):
    assert _engine(tmp_path, {**META, "engine": LAUNCHED}) == ("c0", False)
    stamped_modified = {**LAUNCHED, "engine_modified": True}
    assert _engine(tmp_path / "m", {**META, "engine": stamped_modified}) == ("c0", True)


@pytest.mark.parametrize("launched", [None, LAUNCHED, {**LAUNCHED, "commit": "c9"}],
                         ids=["no_stamp", "same", "other"])
def test_an_exports_engine_wins_whatever_the_launch_stamped(tmp_path, launched):
    meta = {**META, "engine": launched}
    exported = _bundle({"turbotab": "2.0.0.dev0", "commit": "e1", "modified": True, "stages": {}})
    assert _engine(tmp_path, meta, exported) == ("e1", True)


@pytest.mark.parametrize("launched", [None, {"commit": None, "engine_tree": "t0",
                                             "engine_modified": False},
                                      {"commit": "c0", "engine_tree": "t0"},
                                      {"commit": "c0", "engine_modified": "false"},
                                      ["c0", False]],
                         ids=["no_stamp", "no_commit", "no_modified", "modified_not_a_flag",
                              "not_a_stamp"])
def test_a_run_with_no_export_and_no_whole_stamp_names_no_engine(tmp_path, launched):
    """A launch with no engine (every launch before the stamp) keeps the nulls; so does one whose
    stamp does not name both a commit and whether the engine's files were modified."""
    assert _engine(tmp_path, {**META, "engine": launched}) == (None, None)


def test_the_engine_is_never_read_from_driver_json(tmp_path):
    """driver.json's ``commit`` (the driver's ``_git_head``), or an ``engine`` it held, is no
    stamp of the engine: with no export and no engine given, the record names none."""
    assert _engine(tmp_path, dict(META), commit="d" * 40, engine=LAUNCHED) == (None, None)

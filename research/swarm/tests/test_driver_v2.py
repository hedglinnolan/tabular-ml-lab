"""The driver's package-1 commands (SWARM_PLAN §2; ``driver.py``), each guard both ways, with the
server's reads stood in for by what a server serves (a question, a project view, a fit's status),
so what each command does is known from construction: a preview says why; an answer names a marker
of the vocabulary and may name the alternative it rejected; a declaration line is answered by id;
a want and the account are kept, the account once; a fit the engine refused in words is a refusal
with its ways forward, and one that crashed is not; the markers are kept and never fail a command.
A real run through all of it is ``test_driver.py``'s slow one."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from research.swarm import declarations as decl
from research.swarm import driver
from research.swarm import vocabulary as V
from research.swarm.tests.test_driver import RID, fake_run, run_main

MISSING = {"kind": "set_missing", "strategy": "impute"}
COMPLETE = {"kind": "set_missing", "strategy": "complete_case"}
VIEW = {"state": {"purpose": "prediction", "target": "y", "selection": None, "seal_opened": None},
        "stages": {"fit": {"key": "k1", "status": "fresh"}}, "decisions": []}


def question(declared: dict[str, Any] | None = None) -> dict[str, Any]:
    """What ``the_question`` gives ``preview`` and ``answer``: the missing-values question's two
    options, and the declaration lines' composed options."""
    composed = {"options": [{"id": "impute", "label": "Fill", "decision": dict(MISSING)},
                            {"id": "complete_case", "label": "Complete cases",
                             "decision": dict(COMPLETE)}], "fields": []}
    return {"_composed": composed, "_ways": [], "_view": VIEW, "_declared": declared or {}}


def selection() -> dict[str, Any]:
    """The selection line as ``declarations.compose`` gives it, on a view with three columns."""
    def served(stage: str) -> Any:
        return {"columns": [{"name": c} for c in ("y", "x1", "x2")]} if stage == "ingest" else None

    return decl.compose("set_selection", VIEW, served)


@pytest.fixture
def posted(tmp_path, monkeypatch):
    """A run whose server reads are stood in for; every decision the driver would post is kept."""
    fake_run(tmp_path, monkeypatch)
    sent: list[tuple[str, Any]] = []

    class Answered:
        status = 200

        def json(self) -> Any:
            return {"decisions": [{"id": "r9", "seq": 9, "note": "n",
                                   "decision": sent[-1][1] if sent else None}],
                    "interview": []}

        def error(self) -> dict[str, Any]:
            return {}

    monkeypatch.setattr(driver, "the_question", lambda run, client, wait: question(
        {"set_selection": selection()}))
    monkeypatch.setattr(driver, "guard_budget", lambda *a, **k: None)

    def post(run, client, decision, note):
        sent.append(("answer", decision))
        return Answered()

    monkeypatch.setattr(driver, "post_decision", post)

    def preview(self, path, data=None, **query):
        sent.append(("preview", data))
        return driver.Response(200, json.dumps({"kind": data["kind"], "views": []}).encode(), {})

    monkeypatch.setattr(driver.Client, "post", preview)
    return sent


def logged(tmp_path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (tmp_path / RID / "commands.jsonl").read_text().splitlines()]


# ── a preview says why ───────────────────────────────────────────────────────


def test_a_preview_says_why_or_is_not_read_and_a_blank_one_is_refused(posted, tmp_path):
    assert driver.main(["preview", "--run", RID, "impute"], out=lambda s: None) == 2
    code, payload = run_main(["preview", "--run", RID, "impute", "--reason", "  "])
    assert code == 1 and payload["refused"]["code"] == "reason_required" and posted == []
    code, payload = run_main(["preview", "--run", RID, "impute", "--reason", "does a fill keep rows?"])
    assert code == 0 and posted == [("preview", MISSING)]
    assert logged(tmp_path)[-1]["args"]["reason"] == "does a fill keep rows?"


# ── an answer names a marker, and may name the alternative it rejected ──────


def test_an_answer_names_a_marker_of_the_vocabulary(posted, tmp_path):
    assert driver.main(["answer", "--run", RID, "impute", "--reason", "r"], out=lambda s: None) == 2
    code, payload = run_main(["answer", "--run", RID, "impute", "--reason", "r", "--marker",
                              "hunch"])
    assert code == 1 and payload["refused"]["code"] == "marker_unknown" and posted == []
    assert "rows" in payload["refused"]["message"]  # the vocabulary is listed
    code, payload = run_main(["answer", "--run", RID, "impute", "--reason", "r", "--marker",
                              "missingness"])
    assert code == 0 and posted == [("answer", MISSING)]
    assert payload["tag"] == {"marker": "missingness", "rejected": None}
    assert logged(tmp_path)[-1]["summary"]["tag"] == {"marker": "missingness", "rejected": None}


@pytest.mark.parametrize("rejected", ["impute", "listwise", "set_selection:vip"])
def test_the_rejected_alternative_is_another_option_of_the_same_question_or_refused(posted,
                                                                                    rejected):
    """The chosen option itself, an option not offered, and a declaration's option beside the
    missing-values question are each refused, and nothing is posted."""
    code, payload = run_main(["answer", "--run", RID, "impute", "--reason", "r", "--marker",
                              "missingness", "--rejected", rejected])
    assert code == 1 and payload["refused"]["code"] == "not_offered" and posted == []
    assert "complete_case" in payload["refused"]["message"]


def test_the_rejected_alternative_is_kept_with_the_decision_it_would_record(posted):
    code, payload = run_main(["answer", "--run", RID, "impute", "--reason", "r", "--marker",
                              "complete_case_share", "--rejected", "complete_case"])
    assert code == 0
    assert payload["tag"]["rejected"] == {"id": "complete_case", "decision": COMPLETE}


# ── a declaration line is answered and previewed by id ──────────────────────


def test_a_declaration_is_answered_by_its_id_with_its_fields(posted):
    code, payload = run_main(["answer", "--run", RID, "set_selection:univariable", "--set",
                              "threshold=0.05", "--reason", "a strict screen",
                              "--marker", "predictors_per_row",
                              "--rejected", "set_selection:none"])
    assert code == 0, payload
    kind, sent = posted[-1]
    assert kind == "answer" and sent["kind"] == "set_selection"
    assert (sent["method"], sent["threshold"]) == ("univariable", 0.05)
    assert payload["tag"]["rejected"]["decision"]["method"] == "none"


@pytest.mark.parametrize("sets,word", [(["threshold=1.5"], "threshold"), (["keep=2.5"], "keep"),
                                       (["where=sideways"], "where"), (["nope=1"], "nope")])
def test_a_declarations_field_outside_what_its_model_takes_is_refused(posted, sets, word):
    args = [a for s in sets for a in ("--set", s)]
    code, payload = run_main(["answer", "--run", RID, "set_selection:screening", *args,
                              "--reason", "r", "--marker", "rows"])
    assert code == 1 and payload["refused"]["code"] == "not_offered" and posted == []
    assert word in payload["refused"]["message"]


def test_a_declaration_is_previewed_by_its_id(posted):
    code, payload = run_main(["preview", "--run", RID, "set_selection:elastic_net", "--reason",
                              "does the penalty keep fewer?"])
    assert code == 0 and posted[-1][0] == "preview"
    assert posted[-1][1]["kind"] == "set_selection" and posted[-1][1]["method"] == "elastic_net"


def test_ask_lists_each_open_declaration_beside_the_question_and_keeps_the_questions_own(
        tmp_path, monkeypatch):
    """The open question's options stay its own when declaration lines are listed beside it (the
    walk of 2026-10-22 found the last line's options in their place, so no option of the
    question was offered); a Confirm line set for the person is listed too, a line waiting is
    not, and a composer that fails says so on its line alone."""
    fake_run(tmp_path, monkeypatch)
    run = driver.Run.load(RID)
    step = {"key": "missing", "status": "open"}
    lines = [{"source": "question", "key": "missing", "status": "open", "counted": True},
             {"source": "declaration", "key": "set_selection", "status": "open", "counted": True,
              "name": "the predictor-selection question"},
             {"source": "declaration", "key": "set_levers", "status": "set_for_you",
              "counted": False, "name": "the rules repeated inside each fold"},
             {"source": "declaration", "key": "set_updating", "status": "waiting",
              "counted": True, "name": "the shrinkage question"},
             {"source": "declaration", "key": "set_explain", "status": "open", "counted": False,
              "name": "the explanations"}]
    monkeypatch.setattr(driver, "wait_open", lambda run, client, wait: (VIEW, step))
    monkeypatch.setattr(driver, "quest_lines", lambda run, client: (lines, None))
    monkeypatch.setattr(driver, "teaching", lambda client, purpose, key: None)
    monkeypatch.setattr(driver, "finding_options", lambda run, client, purpose: {})
    monkeypatch.setattr(driver, "served", lambda run, client: (lambda stage: None))
    monkeypatch.setattr(driver.opts, "compose", lambda key, step, view, served, entry: {
        "options": question()["_composed"]["options"], "fields": [], "data": {}})
    real = decl.compose

    def compose(kind, view, served):
        if kind == "set_explain":
            raise KeyError("curves")
        return real(kind, view, served)

    monkeypatch.setattr(driver.decl, "compose", compose)
    q = driver.the_question(run, driver.Client(9), 0)
    assert [o["id"] for o in q["_composed"]["options"]] == ["impute", "complete_case"]
    assert [o["id"] for o in q["options"]] == ["impute", "complete_case"]
    assert set(q["_declared"]) == {"set_selection", "set_levers"}
    listed = {ln["key"]: ln for ln in q["also_open"]}
    assert set(listed) == {"set_selection", "set_levers", "set_explain"}
    assert listed["set_selection"]["options"][0]["id"] == "set_selection:none"
    assert "KeyError" in listed["set_explain"]["options_why"]
    assert "options" not in listed["set_explain"]


# ── a want and the account ───────────────────────────────────────────────────


def test_a_want_is_kept_with_its_kind_reason_and_marker(tmp_path, monkeypatch):
    fake_run(tmp_path, monkeypatch)
    good = ["want", "--run", RID, "drift correction by run order", "--kind", "option",
            "--reason", "the QC samples drift", "--marker", "run_order"]
    code, payload = run_main(good)
    assert code == 0
    wanted = payload["wanted"]
    assert {k: wanted[k] for k in ("kind", "what", "reason", "marker")} == {
        "kind": "option", "what": "drift correction by run order",
        "reason": "the QC samples drift", "marker": "run_order"}
    assert logged(tmp_path)[-1]["summary"]["wanted"] == wanted
    for i, bad, code_word in ((5, "feature", "want_kind_unknown"), (9, "hunch", "marker_unknown"),
                              (7, " ", "reason_required"), (3, " ", "want_unsaid")):
        args = list(good)
        args[i] = bad
        code, payload = run_main(args)
        assert code == 1 and payload["refused"]["code"] == code_word, bad
    assert set(V.WANT_KINDS) == {"option", "check", "reading", "eligibility_rule"}


def test_the_account_is_kept_once_marked_after_the_results_and_the_seal(tmp_path, monkeypatch):
    fake_run(tmp_path, monkeypatch)
    sealed = {"opened": False}
    monkeypatch.setattr(driver, "view_of", lambda run, client: {
        "state": {"seal_opened": {"y": 9} if sealed["opened"] else None}})
    code, payload = run_main(["account", "--run", RID, "  "])
    assert code == 1 and payload["refused"]["code"] == "account_unsaid"
    code, payload = run_main(["account", "--run", RID, "I kept the linear model."])
    assert code == 0 and payload["account"]["text"] == "I kept the linear model."
    assert payload["account"]["after_results"] is False and payload["account"]["after_seal"] is False
    code, again = run_main(["account", "--run", RID, "A second account."])
    assert code == 1 and again["refused"]["code"] == "account_given"


def test_an_account_after_the_results_and_the_seal_says_so(tmp_path, monkeypatch):
    folder = fake_run(tmp_path, monkeypatch)
    with (folder / "commands.jsonl").open("a") as fh:
        fh.write(json.dumps({"command": "results", "summary": {"exit": 0}}) + "\n")
    monkeypatch.setattr(driver, "view_of", lambda run, client: {"state": {"seal_opened": {"y": 9}}})
    code, payload = run_main(["account", "--run", RID, "The held-out AUC was the result."])
    assert code == 0
    assert payload["account"]["after_results"] is True and payload["account"]["after_seal"] is True


def test_a_failed_results_command_does_not_count_as_the_results_read(tmp_path, monkeypatch):
    folder = fake_run(tmp_path, monkeypatch)
    with (folder / "commands.jsonl").open("a") as fh:
        fh.write(json.dumps({"command": "results", "summary": {"exit": 1}}) + "\n")
    monkeypatch.setattr(driver, "view_of", lambda run, client: {"state": {}})
    code, payload = run_main(["account", "--run", RID, "Before any results."])
    assert code == 0 and payload["account"]["after_results"] is False


def test_a_stopped_run_takes_no_want_and_no_account(tmp_path, monkeypatch):
    folder = fake_run(tmp_path, monkeypatch)
    meta = json.loads((folder / "driver.json").read_text())
    (folder / "driver.json").write_text(json.dumps({**meta, "ended": "2026-10-18T00:00:00+00:00"}))
    for argv in (["want", "--run", RID, "x", "--kind", "check", "--reason", "y", "--marker", "rows"],
                 ["account", "--run", RID, "z"]):
        code, payload = run_main(argv)
        assert code == 1 and payload["refused"]["code"] == "run_stopped"


# ── a fit the engine refused in words, and one that crashed ──────────────────

REFUSAL = {"stage": "fit", "code": "nothing_kept",
           "message": "The univariable screen kept no predictor on 51 training rows.",
           "exits": [{"label": "No selection: every candidate predictor",
                      "decision": {"kind": "set_selection", "method": "none"}}]}


@pytest.fixture
def fitted(tmp_path, monkeypatch):
    """``fit`` with its server reads stood in for: the press goes through, and the fit's status
    is ``status`` (set by each test)."""
    folder = fake_run(tmp_path, monkeypatch)
    status: dict[str, Any] = {}
    spent = {"fits": 0, "running": 0, "seconds": 0.0, "budget_fits": 3, "budget_seconds": 99.0,
             "keys": [], "running_key": None}
    monkeypatch.setattr(driver, "view_of", lambda run, client: VIEW)
    monkeypatch.setattr(driver, "budget_now", lambda run, client, view=None: dict(spent))
    monkeypatch.setattr(driver.Client, "post", lambda self, path, data=None, **q: driver.Response(
        200, b'{"pressed": true}', {}))
    monkeypatch.setattr(driver, "wait_fit", lambda run, client, seconds: dict(status))
    monkeypatch.setattr(driver, "keep_markers", lambda run, client, command: None)
    return folder, status


def test_a_fit_the_engine_refused_in_words_is_a_refusal_with_its_ways_forward(fitted):
    folder, status = fitted
    status.update(status="error", key="k2", error="ValueError: " + REFUSAL["message"],
                  refusal=REFUSAL)
    code, payload = run_main(["fit", "--run", RID])
    assert code == 1
    refused = payload["refused"]
    assert (refused["by"], refused["stage"], refused["code"]) == ("engine", "fit", "nothing_kept")
    assert refused["message"] == REFUSAL["message"]
    assert [e["decision"] for e in refused["exits"]] == [REFUSAL["exits"][0]["decision"]]
    assert refused["exits"][0]["id"].startswith("x-")
    # Kept, so the next answer can take the way forward by its id.
    kept = json.loads((folder / "exits.json").read_text())
    assert kept["code"] == "nothing_kept" and kept["exits"] == refused["exits"]
    summary = logged(folder.parent)[-1]["summary"]
    assert summary["refused"]["by"] == "engine" and summary["refused"]["stage"] == "fit"


def test_a_fit_that_crashed_is_said_as_it_failed_and_is_no_refusal(fitted):
    folder, status = fitted
    status.update(status="error", key="k2",
                  error="ValueError: at least one array or dtype is required")
    code, payload = run_main(["fit", "--run", RID])
    assert code == 0 and "refused" not in payload
    assert payload["fit"]["status"] == "error" and "refusal" not in payload["fit"]
    assert json.loads((folder / "exits.json").read_text())["exits"] == []  # no way forward kept


def test_a_fit_that_succeeded_is_neither(fitted):
    _, status = fitted
    status.update(status="fresh", key="k2")
    code, payload = run_main(["fit", "--run", RID])
    assert code == 0 and payload["fit"] == {"status": "fresh", "key": "k2"}


# ── the markers ──────────────────────────────────────────────────────────────


def test_the_markers_are_kept_by_the_current_fits_key(tmp_path, monkeypatch):
    folder = fake_run(tmp_path, monkeypatch)
    run = driver.Run.load(RID)
    monkeypatch.setattr(driver, "view_of", lambda run, client: VIEW)
    monkeypatch.setattr(driver, "reader", lambda run, client: (lambda path: None))
    driver.keep_markers(run, driver.Client(9), "results")
    kept = [json.loads(x) for x in (folder / driver.MARKERS_FILE).read_text().splitlines()]
    assert len(kept) == 1 and kept[0]["fit_key"] == "k1" and kept[0]["command"] == "results"
    assert [m["name"] for m in kept[0]["markers"]] == list(V.LIVE)
    assert next(m for m in kept[0]["markers"] if m["name"] == "purpose")["value"] == "prediction"


def test_markers_that_cannot_be_read_keep_nothing_and_never_fail_the_command(tmp_path, monkeypatch):
    folder = fake_run(tmp_path, monkeypatch)
    run = driver.Run.load(RID)

    def broken(run, client):
        raise driver.DriverError("server_unreachable", "no")

    monkeypatch.setattr(driver, "view_of", broken)
    driver.keep_markers(run, driver.Client(9), "fit")  # no exception
    assert not (folder / driver.MARKERS_FILE).exists()


# ── a real run: a declaration answered, a fit refused in words, its way forward taken ─────────

NOISE_PLAN = {
    "lens": [(["clinical"], [])], "target": [(["y"], [])], "task": [(["regression"], [])],
    "follow_up": [(["same"], [])], "purpose": [(["prediction"], [])],
    "grain": [(["one_row_per_unit"], [])], "roles": [(["proposed"], [])],
    "exclusions": [(["none"], [])], "missing": [(["impute"], [])], "split": [(["0.2"], [])],
    "models": [(["linear"], [])],
}


def noise_table(path: Path) -> Path:
    """120 rows: an outcome and four predictors drawn independently (seed 0), so a univariable
    screen at p < 10⁻⁶ keeps no predictor in any training fold (each p-value is uniform under the
    construction; one below 10⁻⁶ in 4 × 50 tests has probability about 2 × 10⁻⁴, and the seed
    fixes the draw)."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"y": rng.normal(size=120).round(4),
                          **{f"x{i}": rng.normal(size=120).round(4) for i in range(1, 5)}})
    frame.to_csv(path, index=False)
    return path


@pytest.mark.slow
def test_a_declaration_a_fit_refused_in_words_and_its_way_forward_through_the_driver(tmp_path):
    from research.swarm import assemble as A
    from research.swarm.record import validate
    from research.swarm.tests import test_driver as T

    data = noise_table(tmp_path / "noise.csv")
    analyst = T.Analyst(tmp_path / "runs", "", plan=NOISE_PLAN)
    out = analyst.drv("start", "--data", str(data), "--budget-fits", "4", "--budget-seconds",
                      "900", "--scripted", "--model", "scripted", "--persona", "skeptical")
    meta = out["started"]
    analyst.run = meta["run"]
    assert (meta["persona"], meta["vocabulary"]) == ("skeptical", V.DIGEST)
    try:
        analyst.until_fit()
        q = analyst.cmd("ask")
        line = next(ln for ln in q.get("also_open") or [] if ln.get("key") == "set_selection")
        assert [o["id"] for o in line["options"]][:2] == ["set_selection:none",
                                                          "set_selection:elastic_net"]
        assert "threshold" in {f["name"] for f in line["fields"]}
        p = analyst.cmd("preview", "set_selection:univariable", "--set", "threshold=0.000001",
                        "--reason", "Would a strict screen keep any of four noise columns?")
        assert p["previewed"]["method"] == "univariable" and "preview" in p
        rec = analyst.cmd("answer", "set_selection:univariable", "--set", "threshold=0.000001",
                          "--reason", "A strict screen, to see what it keeps.",
                          "--marker", "predictors_per_row", "--rejected", "set_selection:none")
        assert rec["recorded"]["decision"]["kind"] == "set_selection"
        assert rec["tag"]["rejected"]["id"] == "set_selection:none"

        refused = analyst.cmd("fit", "--wait", "300", expect=1)["refused"]
        assert (refused["by"], refused["code"]) == ("engine", "nothing_kept"), refused
        ways = {e["decision"]["method"]: e["id"] for e in refused["exits"]}
        assert set(ways) == {"none", "screening"}
        asked = analyst.cmd("ask")
        failed = asked["fit_failed"]
        assert failed[0]["kind"] == "refused" and failed[0]["refusal"]["code"] == "nothing_kept"

        analyst.cmd("answer", ways["none"], "--reason", "No selection, the refusal's way forward.",
                    "--marker", "predictors_per_row")
        fitted = analyst.cmd("fit", "--wait", "300")
        assert fitted["fit"]["status"] == "fresh", fitted
        analyst.cmd("want", "an intercept-only model where a selection keeps nothing",
                    "--kind", "option", "--reason", "The screen kept no predictor.",
                    "--marker", "predictors_per_row")
        analyst.cmd("results", "--wait", "120")
        analyst.cmd("account", "The strict screen kept nothing; with no selection the fit ran.")

        path, ok = A.assemble(tmp_path / "runs" / analyst.run, corpus=tmp_path / "corpus",
                              meta={"design": 2, "analyst_model": "scripted",
                                    "prompt_version": "test", "dataset": {"id": "noise"}},
                              mattered_wait=0)
        assert ok, json.loads(path.read_text())["error"]
        record = validate(json.loads(path.read_text("utf-8")))
    finally:
        analyst.drv("stop", "--run", analyst.run)
    (refusal,) = [f for f in record.surfacings.failed_fits if f.kind == "refused"]
    assert refusal.refusal["code"] == "nothing_kept" and refusal.stage == "fit"
    assert any(r.by == "engine" and r.code == "nothing_kept" for r in record.surfacings.refusals)
    tagged = [d for d in record.decisions if d.kind == "set_selection"]
    assert [d.tag.marker for d in tagged] == ["predictors_per_row", "predictors_per_row"]
    assert tagged[0].tag.rejected["decision"]["method"] == "none"
    assert record.run.persona == "skeptical"
    assert record.attempts and record.attempts[-1].markers is not None
    rows = next(m for m in record.attempts[-1].markers if m.name == "rows")
    assert rows.value == 120
    assert record.wants[0].kind == "option" and record.account.after_results is True
    previewed = [e for d in tagged for e in d.examined if e.how == "preview"]
    assert previewed and previewed[0].reason == "Would a strict screen keep any of four noise columns?"


# ── the run's model and persona ──────────────────────────────────────────────


def test_start_takes_the_model_and_a_persona_of_the_three():
    p = driver.build_parser()
    base = ["start", "--data", "x.csv", "--budget-fits", "1", "--budget-seconds", "1"]
    args = p.parse_args([*base, "--model", "claude-haiku-5-5", "--persona", "explorer"])
    assert (args.model, args.persona) == ("claude-haiku-5-5", "explorer")
    with pytest.raises(SystemExit):
        p.parse_args([*base, "--persona", "bold"])
    assert driver.PERSONAS == ("conventional", "skeptical", "explorer")

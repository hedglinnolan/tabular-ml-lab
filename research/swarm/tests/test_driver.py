"""The swarm's driver (DRIVER_PLAN D1), tested on its guards and against a real local server.

The guards, both ways, on records built here: the fit budget refuses a press past its count or
its seconds and lets one within them through; the decisions the driver never posts are refused
and every other passes; a reason is required.

The real runs drive ``clinical_risk.csv`` (480 encounters, a yes/no readmission) through
``python -m research.swarm.driver`` as an analyst would, one command at a time, under Predict:
a scripted analyst answers each open question with an option the server offers, by its id, and
answers every reading the server asks from the fixture's declared truth
(``truths.FIXTURE_TRUTHS["clinical_risk.csv"]``, BLUEPRINT §14.3). What each step should give is
known from construction or read independently of the driver: the file's SHA-256 by hashlib, each
reason as the note on its own record in ``GET /projects/{pid}``, the fits from ``GET /fits``, the
bundle's SHA-256 from the saved file, the provenance from the bundle's ``provenance.json``, and the
processes from ``ps``."""
from __future__ import annotations

import hashlib
import io
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import pytest

from research.swarm import driver

REPO = Path(__file__).resolve().parents[3]
FIXTURE = REPO / "turbotab" / "sample_data" / "clinical_risk.csv"
PROVENANCE = "https://example.org/data/clinical_risk.csv,2026-08-02,CC0-1.0"


# ── the guards, both ways ────────────────────────────────────────────────────


def spent(fits: int, seconds: float, *, budget_fits: int = 2, budget_seconds: float = 100.0,
          keys: list[str] | None = None, running: str | None = None) -> dict[str, Any]:
    return {"fits": fits, "running": int(running is not None), "seconds": seconds,
            "budget_fits": budget_fits, "budget_seconds": budget_seconds,
            "keys": keys or [f"k{i}" for i in range(fits)], "running_key": running}


def test_the_fit_budget_refuses_a_press_past_its_count():
    """Two fits budgeted: with none or one made, a press that would make the next passes; with two
    made, a press that would make a third is refused, saying the budget and the count."""
    assert driver.over_budget(spent(0, 0.0), None) is None
    assert driver.over_budget(spent(1, 9.0), "new") is None
    why = driver.over_budget(spent(2, 18.0), "new")
    assert why == "Fit is not pressed: the run's fit budget is 2 fits, and 2 were made already."
    assert driver.over_budget(spent(1, 9.0, budget_fits=1), "new") == (
        "Fit is not pressed: the run's fit budget is 1 fit, and 1 was made already.")


def test_a_fit_already_made_for_these_answers_is_not_counted_twice():
    """The scheduler fits a quick fit before the press: pressing Fit for that fit counts it once,
    so the budget's last fit can still be pressed; one past it cannot, recorded or not."""
    assert driver.over_budget(spent(2, 18.0, keys=["a", "b"]), "b") is None
    assert driver.over_budget(spent(3, 27.0, keys=["a", "b", "c"]), "c") is not None
    assert driver.over_budget(spent(3, 27.0, keys=["a", "b", "c"]), "a") is not None


def test_the_fit_budget_refuses_a_press_once_its_seconds_are_spent():
    assert driver.over_budget(spent(1, 99.9), "new") is None
    why = driver.over_budget(spent(1, 100.0), "new")
    assert why == ("Fit is not pressed: the run's fit budget is 100 seconds of fitting, and 100 "
                   "were spent already.")
    assert driver.over_budget(spent(1, 250.0, keys=["a"]), "a") is not None


def test_a_fit_under_way_counts_as_made():
    """The fits made are GET /fits' records and the fit the scheduler is making now: the current
    fit's key, recorded by no record, neither held for Fit nor failed. With one under way, the
    budget's count is met one fit sooner, and its words say so."""
    records = [{"fit_key": "a", "seconds": 2.5}]
    budget = {"fits": 2, "seconds": 100.0}
    for fit, running in [({"status": "running", "key": "b"}, "b"),
                         ({"status": "queued", "key": "b"}, "b"),
                         ({"status": "stale", "key": "b", "held": "estimate"}, "b"),
                         ({"status": "stale", "key": "b", "held": "fit"}, None),  # waits for Fit
                         ({"status": "fresh", "key": "a"}, None),  # recorded
                         ({"status": "error", "key": "b"}, None),
                         ({"status": "blocked", "key": None}, None),
                         ({"status": "running", "key": "b", "cancelled": True}, None)]:
        got = driver.spent_of(records, fit, budget)
        assert (got["running_key"], got["running"]) == (running, int(running is not None)), fit
        assert (got["fits"], got["seconds"], got["keys"]) == (1, 2.5, ["a"])
    met = driver.spent_of(records, {"status": "running", "key": "b"}, budget)
    assert driver.budget_met(met) and not driver.budget_met(driver.spent_of(records, {}, budget))
    assert driver.over_budget(met, "b") is None  # a press for the fit under way counts it once
    one = driver.spent_of(records, {"status": "running", "key": "b"}, {"fits": 1, "seconds": 100.0})
    assert driver.over_budget(one, "b") == ("Fit is not pressed: the run's fit budget is 1 fit, "
                                            "and 1 was made and 1 is under way.")


def test_past_the_budget_an_answer_that_makes_a_new_fit_is_refused():
    """Within the budget every answer passes, whatever fit it makes. Once the fits made meet it, an
    answer passes only when the fit it leaves is none (the models unanswered), one already made,
    or the one under way; one that makes a new fit is refused, by count or by seconds."""
    within = spent(1, 9.0, keys=["a"])
    assert driver.attempt_over_budget(within, "new") is None
    met = spent(2, 18.0, keys=["a", "b"])
    for after in (None, "a", "b"):
        assert driver.attempt_over_budget(met, after) is None, after
    assert driver.attempt_over_budget(met, "new") == (
        "Not recorded: this would make a new fit, and the run's fit budget is 2 fits, and 2 were "
        "made already. An answer that makes no new fit is still recorded, and results, seal and "
        "export still run.")
    running = spent(1, 9.0, keys=["a"], running="b")
    assert driver.attempt_over_budget(running, "b") is None
    assert driver.attempt_over_budget(running, "new").startswith(
        "Not recorded: this would make a new fit, and the run's fit budget is 2 fits, and 1 was "
        "made and 1 is under way.")
    seconds = spent(1, 100.0, keys=["a"])
    assert driver.attempt_over_budget(seconds, "a") is None
    assert driver.attempt_over_budget(seconds, "new").startswith(
        "Not recorded: this would make a new fit, and the run's fit budget is 100 seconds of "
        "fitting, and 100 were spent already.")


def test_the_fit_an_answer_would_make_is_told_only_where_the_servers_keys_are_matched(tmp_path):
    """The driver tells the fit an answer would leave by the engine's own key rule
    (``graph.stage_keys`` over the log folded with the answer). It trusts that rule only where the
    keys it computes for the log as it stands are the keys the server serves: with one stage's
    key not the server's, it cannot tell, and says why; with the server's keys, it answers."""
    from turbotab.core import decisions
    from turbotab.core.graph import stage_keys
    from turbotab.core.workspace import stage_graph

    run = driver.Run(tmp_path, {"fingerprint": "f" * 32, "data": {"path": str(FIXTURE)}})
    keys = stage_keys(stage_graph().order(), decisions.fold([]).model_dump(mode="json"), "f" * 32)
    view = {"decisions": [], "stages": {name: {"key": key} for name, key in keys.items()}}
    assert driver.fit_key_after(run, view, {"kind": "set_purpose", "purpose": "prediction"}) is None
    wrong = {**view, "stages": {**view["stages"], "ingest": {"key": "0" * 64}}}
    with pytest.raises(driver.CannotTell) as caught:
        driver.fit_key_after(run, wrong, {"kind": "set_purpose", "purpose": "prediction"})
    assert "ingest" in str(caught.value)
    with pytest.raises(driver.CannotTell):  # not a decision
        driver.fit_key_after(run, view, {"kind": "no_such_kind"})


@pytest.mark.parametrize("kind", ["reseal", "lock_plan", "revert", "open_seal"])
def test_the_decisions_the_driver_never_answers_with_are_refused(kind):
    with pytest.raises(driver.DriverError) as caught:
        driver.guard_kind({"kind": kind})
    assert caught.value.code == ("no_reseal" if kind == "reseal" else "not_this_way")


@pytest.mark.parametrize("kind", ["set_missing", "confirm_readings", "apply_repair", "confirm_sweep",
                                  "select_models", "set_split"])
def test_every_other_decision_passes(kind):
    driver.guard_kind({"kind": kind})


def test_a_reseal_is_refused_even_where_open_seal_is_previewed():
    driver.guard_kind({"kind": "open_seal"}, allowed=("open_seal",))
    with pytest.raises(driver.DriverError):
        driver.guard_kind({"kind": "reseal"}, allowed=("open_seal",))


def test_after_the_seal_nothing_more_is_tried():
    with pytest.raises(driver.DriverError) as caught:
        driver.guard_sealed({"state": {"seal_opened": {"readmit_30d": 7}}})
    assert caught.value.code == "after_seal"
    driver.guard_sealed({"state": {"seal_opened": None}})


def fake_run(tmp_path: Path, monkeypatch, run_id: str = "r1") -> Path:
    monkeypatch.setenv(driver.RUNS_ENV, str(tmp_path))
    folder = tmp_path / run_id
    folder.mkdir()
    (folder / "driver.json").write_text(json.dumps({"run": run_id, "pid": None, "port": 9,
                                                    "project": "p0", "budget": {"fits": 1, "seconds": 1},
                                                    "ended": None}))
    return folder


def run_main(argv: list[str]) -> tuple[int, Any]:
    out: list[str] = []
    code = driver.main(argv, out=out.append)
    return code, json.loads(out[0]) if out else None


def test_an_answer_says_why(tmp_path, monkeypatch):
    """A reason is required: none at all is a command line that does not read (2); a blank one is
    refused by the driver before anything is posted (1), and the refusal is logged in the run."""
    folder = fake_run(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        driver.build_parser().parse_args(["answer", "--run", "r1", "impute"])
    assert driver.main(["answer", "--run", "r1", "impute"], out=lambda s: None) == 2
    code, payload = run_main(["answer", "--run", "r1", "impute", "--reason", "   "])
    assert code == 1 and payload["refused"]["code"] == "reason_required"
    logged = [json.loads(line) for line in (folder / "commands.jsonl").read_text().splitlines()]
    assert logged[-1]["command"] == "answer" and logged[-1]["summary"]["refused"]["code"] == "reason_required"


@pytest.mark.parametrize("run_id", ["../etc", "a/b", "", ".hidden", "x" * 81])
def test_a_run_is_named_by_its_id_alone(tmp_path, monkeypatch, run_id):
    fake_run(tmp_path, monkeypatch)
    code, payload = run_main(["ask", "--run", run_id])
    assert code == 1 and payload["refused"]["code"] == "no_such_run"


def test_a_server_that_does_not_answer_is_said_in_json(tmp_path, monkeypatch):
    """The fake run's port has nothing listening: the command answers with the driver's refusal,
    never a traceback; a defect inside a command is answered the same way, and both are logged."""
    folder = fake_run(tmp_path, monkeypatch)
    meta = json.loads((folder / "driver.json").read_text())
    (folder / "driver.json").write_text(json.dumps({**meta, "port": driver.free_port()}))  # closed
    code, payload = run_main(["ask", "--run", "r1", "--wait", "0"])
    assert code == 1 and payload["refused"]["code"] == "server_unreachable"

    def broken(args):
        raise KeyError("columns")

    monkeypatch.setitem(driver.COMMANDS, "results", broken)
    code, payload = run_main(["results", "--run", "r1"])
    assert code == 1 and payload["refused"]["code"] == "driver_failed"
    assert "KeyError" in payload["refused"]["message"]
    logged = [json.loads(line)["summary"]["refused"]["code"]
              for line in (folder / "commands.jsonl").read_text().splitlines()]
    assert logged == ["server_unreachable", "driver_failed"]


def test_a_stopped_run_takes_no_more_commands(tmp_path, monkeypatch):
    folder = fake_run(tmp_path, monkeypatch)
    meta = json.loads((folder / "driver.json").read_text())
    (folder / "driver.json").write_text(json.dumps({**meta, "ended": "2026-10-18T00:00:00+00:00"}))
    code, payload = run_main(["ask", "--run", "r1"])
    assert code == 1 and payload["refused"]["code"] == "run_stopped"


HOOKED = {"hooks": {"PreToolUse": [{"matcher": "*", "hooks": [
    {"type": "command", "command": "python3 /repo/research/swarm/analyst_hook.py"}]}]}}


@pytest.mark.parametrize("settings,wired", [
    (HOOKED, True),
    ({"hooks": {"PreToolUse": [{"hooks": [{"type": "command",
                                           "command": "python3 research/swarm/analyst_hook.py"}]}]}}, True),
    ({}, False),
    ({"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": HOOKED["hooks"]["PreToolUse"][0]["hooks"]}]}},
     False),  # not every tool
    ({"hooks": {"PostToolUse": HOOKED["hooks"]["PreToolUse"]}}, False),  # after the call, too late
    ({"hooks": {"PreToolUse": [{"matcher": "*", "hooks": [
        {"type": "command", "command": "python3 .claude/hooks/guard.py"}]}]}}, False),  # another hook
])
def test_the_hook_counts_as_wired_only_before_every_tool(tmp_path, settings, wired):
    path = tmp_path / "settings.local.json"
    path.write_text(json.dumps(settings))
    assert (driver.hook_wired((tmp_path / "missing.json", path)) == str(path)) is wired


def test_a_run_does_not_start_where_the_analyst_is_not_held(tmp_path, monkeypatch):
    """Without the hook wired, ``start`` is refused before any folder or server exists; with it
    wired, or for a scripted analyst, the guard passes and says which."""
    monkeypatch.setenv(driver.RUNS_ENV, str(tmp_path / "runs"))
    monkeypatch.setattr(driver, "SETTINGS_FILES", (tmp_path / "none.json",))
    code, payload = run_main(["start", "--data", str(FIXTURE), "--budget-fits", "1",
                              "--budget-seconds", "60", "--run", "unguarded"])
    assert code == 1 and payload["refused"]["code"] == "unguarded"
    assert "analyst_hook.py" in payload["refused"]["message"]
    assert not (tmp_path / "runs").exists()
    assert driver.guarded(True) == {"analyst": "scripted"}
    wired = tmp_path / "settings.local.json"
    wired.write_text(json.dumps(HOOKED))
    monkeypatch.setattr(driver, "SETTINGS_FILES", (wired,))
    assert driver.guarded(False) == {"analyst": "agent", "hook": str(wired)}


def test_the_analyst_type_has_bash_alone(tmp_path):
    assert driver.agent_restricted() is True  # .claude/agents/analyst.md as committed
    other = tmp_path / "analyst.md"
    other.write_text("---\nname: analyst\ntools: Bash, Read\n---\nbody\n")
    assert driver.agent_restricted(other) is False
    other.write_text("---\nname: analyst\n---\nbody\n")  # every tool
    assert driver.agent_restricted(other) is False


def test_long_lists_are_cut_with_a_count():
    assert driver.trim({"points": list(range(45)), "n": 3}, limit=40) == {
        "points": [*range(40), {"more": 5}], "n": 3}
    assert driver.trim([1, 2], limit=40) == [1, 2]


# ── a real run ───────────────────────────────────────────────────────────────


def truth() -> dict[str, str]:
    from turbotab.core.tests.truths import FIXTURE_TRUTHS

    return dict(FIXTURE_TRUTHS["clinical_risk.csv"])


Plan = dict[str, list[tuple[list[str], list[str]]]]  # each question's answers, in turn

# Under Predict: the outcome, one row each, the proposed roles, every row, a fill in each training
# fold (complete cases previewed first), a fifth held out with k-fold beside it, logistic regression.
PREDICT: Plan = {
    "lens": [(["clinical"], [])],
    "target": [(["readmit_30d"], [])],
    "event": [(["1"], [])],
    "follow_up": [(["same"], [])],
    "purpose": [(["prediction"], [])],
    "grain": [(["one_row_per_unit"], [])],
    "roles": [(["proposed"], [])],
    "exclusions": [(["none"], [])],
    "missing": [(["impute"], [])],
    "split": [(["0.2"], ["validation=kfold"])],
    "models": [(["linear"], [])],
}
# Under Estimate, the test's own question (no causal truth is declared for this fixture): the
# comorbidity index's total effect on readmission as a conditional odds ratio, the labs as
# covariates; age and sex confounders by their group's one tap, every other covariate answered per
# column; age entered as a straight line, so the curve of which decisions mattered can be drawn.
LABS = ("prior_admissions_12mo", "albumin_g_dl", "creatinine_mg_dl", "hemoglobin_g_dl",
        "sodium_mmol_l", "length_of_stay_days")
ESTIMATE: Plan = {
    **{k: PREDICT[k] for k in ("lens", "target", "event", "follow_up")},
    "purpose": [(["inference"], [])],
    "grain": PREDICT["grain"],
    "roles": [(["proposed"], [f"{c}=covariate" for c in LABS])],
    "clusters": [(["none"], [])],
    "exclusions": [(["none"], [])],
    "missing": [(["complete_case"], [])],
    "estimand": [(["charlson_index"], ["measure=odds_ratio"])],
    "adjustment": [(["demographic"], []),
                   (["answers"], ["prior_admissions_12mo=yes,yes,no", "albumin_g_dl=unknown,yes,yes",
                                  "creatinine_mg_dl=unknown,yes,yes", "hemoglobin_g_dl=unknown,yes,yes",
                                  "sodium_mmol_l=unknown,yes,yes", "length_of_stay_days=no,yes,yes"])],
    "form": [(["proposed"], ["age=linear"])],
    "models": [(["linear"], [])],
}


class Analyst:
    """A scripted analyst working one run through the driver's command line."""

    def __init__(self, runs: Path, run_id: str, plan: Plan | None = None,
                 before: dict[str, Any] | None = None):
        self.runs, self.run = runs, run_id
        self.plan = {k: list(v) for k, v in (plan or PREDICT).items()}
        self.before = dict(before or {})  # question -> a check run on its ask before answering
        self.truth = truth()
        self.proposed: dict[str, str] = {}
        self.reasons: list[str] = []  # every reason given, in order
        self.asked: list[str] = []

    def drv(self, *args: str, expect: int | None = 0) -> dict[str, Any]:
        env = {**os.environ, driver.RUNS_ENV: str(self.runs), "OMP_NUM_THREADS": "2",
               "TURBOTAB_WORKERS": "2"}
        r = subprocess.run([sys.executable, "-m", "research.swarm.driver", *args], cwd=REPO, env=env,
                           capture_output=True, text=True, timeout=300)
        assert r.stdout.count("\n") == 1, (args, r.stdout[-2000:], r.stderr[-2000:])
        payload = json.loads(r.stdout)
        if expect is not None:
            assert r.returncode == expect, (args, json.dumps(payload)[:3000], r.stderr[-1500:])
        return payload

    def cmd(self, name: str, *args: str, expect: int | None = 0) -> dict[str, Any]:
        return self.drv(name, "--run", self.run, *args, expect=expect)

    def answer(self, ids: list[str], sets: list[str] = (), why: str = "") -> dict[str, Any]:
        reason = why or f"Scripted analyst, answer {len(self.reasons) + 1}: {' '.join(ids)}."
        self.reasons.append(reason)
        out = self.cmd("answer", *ids, *[a for s in sets for a in ("--set", s)], "--reason", reason)
        assert set(out) == {"recorded", "next"}, out
        assert out["recorded"]["note"] == reason and out["recorded"]["id"]
        return out

    # what the fixture's author answers, from what the server offers

    def reading_exit(self, ways: list[dict[str, Any]]) -> str | None:
        """A reading the server asks about, answered from the declared truth: the first way whose
        decision confirms one asked reading at its declared value."""
        for way in ways:
            d = way.get("decision") or {}
            if d.get("kind") == "confirm_reading" and d.get("reading") != "role":
                key = f"{d['reading']}:{d['column']}"
                if self.truth.get(key) == d.get("value"):
                    return way["id"]
        return None

    def role_press(self, line: dict[str, Any]) -> str | None:
        """A role proposed below high confidence, confirmed as proposed (the author's roles are
        the proposal, answered at the roles question)."""
        for item in line.get("items") or []:
            for way in item.get("options") or []:
                d = way.get("decision") or {}
                if d.get("kind") == "confirm_role" and self.proposed.get(d.get("column")) == d.get("role"):
                    return way["id"]
                if d.get("kind") == "confirm_readings" and d.get("items") and all(
                        i.get("reading") == "role" and self.proposed.get(i.get("column")) == i.get("value")
                        for i in d["items"]):
                    return way["id"]
        return None

    def finding_act(self, line: dict[str, Any]) -> str | None:
        """The author's coding of the one two-level text column: male as 1."""
        for way in line.get("options") or []:
            d = way.get("decision") or {}
            if d.get("kind") == "apply_repair" and (d.get("params") or {}).get("one") == "m":
                return way["id"]
        return None

    def until_fit(self, limit: int = 60) -> dict[str, Any]:
        """Answer the Router until it reaches Fit; returns that ask."""
        for _ in range(limit):
            q = self.cmd("ask")
            assert {"run", "question", "decisions", "seal_opened"} <= set(q), q
            assert q["run"] == self.run
            question = q["question"]
            ways =[*((q.get("readings") or {}).get("exits") or []), *(q.get("last_refusal_exits") or []),
                    *(w for line in q.get("also_open") or [] for item in line.get("items") or []
                      for w in item.get("options") or [])]
            way = self.reading_exit(ways)
            if way:
                self.answer([way])
                continue
            pressed = None
            for line in q.get("also_open") or []:
                pressed = self.role_press(line) or (self.finding_act(line)
                                                    if line.get("source") == "finding" else None)
                if pressed:
                    break
            if pressed:
                self.answer([pressed])
                continue
            if q.get("fit_reached"):
                return q
            assert question is not None, q
            key = question["key"]
            assert question["status"] == "open", q
            self.asked.append(key)
            offered = [o["id"] for o in q.get("options") or []]
            assert self.plan.get(key), (key, q)
            ids, sets = self.plan[key].pop(0)
            assert set(ids) <= set(offered), (key, ids, offered)
            if key == "roles":
                self.proposed = {c["column"]: c["proposed"] for c in q["data"]["columns"]}
            if key == "missing" and ids == ["impute"]:
                self.examine("complete_case", q)
            if key in self.before:
                self.before.pop(key)(q)
            self.answer(ids, sets)
        raise AssertionError(f"the Router never reached Fit: {self.asked}")

    def examine(self, option: str, q: dict[str, Any]) -> dict[str, Any]:
        """An alternative examined before choosing: its preview, which records nothing."""
        before = len(q["decisions"])
        p = self.cmd("preview", option)
        assert set(p) == {"previewed", "preview"}, p
        offered = next(o for o in q["options"] if o["id"] == option)
        assert p["previewed"] == offered["decision"]
        assert p["preview"]["kind"] == offered["decision"]["kind"] and isinstance(p["preview"]["views"], list)
        assert len(self.cmd("ask")["decisions"]) == before  # nothing recorded
        return p


def http(port: int, path: str) -> tuple[int, Any]:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api{path}", timeout=60) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, None


def listening(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(1.0)
        return s.connect_ex(("127.0.0.1", port)) == 0


def start(analyst: Analyst, budget_fits: int) -> dict[str, Any]:
    out = analyst.drv("start", "--data", str(FIXTURE), "--provenance", PROVENANCE,
                      "--budget-fits", str(budget_fits), "--budget-seconds", "900",
                      "--run", analyst.run, "--scripted")
    meta = out["started"]
    assert meta["analyst"] == "scripted"
    folder = analyst.runs / analyst.run
    assert json.loads((folder / "driver.json").read_text()) == meta
    assert {"run", "pid", "port", "project", "budget", "started", "ended", "data",
            "provenance"} <= set(meta)
    assert meta["run"] == analyst.run and meta["ended"] is None
    assert meta["budget"] == {"fits": budget_fits, "seconds": 900.0}
    assert meta["data"]["sha256"] == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert meta["provenance"] == {"url": "https://example.org/data/clinical_risk.csv",
                                  "release_date": "2026-08-02", "license": "CC0-1.0"}
    assert (folder / "home").is_dir() and str(folder / "home") == meta["home"]
    command = driver.command_of(meta["pid"])
    assert "turbotab.server" in command and "--analyst" in command and f"--port {meta['port']}" in command
    # analyst mode: the analyst log is served (it is refused, 403, outside analyst mode)
    assert http(meta["port"], f"/projects/{meta['project']}/analyst-log")[0] == 200
    return meta


def stop(analyst: Analyst, meta: dict[str, Any]) -> None:
    workers = driver.descendants(meta["pid"])
    assert workers, "the server started its job workers"
    out = analyst.drv("stop", "--run", analyst.run)
    assert out["stopped"]["pid"] == meta["pid"] and out["stopped"]["running"] == []
    for pid in [meta["pid"], *workers]:
        assert not driver.alive(pid), pid
    assert driver.descendants(meta["pid"]) == []
    assert not listening(meta["port"])
    ended = json.loads((analyst.runs / analyst.run / "driver.json").read_text())["ended"]
    assert ended is not None
    again = analyst.drv("ask", "--run", analyst.run, expect=1)
    assert again["refused"]["code"] == "run_stopped"


def notes_of(meta: dict[str, Any]) -> list[str]:
    status, view = http(meta["port"], f"/projects/{meta['project']}")
    assert status == 200
    return [r.get("note") for r in view["decisions"]]


@pytest.mark.slow
def test_a_run_from_start_to_export_through_the_driver(tmp_path):
    """One run, start to stop, every command through ``python -m research.swarm.driver``:

    * ``start`` records the server's PID and port, the project, the budget, the file's SHA-256 and
      where it came from, in analyst mode;
    * ``ask`` serves each open question with options by the server's ids; ``preview`` records
      nothing; each ``answer``'s reason is its own record's note in the decision log;
    * ``fit`` presses Fit within the budget, read back from ``GET /fits``; ``results`` serves the
      cross-validated scores;
    * ``seal`` opens the held-out rows once (after the triage the server asks for, taken by its
      way forward's id) and a second ``seal`` is refused by the server; after the opening the
      driver refuses any further attempt;
    * ``export`` saves the bundle under the run, its provenance echoing the file's source;
    * ``stop`` leaves no process and no listener behind.
    """
    analyst = Analyst(tmp_path, "full")
    meta = start(analyst, budget_fits=2)
    try:
        q = analyst.until_fit()
        assert q["question"] == {"key": "open_seal", "status": "waiting", "waiting_on": ["fit"]}
        assert analyst.asked == list(PREDICT)  # every planned question, in the Router's order
        # every reason given is the note on its own record, in order
        assert [n for n in notes_of(meta) if n is not None] == analyst.reasons
        # the preview of complete cases was logged by the server, and recorded nothing
        status, log = http(meta["port"], f"/projects/{meta['project']}/analyst-log")
        assert status == 200
        assert [p["decision"]["strategy"] for p in log["previews"]
                if (p.get("decision") or {}).get("kind") == "set_missing"] == ["complete_case"]

        fit = analyst.cmd("fit")
        assert set(fit) == {"pressed", "fit", "spent"}
        assert fit["pressed"]["pressed"] is True and fit["fit"]["status"] == "fresh"
        _, fits = http(meta["port"], f"/projects/{meta['project']}/fits")
        assert fit["spent"] == {"fits": len(fits["records"]), "running": 0, "budget_fits": 2,
                                "budget_seconds": 900.0,
                                "seconds": round(sum(r["seconds"] or 0 for r in fits["records"]), 3)}
        assert 1 <= fit["spent"]["fits"] <= 2

        results = analyst.cmd("results")
        assert set(results) >= {"results", "stages"}
        res = results["results"]
        assert res["purpose"] == "prediction" and res["holdout_sealed"] is True
        assert [f["family"] for f in res["families"]] == ["linear"]
        assert 0.5 < res["families"][0]["cv"]["auc"]["mean"] < 1.0
        assert res["families"][0]["holdout"] is None  # sealed: no held-out score before the opening

        q = analyst.cmd("ask")
        assert q["question"]["key"] == "open_seal" and q["by"] == "seal"
        assert [o["id"] for o in q["options"]] == ["linear"]
        refused = analyst.cmd("answer", "linear", "--reason", "an opening is not an answer", expect=1)
        assert refused["refused"]["by"] == "driver" and refused["refused"]["code"] == "not_this_way"

        why = "Logistic regression is the one family fitted; its held-out score is the result."
        first = analyst.cmd("seal", "linear", "--reason", why, expect=None)
        for _ in range(4):  # the noticings' triage, which the server asks for first
            if "recorded" in first:
                break
            assert first["refused"]["by"] == "server" and first["refused"]["code"] == "triage_open", first
            way = next(e for e in first["refused"]["exits"]
                       if (e.get("decision") or {}).get("kind") == "confirm_sweep")
            analyst.answer([way["id"]], why="The noticings are triaged before any held-out score.")
            first = analyst.cmd("seal", "linear", "--reason", why, expect=None)
        opening = first["recorded"]
        assert opening["note"] == why
        assert opening["decision"]["kind"] == "open_seal" and opening["decision"]["family"] == "linear"
        assert opening["decision"]["n_holdout"] == 96  # 20% of 480, stratified

        second = analyst.cmd("seal", "linear", "--reason", "a second look", expect=1)
        assert second["refused"]["by"] == "server" and second["refused"]["code"] == "seal_already_open"
        for args in (["fit"], ["answer", "impute", "--reason", "after the seal"],
                     ["preview", "complete_case"]):
            after = analyst.cmd(*args, expect=1)
            assert after["refused"]["by"] == "driver" and after["refused"]["code"] == "after_seal", args

        results = analyst.cmd("results")["results"]
        assert results["at_opening"] is not None
        assert results["at_opening"]["scores"]["linear"] == opening["decision"]["scores"]["linear"]

        exported = analyst.cmd("export")["exported"]
        bundle = analyst.runs / analyst.run / "export.zip"
        assert Path(exported["path"]) == bundle
        assert exported["sha256"] == hashlib.sha256(bundle.read_bytes()).hexdigest()
        with zipfile.ZipFile(io.BytesIO(bundle.read_bytes())) as z:
            names = [n.split("/", 1)[1] for n in z.namelist() if not n.endswith("/")]
            provenance = json.loads(z.read(next(n for n in z.namelist() if n.endswith("/provenance.json"))))
            methods = z.read(next(n for n in z.namelist() if n.endswith("/methods.md"))).decode()
        assert sorted(exported["files"]) == sorted(names)
        assert "methods.md" in names and exported["methods"] == methods
        assert "https://example.org/data/clinical_risk.csv" in json.dumps(provenance)
        for reason in analyst.reasons:  # a reason is the record's, never the methods'
            assert reason not in methods

        logged = [json.loads(line) for line in
                  (analyst.runs / analyst.run / "commands.jsonl").read_text().splitlines()]
        assert [e["command"] for e in logged][0] == "start"
        assert {"ask", "preview", "answer", "fit", "results", "seal", "export"} <= {e["command"] for e in logged}
        assert all({"at", "command", "args", "summary"} <= set(e) for e in logged)
    finally:
        stop(analyst, meta)


TARGET = "readmit_30d"  # the fixture's outcome (clinical_risk.md)


def numeric_columns() -> list[str]:
    """The fixture's columns whose every recorded value reads as a number, read here with the csv
    module, the outcome left out: the columns a range can be kept on."""
    import csv

    with FIXTURE.open(newline="") as fh:
        rows = list(csv.DictReader(fh))

    def number(text: str) -> bool:
        try:
            float(text)
        except ValueError:
            return False
        return True

    return [c for c in rows[0] if c != TARGET and all(number(r[c]) for r in rows if r[c] != "")]


# The app's own range on age, 18 to 95 kept, as its card records it (ExclusionsAsk,
# turbotab/frontend/src/components/record/ask/ChoiceQuestions.tsx: by none, the card's reason, a
# row with no age not confirmed eligible).
AGE_RANGE = {"kind": "range", "column": "age", "low": 18.0, "high": 95.0, "by": None,
             "reason": "implausible values (your own range)", "missing": "exclude"}


def models_of(analyst: Analyst) -> str:
    return next(d["id"] for d in reversed(analyst.cmd("ask")["decisions"])
                if d["kind"] == "select_models")


def records_of(meta: dict[str, Any]) -> list[dict[str, Any]]:
    status, fits = http(meta["port"], f"/projects/{meta['project']}/fits")
    assert status == 200
    return fits["records"]


@pytest.mark.slow
def test_the_budget_bounds_the_fits_the_analyst_gets(tmp_path):
    """A run budgeted two fits, on the fixture with its own range of ages kept:

    * the exclusions question offers keeping every row in the teaching entry's words and the card's
      own range on a numeric column; a range with no bound, or a bound that is not a number, is
      refused, and one on age records the rule the app's card records;
    * within the budget a revisit and a new model make the second fit, and ``results`` serves its
      cross-validated scores, the same as its record in ``GET /fits``;
    * past it a third model is refused before anything is recorded, and ``/fits`` does not grow;
      going back to the first model, whose fit was made, is recorded and serves that fit's scores;
      a press for it, the triage and the seal still pass.
    """
    def exclusions_card(q: dict[str, Any]) -> None:
        assert [o["id"] for o in q["options"]] == ["none", "custom"]
        _, taught = http(meta["port"], "/teaching?purpose=prediction")
        entry = {o["value"]: o for o in next(e for e in taught if e["key"] == "exclusions")["options"]}
        none, custom = q["options"]
        assert (none["label"], none["consequence"]) == (entry["none"]["label"],
                                                         entry["none"]["consequence"])
        assert none["decision"] == {"kind": "set_exclusions", "rules": []}
        assert (custom["label"], custom["consequence"], custom["decision"]) == (
            entry["custom"]["label"], entry["custom"]["consequence"], None)
        fields = {f["name"]: f for f in q["fields"]}
        assert [c["value"] for c in fields["column"]["choices"]] == numeric_columns()
        for sets, word in ([["column=age"]], "low"), ([["column=age", "low=eighteen"]], "low"), \
                ([["column=age", "high=inf"]], "high"), ([["column=sex", "low=1"]], "column"), \
                ([["low=18"]], "column"):
            refused = analyst.cmd("answer", "custom", *[a for s in sets[0] for a in ("--set", s)],
                                  "--reason", "a range", expect=1)["refused"]
            assert refused["by"] == "driver" and refused["code"] == "not_offered", (sets, refused)
            assert word in refused["message"], (sets, refused)
        assert len(analyst.cmd("ask")["decisions"]) == len(q["decisions"])  # nothing recorded

    plan = {**PREDICT, "exclusions": [(["custom"], ["column=age", "low=18", "high=95"])]}
    analyst = Analyst(tmp_path, "budget", plan, before={"exclusions": exclusions_card})
    meta = start(analyst, budget_fits=2)
    try:
        analyst.until_fit()
        status, view = http(meta["port"], f"/projects/{meta['project']}")
        assert view["state"]["exclusions"] == [AGE_RANGE]

        fit = analyst.cmd("fit")
        assert fit["spent"]["fits"] == 1 and fit["spent"]["budget_fits"] == 2
        (first,) = records_of(meta)
        linear = analyst.cmd("results")["results"]["families"][0]
        assert linear["family"] == "linear" and linear["cv"] == first["scores"]["linear"]

        # within the budget: a second model makes the second fit, and its scores are served
        revisit = analyst.cmd("revisit", models_of(analyst), "--reason", "Ridge may calibrate better.")
        assert revisit["recorded"]["note"] == "Ridge may calibrate better."
        assert revisit["next"]["key"] == "models"
        analyst.answer(["ridge"], why="Ridge may calibrate better.")
        served = analyst.cmd("results", "--wait", "120")["results"]
        records = records_of(meta)
        assert [r["families"] for r in records] == [["linear"], ["ridge"]]
        assert [f["family"] for f in served["families"]] == ["ridge"]
        assert served["families"][0]["cv"] == records[1]["scores"]["ridge"]
        assert served["fit"]["key"] == records[1]["fit_key"]

        # past it: a third model would make a third fit, and is refused before it is recorded
        analyst.cmd("revisit", models_of(analyst), "--reason", "Elastic net may do better still.")
        before = analyst.cmd("ask")
        refused = analyst.cmd("answer", "elastic_net", "--reason", "Elastic net may do better.",
                              expect=1)["refused"]
        assert refused["by"] == "driver" and refused["code"] == "fit_budget", refused
        assert refused["message"].startswith("Not recorded: this would make a new fit, and the "
                                             "run's fit budget is 2 fits, and 2 were made already.")
        assert refused["spent"] == {"fits": 2, "running": 0, "budget_fits": 2,
                                    "budget_seconds": 900.0,
                                    "seconds": round(sum(r["seconds"] or 0 for r in records), 3)}
        after = analyst.cmd("ask")
        assert after["decisions"] == before["decisions"] and after["question"]["key"] == "models"
        time.sleep(2.0)
        assert len(records_of(meta)) == 2
        # going back to a fit already made is recorded, and serves that fit
        analyst.answer(["linear"], why="Back to plain logistic, whose fit was made first.")
        again = analyst.cmd("results", "--wait", "120")["results"]
        assert [f["family"] for f in again["families"]] == ["linear"]
        assert again["families"][0]["cv"] == first["scores"]["linear"]
        assert again["fit"]["key"] == first["fit_key"]
        pressed = analyst.cmd("fit")
        assert pressed["spent"]["fits"] == 2
        unknown = analyst.cmd("revisit", "no-such-id", "--reason", "x", expect=1)["refused"]
        assert unknown["code"] == "no_such_decision"
        # the triage and the seal still pass, past the budget
        why = "Plain logistic and ridge scored alike; the simpler model is final."
        sealed = analyst.cmd("seal", "linear", "--reason", why, expect=None)
        for _ in range(4):
            if "recorded" in sealed:
                break
            assert sealed["refused"]["code"] == "triage_open", sealed
            way = next(e for e in sealed["refused"]["exits"]
                       if (e.get("decision") or {}).get("kind") == "confirm_sweep")
            analyst.answer([way["id"]], why="The noticings are triaged before any held-out score.")
            sealed = analyst.cmd("seal", "linear", "--reason", why, expect=None)
        assert sealed["recorded"]["decision"]["kind"] == "open_seal"
        assert len(records_of(meta)) == 2
    finally:
        stop(analyst, meta)


@pytest.mark.slow
def test_an_estimate_run_through_the_driver(tmp_path):
    """Under Estimate the driver's composed answers for the grouping, the exposure and its measure,
    the adjustment set (a group's one tap, then answers per column) and the forms are each
    recorded as posted; ``fit`` locks the plan; ``results`` serves the estimate and, after the
    lock, which decisions mattered, each the server's own document; past the budget of one fit a
    secondary analysis that would refit is refused; the run exports; and ``seal`` is refused by
    the server, nothing being held out."""
    analyst = Analyst(tmp_path, "estimate", ESTIMATE)
    meta = start(analyst, budget_fits=1)
    try:
        q = analyst.until_fit()
        assert q["question"] is None and q["fit_reached"] is True
        assert analyst.asked == [k for k, answers in ESTIMATE.items() for _ in answers]
        status, view = http(meta["port"], f"/projects/{meta['project']}")
        state = view["state"]
        assert state["estimand"]["exposure"] == "charlson_index"
        assert state["estimand"]["measure"] == "odds_ratio"
        assert set(state["adjustment"]) == {"age", "sex", *LABS}
        assert state["adjustment"]["length_of_stay_days"] == {
            **state["adjustment"]["length_of_stay_days"], "exposure": "charlson_index",
            "causes_exposure": "no", "causes_outcome": "yes", "after_exposure": "yes"}
        assert state["exposure_forms"]["age"]["form"] == "linear"
        assert [n for n in (r.get("note") for r in view["decisions"]) if n] == analyst.reasons

        fit = analyst.cmd("fit")
        assert fit["pressed"]["locked"] is True and fit["fit"]["status"] == "fresh"
        out = analyst.cmd("results", "--wait", "240")
        assert out["stages"]["mattered"] == "fresh", out["stages"]
        _, served = http(meta["port"], f"/projects/{meta['project']}/results")
        assert out["results"] == served
        _, mattered = http(meta["port"], f"/projects/{meta['project']}/mattered")
        assert out["mattered"] == mattered and mattered["curve"] is not None
        (effect,) = [f["effect"] for f in out["results"]["families"] if f["family"] == "linear"]
        assert effect[0]["feature"] == "charlson_index"

        # one fit budgeted, and made: after the lock a secondary analysis that would refit is
        # refused, told by the engine's keys as the server serves them (not "cannot tell"), while
        # a revisit that leaves no fit, and going back to the locked plan's model, are recorded
        made = records_of(meta)
        assert len(made) == 1
        analyst.cmd("revisit", models_of(analyst), "--reason", "A second family as a secondary analysis.")
        offered = [o["id"] for o in analyst.cmd("ask")["options"]]
        other = next(o for o in offered if o != "linear")
        refused = analyst.cmd("answer", other, "--reason", "A second family.", expect=1)["refused"]
        assert refused["code"] == "fit_budget", refused
        analyst.answer(["linear"], why="Back to the locked plan's model.")
        assert records_of(meta) == made

        sealed = analyst.cmd("seal", "linear", "--reason", "Nothing is held out here.", expect=1)
        assert sealed["refused"]["by"] == "server" and sealed["refused"]["code"] == "nothing_sealed"
        exported = analyst.cmd("export")["exported"]
        assert "results/table2.csv" in exported["files"]
    finally:
        stop(analyst, meta)

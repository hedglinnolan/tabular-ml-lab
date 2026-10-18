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
          keys: list[str] | None = None) -> dict[str, Any]:
    return {"fits": fits, "seconds": seconds, "budget_fits": budget_fits,
            "budget_seconds": budget_seconds, "keys": keys or [f"k{i}" for i in range(fits)]}


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
    "exclusions": [(["keep_every_row"], [])],
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
    "exclusions": [(["keep_every_row"], [])],
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

    def __init__(self, runs: Path, run_id: str, plan: Plan | None = None):
        self.runs, self.run = runs, run_id
        self.plan = {k: list(v) for k, v in (plan or PREDICT).items()}
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
        assert fit["spent"] == {"fits": len(fits["records"]), "budget_fits": 2, "budget_seconds": 900.0,
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


@pytest.mark.slow
def test_a_fit_past_the_budget_is_refused(tmp_path):
    """A run budgeted one fit: the first press passes; after a revisit changes the models, a press
    that would make a second fit is refused by the driver, with what was spent."""
    analyst = Analyst(tmp_path, "budget")
    meta = start(analyst, budget_fits=1)
    try:
        analyst.until_fit()
        fit = analyst.cmd("fit")
        assert fit["spent"]["fits"] == 1 and fit["spent"]["budget_fits"] == 1
        q = analyst.cmd("ask")
        models = next(d for d in q["decisions"] if d["kind"] == "select_models")
        revisit = analyst.cmd("revisit", models["id"], "--reason", "Ridge may calibrate better.")
        assert revisit["recorded"]["decision"] == {"kind": "revert", "decision_id": models["id"]}
        assert revisit["recorded"]["note"] == "Ridge may calibrate better."
        assert revisit["next"]["key"] == "models"
        analyst.answer(["ridge"], why="Ridge may calibrate better.")
        refused = analyst.cmd("fit", expect=1)["refused"]
        assert refused["by"] == "driver" and refused["code"] == "fit_budget"
        assert refused["message"].startswith("Fit is not pressed: the run's fit budget is 1 fit,")
        _, fits = http(meta["port"], f"/projects/{meta['project']}/fits")
        assert refused["spent"]["fits"] == len(fits["records"])
        unknown = analyst.cmd("revisit", "no-such-id", "--reason", "x", expect=1)["refused"]
        assert unknown["code"] == "no_such_decision"
    finally:
        stop(analyst, meta)


@pytest.mark.slow
def test_an_estimate_run_through_the_driver(tmp_path):
    """Under Estimate the driver's composed answers for the grouping, the exposure and its measure,
    the adjustment set (a group's one tap, then answers per column) and the forms are each
    recorded as posted; ``fit`` locks the plan; ``results`` serves the estimate and, after the
    lock, which decisions mattered, each the server's own document; the run exports; and ``seal``
    is refused by the server, nothing being held out."""
    analyst = Analyst(tmp_path, "estimate", ESTIMATE)
    meta = start(analyst, budget_fits=2)
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

        sealed = analyst.cmd("seal", "linear", "--reason", "Nothing is held out here.", expect=1)
        assert sealed["refused"]["by"] == "server" and sealed["refused"]["code"] == "nothing_sealed"
        exported = analyst.cmd("export")["exported"]
        assert "results/table2.csv" in exported["files"]
    finally:
        stop(analyst, meta)

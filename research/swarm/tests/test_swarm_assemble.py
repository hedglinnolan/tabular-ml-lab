"""The record's assembler (DRIVER_PLAN D3), first on constructed inputs, then end to end.

The constructed tests take their truths from construction: a transcript whose counts are written
here, decisions and previews built by hand, a curve built in ``core/mattered.py``'s own models. The
end-to-end test drives ``clinical_risk.csv`` (its declared truth: ``truths.FIXTURE_TRUTHS``) in
an in-process server in analyst mode (``server_drive.local_server``), every decision posted with a
note, through two fits, a preview of each kind, the seal and a change after it, to the export. Its
record is compared with what each route serves, with the run's own notes, with hashlib over the
fixture's bytes and the bundle's, with pandas over the fixture, and with git for the engine's
commit; then every check runs on it and gives what the run's construction implies.
"""
from __future__ import annotations

import copy
import functools
import hashlib
import json
import os
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from research.swarm import assemble as A
from research.swarm import checks as C
from research.swarm.record import Claim, validate

# ── spend ────────────────────────────────────────────────────────────────────

# Two assistant messages; the first streamed over two lines that share its id, the later line with
# its final output count; a user line with no usage.
TRANSCRIPT = [
    {"type": "user", "message": {"role": "user", "content": "the card"}},
    {"type": "assistant", "message": {"id": "msg_a", "usage": {
        "input_tokens": 3, "cache_creation_input_tokens": 500, "cache_read_input_tokens": 40_000,
        "output_tokens": 16}}},
    {"type": "assistant", "message": {"id": "msg_a", "usage": {
        "input_tokens": 3, "cache_creation_input_tokens": 500, "cache_read_input_tokens": 40_000,
        "output_tokens": 200}}},
    {"type": "assistant", "message": {"id": "msg_b", "usage": {
        "input_tokens": 2, "cache_creation_input_tokens": 1_000, "cache_read_input_tokens": 45_000,
        "output_tokens": 75}}},
]
FRESH = (3 + 500 + 200) + (2 + 1_000 + 75)
CACHED = 40_000 + 45_000


def transcript(path: Path, lines: list[dict[str, Any]] = TRANSCRIPT) -> Path:
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), "utf-8")
    return path


def test_spend_counts_each_message_once_at_its_final_counts(tmp_path):
    got = A.spend(transcript(tmp_path / "t.jsonl"))
    assert (got.fresh, got.cached, got.messages) == (FRESH, CACHED, 2)
    assert got.fresh != (3 + 500 + 16) + (2 + 1_000 + 75)  # not the streamed line's partial count


def test_spend_is_refused_when_a_line_cannot_be_counted_once(tmp_path):
    lines = TRANSCRIPT + [{"type": "assistant", "message": {"usage": {"output_tokens": 9}}}]
    with pytest.raises(A.SpendRefused, match="no message id"):
        A.spend(transcript(tmp_path / "t.jsonl", lines))


# ── what a decision answers, and what was examined before it ─────────────────


def test_each_decision_answers_its_questions_keyed_as_a_fixtures_truth_keys_them():
    assert A.answers_of({"kind": "confirm_readings", "items": [
        {"reading": "code_or_count", "column": "age", "value": "amount"},
        {"reading": "role", "column": "id", "value": "identifier"}]}) == [
        ("code_or_count:age", "amount"), ("role:id", "identifier")]
    assert A.answers_of({"kind": "set_column_unit", "column": "kcal", "unit": "kcal",
                         "days": 1}) == [("unit:kcal", "kcal"), ("day_count:kcal", "1")]
    assert A.answers_of({"kind": "set_roles", "roles": {"a": "exposure"}, "unconfirmed": []}) == [
        ("role:a", "exposure")]
    assert A.answers_of({"kind": "apply_repair", "finding_id": "below_detection__crp",
                         "option": "half_limit", "params": {}}) == [
        ("detection_limit:crp", "half_limit")]
    assert A.answers_of({"kind": "confirm_sweep", "stage": "fit", "sweep": "noticings",
                         "lines": None}) == [("confirm_sweep:fit:noticings", {"lines": None})]
    assert A.answers_of({"kind": "select_models", "models": ["linear"]}) == [
        ("select_models", {"models": ["linear"]})]


def test_a_field_the_server_fills_is_no_difference_and_any_other_field_is():
    asked = {"family": "linear", "target": None, "scores": None}
    assert A.same_option(asked, {"family": "linear", "target": "y", "scores": {"linear": {}}})
    assert not A.same_option(asked, {"family": "elastic_net", "target": "y", "scores": None})
    assert not A.same_option({"models": ["linear"]}, {"models": None})
    assert A.same_option("exposure", "exposure") and not A.same_option("exposure", "excluded")


def _rec(seq, minute, decision, note="why"):
    return {"id": f"r{seq}", "seq": seq, "at": f"2026-10-18T10:{minute:02d}:00Z", "note": note,
            "post_seal": False, "after_estimates": False, "recorded_by": "you",
            "decision": decision}


def _preview(minute, decision, refused=None):
    return {"at": f"2026-10-18T10:{minute:02d}:00Z", "decision": decision, "refused": refused}


ROLE_A = {"kind": "confirm_role", "column": "a", "role": "excluded"}
RECORDS = [_rec(1, 1, {"kind": "set_roles", "roles": {"a": "exposure", "b": "covariate"}}),
           _rec(2, 2, {"kind": "confirm_role", "column": "b", "role": "identifier"}),
           _rec(3, 3, {"kind": "confirm_role", "column": "b", "role": "covariate"}),
           _rec(4, 9, ROLE_A)]
PREVIEWS = [_preview(4, ROLE_A), _preview(5, {**ROLE_A, "role": "covariate"}, "reading_unsettled"),
            _preview(6, {**ROLE_A, "role": "flag"}), _preview(10, {**ROLE_A, "role": "time"})]
FITS = [{"id": "fit1", "seq": 1, "at": "2026-10-18T10:05:00Z", "decisions": [1, 3]}]


def test_an_option_is_examined_when_previewed_or_fitted_before_the_decision_or_while_it_stood():
    got = A.examined_of(RECORDS[3], RECORDS, PREVIEWS, FITS)
    assert [(e["how"], e["value"], e["alternative"]) for e in got] == [
        ("preview", "excluded", False),  # the option chosen, previewed: not an alternative
        ("preview", "covariate", False),  # refused, so nothing was previewed
        ("preview", "flag", True),
        ("preview", "time", True),  # at 10:10, while the answer stood: previewed, then kept
        ("fit", "exposure", True),  # record #1's answer, in force in fit 1
    ]  # record #2 was never in force in a fit
    assert got[-1]["seq"] == 1 and got[-1]["attempt"] == "fit1"
    # Record #1 answered role:a first, and its window closes at record #4 (10:09): the previews at
    # 10:04–10:06 examined it too, while it stood.
    first = A.examined_of(RECORDS[0], RECORDS, PREVIEWS, FITS)
    assert [(e["value"], e["alternative"]) for e in first] == [
        ("excluded", True), ("covariate", False), ("flag", True)]


def test_a_preview_after_the_questions_next_answer_examines_that_answer_not_this_one():
    later = _rec(5, 12, {**ROLE_A, "role": "flag"})
    got = A.examined_of(RECORDS[3], RECORDS + [later], PREVIEWS, FITS)
    assert ("preview", "time") in [(e["how"], e["value"]) for e in got]  # 10:10, before #5
    after = _preview(13, {**ROLE_A, "role": "identifier"})
    got = A.examined_of(RECORDS[3], RECORDS + [later], PREVIEWS + [after], FITS)
    assert "identifier" not in [e["value"] for e in got]


def test_an_option_examined_the_same_way_twice_is_listed_once():
    block = _rec(2, 2, {"kind": "confirm_readings", "items": [
        {"reading": "role", "column": "a", "value": "exposure"}]})
    records = [RECORDS[0], block, _rec(3, 9, ROLE_A)]
    fits = [{"id": "fit1", "seq": 1, "at": "2026-10-18T10:05:00Z", "decisions": [1, 2]}]
    previews = [_preview(4, {**ROLE_A, "role": "flag"}), _preview(6, {**ROLE_A, "role": "flag"})]
    got = A.examined_of(records[2], records, previews, fits)
    assert [(e["how"], e["value"], e.get("seq")) for e in got] == [
        ("preview", "flag", None), ("fit", "exposure", 1)]  # #2 held the same option in fit 1


# ── materiality ──────────────────────────────────────────────────────────────

MOVED = [{"family": "linear", "against": "fit1", "movement": {"value": 0.77}, "why": None}]


def _fits():
    return [{"id": "fit1", "seq": 1, "decisions": [1, 2], "changed": [1, 2], "movement": [],
             "movement_why": "No earlier fit is recorded to compare this one with.",
             "withheld": None},
            {"id": "fit2", "seq": 2, "decisions": [1, 2, 3], "changed": [3], "movement": MOVED,
             "movement_why": None, "withheld": None},
            {"id": "fit3", "seq": 3, "decisions": [1, 2, 4, 5], "changed": [4, 5],
             "movement": MOVED, "movement_why": None, "withheld": None}]


def test_a_decision_has_the_fits_movement_only_where_it_alone_changed():
    fits = _fits()
    assert A.fits_materiality({"seq": 3}, fits) == {"source": "fits", "attempt": "fit2",
                                                    "movement": MOVED}
    together = A.fits_materiality({"seq": 4}, fits)
    assert together["source"] is None and together["attempt"] == "fit3"
    assert together["why"] == ("It took effect at fit #3 together with record #5: the fit "
                               "records measure the change between two fits, not each record's "
                               "share of it.")
    assert A.fits_materiality({"seq": 1}, fits)["why"] == fits[0]["movement_why"]
    assert A.fits_materiality({"seq": 6}, fits) == {"source": None, "why": A.NOT_IN_A_FIT}
    quiet = copy.deepcopy(fits)
    quiet[1].update(movement=[], withheld="Fit was not pressed.")
    assert A.fits_materiality({"seq": 3}, quiet)["why"] == "Fit was not pressed."
    swapped = copy.deepcopy(fits)
    swapped[1]["movement"] = [{"family": "elastic_net", "against": "fit1", "movement": None,
                               "why": "`elastic_net` was not fitted in the previous fit."}]
    assert A.fits_materiality({"seq": 3}, swapped) == {
        "source": None, "attempt": "fit2",
        "why": "`elastic_net`: `elastic_net` was not fitted in the previous fit."}


def _curve_doc() -> dict[str, Any]:
    from turbotab.core import mattered as M

    def choice(key, label, decision):
        return M.MatteredChoice(key=key, label=label, primary="recorded", decision=decision,
                                options=[M.MatteredOption(key="recorded", label="As recorded"),
                                         M.MatteredOption(key="other", label="Changed")])

    recorded = {"adjusted:age": "recorded", "exclusion:0": "recorded"}
    curve = M.MatteredCurve(
        estimate_label="β", choices=[choice("adjusted:age", "`age` adjusted", "set_adjustment"),
                                     choice("exclusion:0", "The rule", "set_exclusions")],
        specs=[M.MatteredSpec(key="s0", picks=recorded, primary=True, estimate=0.3, low=0.1,
                              high=0.5, n=100),
               M.MatteredSpec(key="s1", picks={**recorded, "adjusted:age": "other"},
                              primary=False, estimate=0.4, low=0.2, high=0.6, n=100)],
        not_estimated=[M.MatteredNotEstimated(key="s2", picks={**recorded, "exclusion:0": "other"},
                                              primary=False, reason="A reading waits.")],
        total=3, drawn=3, sampled=False, rule=None, crossing="one_at_a_time", summary=None,
        level=0.95, sealed=None, plan_lock="L", weighted=False, clustered=False, exits=[])
    return M.Mattered(status="fresh", fresh=True, key="k", plan_lock="L", curve=curve, says=None,
                      methods=None, reported=None, examined=[], error=None).model_dump(mode="json")


def test_under_estimate_a_choice_the_curve_varies_has_its_departures_beside_the_primary():
    doc = _curve_doc()
    served = A.Read(200, doc)
    adjust = _rec(5, 5, {"kind": "set_adjustment", "exposure": "x", "answers": {}})
    rules = _rec(6, 6, {"kind": "set_exclusions", "rules": []})
    earlier = _rec(4, 4, {"kind": "set_adjustment", "exposure": "x", "answers": {"age": 1}})
    in_force = A.in_force_of([earlier, adjust, rules])
    got = A.mattered_materiality(adjust, served, in_force)
    assert got == {"source": "mattered", "primary": doc["curve"]["specs"][0], "departures": [
        {"choice": "adjusted:age", "label": "`age` adjusted", "option": "other",
         "spec": doc["curve"]["specs"][1], "not_estimated": None}]}
    # The rule's one departure was not estimated: no number, so its why is the curve's reason.
    assert A.mattered_materiality(rules, served, in_force) == {
        "source": None, "why": "The curve estimated no specification that changes this "
                               "decision's choices: `The rule` Changed: A reading waits."}
    assert A.mattered_materiality(earlier, served, in_force)["why"] == A._superseded(5)
    assert A.mattered_materiality(_rec(7, 7, {"kind": "set_purpose"}), served, in_force) is None
    refused = A.Read(409, {"error": {"code": "curve_sealed", "message": "Opens after the lock.",
                                     "exits": []}})
    assert A.mattered_materiality(adjust, refused, in_force) == {"source": None,
                                                                 "why": "Opens after the lock."}


# ── the result ───────────────────────────────────────────────────────────────

SCORES = {"linear": {"auc": 0.65, "log_loss": 0.54}}


def test_under_predict_the_result_is_the_openings_scores_once():
    records = [_rec(1, 1, {"kind": "select_models", "models": ["linear"]}),
               _rec(2, 3, {"kind": "open_seal", "family": "linear", "target": "y",
                           "metric": "log_loss", "n_holdout": 96, "scores": SCORES}),
               _rec(3, 4, {"kind": "select_models", "models": ["linear", "elastic_net"]})]
    opened = {"seq": 2, "current": True, "note": "the reported result"}
    fits = [{"id": "fit1", "target": "y", "changed_after_seal": False, "at_opening": opened,
             "plan_digest": "d1", "decisions": [1]},
            {"id": "fit2", "target": "y", "changed_after_seal": True, "at_opening": None,
             "plan_digest": "d2", "decisions": [1, 2, 3]}]
    intervals = {"auc": {"ci_low": 0.52, "ci_high": 0.78}}
    doc = {"at_opening": opened, "families": [{"family": "linear", "intervals": intervals}]}
    got = A.result_of("prediction", records, fits, A.Read(200, doc))
    assert (got["attempt"], got["decisions"], got["scores"], got["intervals"]) == (
        "fit1", [1], SCORES, intervals)
    later = {**doc, "at_opening": {**opened, "current": False, "note": "changed after it"}}
    got = A.result_of("prediction", records, fits, A.Read(200, later))
    assert got["intervals"] is None and got["intervals_why"] == "changed after it"
    assert A.result_of("prediction", records[:1], fits, A.Read(200, doc))["refused"]


def test_under_estimate_the_result_is_the_fit_the_lock_in_force_was_recorded_on():
    effect = [{"feature": "x", "estimate": 0.3, "ci_low": 0.1, "ci_high": 0.5}]
    records = [_rec(1, 1, {"kind": "set_estimand"}), _rec(2, 2, {"kind": "set_adjustment"}),
               _rec(3, 3, {"kind": "lock_plan", "digest": "lockdigest"}),
               _rec(4, 6, {"kind": "set_exclusions", "rules": []})]
    fits = [{"id": "fit1", "decisions": [1], "effects": {}},
            {"id": "fit2", "decisions": [1, 2, 3], "effects": {"linear": effect}},
            {"id": "fit3", "decisions": [1, 2, 3, 4], "effects": {"linear": effect}}]
    got = A.result_of("inference", records, fits, A.Read(200, {}))
    assert (got["attempt"], got["lock"], got["plan_digest"], got["effects"]) == (
        "fit2", 3, "lockdigest", {"linear": effect})
    assert A.result_of("inference", records[:2], fits, A.Read(200, {}))["refused"]


def test_a_lock_the_server_withdrew_is_not_the_lock_in_force():
    effect = [{"feature": "x", "estimate": 0.3, "ci_low": 0.1, "ci_high": 0.5}]
    records = [_rec(1, 1, {"kind": "set_estimand"}),
               _rec(2, 2, {"kind": "lock_plan", "digest": "first"}),
               _rec(3, 3, {"kind": "revert", "decision_id": "r2"}),
               _rec(4, 4, {"kind": "set_adjustment"}),
               _rec(5, 5, {"kind": "lock_plan", "digest": "second"})]
    fits = [{"id": "fit1", "decisions": [1, 4, 5], "effects": {"linear": effect}}]
    got = A.result_of("inference", records, fits, A.Read(200, {}))
    assert (got["lock"], got["plan_digest"]) == (5, "second")
    withdrawn = A.result_of("inference", records[:4], fits, A.Read(200, {}))
    assert withdrawn["refused"] == "The plan was never locked, so no estimate was served."
    rules = _rec(6, 6, {"kind": "set_exclusions", "rules": []})
    undone = _rec(7, 7, {"kind": "revert", "decision_id": "r6"})
    assert A.mattered_materiality(rules, A.Read(200, _curve_doc()),
                                  A.in_force_of([rules, undone])) == {"source": None,
                                                                      "why": A.WITHDRAWN}


# ── reading the server ───────────────────────────────────────────────────────


class FakeResponse:
    def __init__(self, status: int, body: Any):
        self.status_code, self._body = status, body
        self.content = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.text = self.content.decode("utf-8", "replace")

    def json(self) -> Any:
        if isinstance(self._body, bytes):
            raise ValueError("not JSON")
        return self._body


class FakeClient:
    """Serves ``routes`` by path suffix and keeps the order of every GET."""

    def __init__(self, routes: dict[str, Any]):
        self.routes, self.asked = routes, []

    def get(self, url: str) -> FakeResponse:
        self.asked.append(url)
        for suffix, answer in self.routes.items():
            if url.endswith(suffix):
                answer = answer() if callable(answer) else answer
                return answer if isinstance(answer, FakeResponse) else FakeResponse(200, answer)
        return FakeResponse(404, {"error": {"code": "unknown", "message": url, "exits": []}})


def test_the_analyst_log_is_read_before_any_route_that_can_render_a_claim():
    states = iter(["running", "running", "fresh"])
    client = FakeClient({"/analyst-log": {"claims": [], "previews": []},
                         "/mattered": lambda: {"status": next(states)}})
    served = A.fetch(client, "p1", poll=0.0)
    assert client.asked[0] == "/api/projects/p1/analyst-log"
    assert client.asked.count("/api/projects/p1/mattered") == 3  # waited while it computed
    assert served.mattered.body == {"status": "fresh"}


def _run_dir(folder: Path, **driver: Any) -> Path:
    folder.mkdir(parents=True)
    (folder / A.DRIVER_FILE).write_text(json.dumps({"run_id": folder.name, "project_id": "p1",
                                                    "port": 1, **driver}), "utf-8")
    return folder


def test_a_server_that_serves_no_analyst_log_leaves_a_failed_run(tmp_path):
    refused = FakeResponse(403, {"error": {"code": "analyst_mode_only", "message": "not analyst",
                                           "exits": []}})
    client = FakeClient({"/analyst-log": refused, "/p1": {"decisions": [], "state": {}}})
    path, ok = A.assemble(_run_dir(tmp_path / "runs" / "r1"), client=client,
                          corpus=tmp_path / "corpus", mattered_wait=0)
    assert not ok and path == tmp_path / "corpus" / "failed" / "r1.json"
    assert "analyst_mode_only" in json.loads(path.read_text("utf-8"))["assembled"]["unreadable"]


# ── end to end ───────────────────────────────────────────────────────────────

FIXTURE = "clinical_risk.csv"
RUN_ID = "e2e-clinical-1"
SPLIT = {"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5}
ROLE = {"kind": "confirm_role", "column": "sodium_mmol_l", "role": "excluded"}
AFTER_SEAL = {"kind": "select_models", "models": ["linear", "elastic_net"]}


@contextmanager
def analyst_server(home: Path):
    """``server_drive.local_server`` in analyst mode, as the driver runs its server: the harness
    takes no analyst flag, so its ``Settings`` are given one. Claims render strictly, as in every
    core test (``core/tests/conftest.py``)."""
    from turbotab.core import config
    from turbotab.core.tests.acceptance.server_drive import local_server

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "Settings", functools.partial(config.Settings, analyst=True))
        mp.setenv("TURBOTAB_CLAIMS_STRICT", "1")
        with local_server(home) as client:
            yield client


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ScriptedDriver:
    """A scripted analyst's client: every decision posted with a note, kept by the seq it was
    recorded at, and each command logged as a line of ``commands.jsonl`` with what it returned."""

    def __init__(self, client: Any):
        self.c, self.commands, self.notes, self.n = client, [], {}, 0

    @property
    def app(self) -> Any:
        return self.c.app

    def get(self, url: str, **kw: Any) -> Any:
        r = self.c.get(url, **kw)
        for command in ("results", "export"):
            if url.endswith(f"/{command}"):
                out = r.json() if r.status_code != 200 or command != "export" else {
                    "sha256": hashlib.sha256(r.content).hexdigest()}
                self.commands.append({"command": command, "at": _now(), "output": out})
        return r

    def post(self, url: str, json: Any = None, **kw: Any) -> Any:  # noqa: A002
        if url.endswith("/decisions"):
            self.n += 1
            note = f"Reason {self.n}: {json.get('kind')} as the evidence on its card reads."
            kw.setdefault("params", {})["note"] = note
            r = self.c.post(url, json=json, **kw)
            self.commands.append({"command": "answer", "at": _now(), "output": r.json()})
            if r.status_code == 200:  # TurboTab may record its own answers in the same request
                mine = next(d for d in r.json()["decisions"] if d["note"] == note)
                self.notes[mine["seq"]] = note
            return r
        r = self.c.post(url, json=json, **kw)
        command = "preview" if url.endswith("/preview") else "fit" if url.endswith("/fit") else None
        if command:
            self.commands.append({"command": command, "at": _now(), "output": r.json()})
        return r


def _drive(driver: ScriptedDriver, pid: str) -> dict[str, Any]:
    from turbotab.core.tests.acceptance.server_drive import press_fit
    from turbotab.core.tests.truths import FIXTURE_TRUTHS, answer_refusal
    from turbotab.server.tests.conftest import answer_settled, declare, prepare, truth_of, wait_for

    declare(pid, FIXTURE_TRUTHS[FIXTURE], fixture=FIXTURE)
    wait_for(driver, pid, {"ingest": "fresh", "profile": "fresh"}, timeout=120)
    for body in ({"kind": "set_lens", "lenses": ["clinical"]},
                 {"kind": "set_target", "column": "readmit_30d"},
                 {"kind": "set_purpose", "purpose": "prediction"}, SPLIT):
        prepare(driver, pid, body)
        assert answer_settled(driver, pid, None, body).status_code == 200, body
    for body in (ROLE, {"kind": "set_missing", "strategy": "impute"}):
        assert driver.post(f"/api/projects/{pid}/preview", json=body).status_code == 200
    models = {"kind": "select_models", "models": ["linear"]}
    prepare(driver, pid, models)
    assert answer_settled(driver, pid, None, models).status_code == 200
    wait_for(driver, pid, {"fit": "fresh"}, timeout=600)
    assert press_fit(driver, pid)
    assert answer_settled(driver, pid, None, ROLE).status_code == 200
    wait_for(driver, pid, {"fit": "fresh"}, timeout=600)
    assert driver.get(f"/api/projects/{pid}/results").status_code == 200
    url = f"/api/projects/{pid}/decisions"
    opening = {"kind": "open_seal", "family": "linear"}
    opened = answer_refusal(lambda d: driver.post(url, json=d), driver.post(url, json=opening),
                            truth_of(pid), lambda: driver.post(url, json=opening))
    assert opened.status_code == 200, opened.text
    assert answer_settled(driver, pid, None, AFTER_SEAL).status_code == 200
    wait_for(driver, pid, {"fit": "fresh"}, timeout=600)
    resealed = driver.post(url, json={"kind": "reseal", "reason": "a better draw"})
    assert resealed.status_code == 409
    exported = driver.get(f"/api/projects/{pid}/export")
    assert exported.status_code == 200
    return {"reseal": resealed.json()["error"]}


def finish(run_dir: Path, driver: ScriptedDriver, pid: str, start: str,
           dataset: dict[str, Any]) -> None:
    """The run directory as the driver leaves it: its commands, the analyst's transcript (here the
    constructed one) and driver.json."""
    (run_dir / A.COMMANDS_FILE).write_text(
        "".join(json.dumps(c) + "\n" for c in driver.commands), "utf-8")
    transcript(run_dir / "transcript.jsonl")
    (run_dir / A.DRIVER_FILE).write_text(json.dumps({
        "run_id": run_dir.name, "pid": os.getpid(), "port": None, "project_id": pid,
        "fit_budget": {"count": 10, "seconds": 600}, "start": start, "end": _now(), "design": 2,
        "analyst_model": "scripted", "effort": None, "prompt_version": "e2e-1",
        "transcript": "transcript.jsonl", "dataset": dataset}), "utf-8")


@pytest.fixture(scope="module")
def e2e(tmp_path_factory) -> dict[str, Any]:
    """One scripted run to its record, and what the server serves once it is assembled."""
    folder = tmp_path_factory.mktemp("swarm_e2e")
    run_dir = folder / "runs" / RUN_ID
    start = _now()
    with analyst_server(run_dir / "home") as client:
        made = client.post("/api/projects", json={"path": str(SAMPLE)})
        assert made.status_code == 200, made.text
        pid = made.json()["id"]
        driver = ScriptedDriver(client)
        seen = _drive(driver, pid)
        finish(run_dir, driver, pid, start,
               {"id": "clinical-risk", "control": "known_answer", "fixture": FIXTURE})
        log_before = client.get(f"/api/projects/{pid}/analyst-log").json()
        began = time.monotonic()
        path, ok = A.assemble(run_dir, client=client, corpus=folder / "corpus")
        seconds = time.monotonic() - began
        base = f"/api/projects/{pid}"
        served = {name: client.get(f"{base}{route}").json() for name, route in (
            ("view", ""), ("fits", "/fits"), ("results", "/results"))}
        bundle = client.get(f"{base}/export").content
    return {"path": path, "ok": ok, "notes": driver.notes, "served": served, "bundle": bundle,
            "log_before": log_before, "seconds": seconds, "corpus": folder / "corpus", **seen}


SAMPLE = Path(__file__).resolve().parents[3] / "turbotab" / "sample_data" / FIXTURE


def test_the_run_leaves_one_record_that_validates(e2e):
    assert e2e["ok"], e2e["path"].read_text("utf-8")[:2000]
    assert e2e["path"] == e2e["corpus"] / f"{RUN_ID}.json"
    assert not (e2e["corpus"] / "failed").exists()
    rec = validate(json.loads(e2e["path"].read_text("utf-8")))
    assert rec.run.id == RUN_ID and rec.run.design == 2


@pytest.fixture(scope="module")
def rec(e2e):
    return validate(json.loads(e2e["path"].read_text("utf-8")))


def test_each_decision_is_the_logs_record_with_the_reason_the_driver_posted(rec, e2e):
    log = e2e["served"]["view"]["decisions"]
    assert [(d.id, d.seq, d.kind) for d in rec.decisions] == [
        (r["id"], r["seq"], r["decision"]["kind"]) for r in log]
    assert {d.seq: d.reason for d in rec.decisions} == e2e["notes"]
    assert all(d.source == "analyst" for d in rec.decisions)  # TurboTab recorded nothing itself
    after = [d for d in rec.decisions if d.post_seal]
    assert [(d.kind, d.phase, d.option["models"]) for d in after] == [
        ("select_models", "after_seal", AFTER_SEAL["models"])]


def test_each_attempt_is_the_fit_record_as_served(rec, e2e):
    served = e2e["served"]["fits"]["records"]
    assert len(rec.attempts) == len(served) == 3
    for attempt, fit in zip(rec.attempts, served):
        mine = attempt.model_dump(mode="json")
        for key in ("id", "fit_key", "plan_digest", "decisions", "families", "scores",
                    "movement", "movement_why", "holdout", "changed_after_seal", "at_opening"):
            assert mine[key] == fit[key], key
    role = next(d for d in rec.decisions if d.kind == "confirm_role")
    assert rec.attempts[1].changed == [role.seq]
    assert rec.run.fits == 3
    assert rec.run.fit_seconds == pytest.approx(sum(f["seconds"] for f in served))


def test_the_role_change_has_the_engines_movement_and_its_examined_options(rec, e2e):
    """Dropping sodium between fits 1 and 2 was the only change, the rows and folds unchanged
    (the fixture has no blanks), so its materiality is fit 2's served paired movement."""
    role = next(d for d in rec.decisions if d.kind == "confirm_role")
    fit2 = e2e["served"]["fits"]["records"][1]
    assert role.materiality.source == "fits" and role.materiality.attempt == fit2["id"]
    assert role.materiality.movement == fit2["movement"]
    assert fit2["movement"][0]["movement"]["value"] is not None
    examined = [(e.how, e.value, e.alternative) for e in role.examined
                if e.question == "role:sodium_mmol_l"]
    assert ("preview", "excluded", False) in examined  # the option chosen, previewed
    assert any(how == "fit" and value != "excluded" and alt for how, value, alt in examined)


def test_the_result_is_the_first_openings_scores_and_no_later_record_reached_it(rec, e2e):
    log = e2e["served"]["view"]["decisions"]
    opening = next(r for r in log if r["decision"]["kind"] == "open_seal")
    after = [r["seq"] for r in log if r["post_seal"]]
    r = rec.result
    assert (r.opening, r.family, r.scores) == (opening["seq"], "linear",
                                               opening["decision"]["scores"])
    opened = next(f for f in e2e["served"]["fits"]["records"]
                  if (f["at_opening"] or {}).get("seq") == opening["seq"]
                  and not f["changed_after_seal"])
    assert r.attempt == opened["id"] and r.decisions == opened["decisions"]
    assert after and not set(after) & set(r.decisions)
    # A change after the opening: the intervals served are the changed fit's, not the opening's.
    assert r.intervals is None
    assert r.intervals_why == e2e["served"]["results"]["at_opening"]["note"]


def test_the_run_dataset_regime_and_export_are_the_servers_and_the_files(rec, e2e):
    frame = pd.read_csv(SAMPLE)
    assert rec.dataset.files[0].sha256 == hashlib.sha256(SAMPLE.read_bytes()).hexdigest()
    assert rec.regime.n_rows == len(frame) and rec.regime.missing == {}
    assert rec.regime.events == int((frame["readmit_30d"] == 1).sum())
    assert rec.regime.n_analyzed == len(frame.dropna())
    assert rec.export.reached and rec.export.sha256 == hashlib.sha256(e2e["bundle"]).hexdigest()
    assert rec.export.methods.startswith("# Methods")
    assert rec.export.checklist["checklist"]
    head = subprocess.run(["git", "-C", str(SAMPLE.parent), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    assert rec.run.engine_commit == head
    assert (rec.run.spend.fresh, rec.run.spend.cached) == (FRESH, CACHED)
    # The claims are the analyst log as it stood when the run ended, before any read of assembly.
    assert rec.claims == [Claim.model_validate(c) for c in e2e["log_before"]["claims"]]


def test_the_reseal_the_server_refused_is_a_surfacing(rec, e2e):
    refusal = next(s for s in rec.surfacings.refusals if s.code == e2e["reseal"]["code"])
    assert refusal.command == "answer" and refusal.message == e2e["reseal"]["message"]
    assert refusal.then == "export"


def test_every_check_gives_what_the_runs_construction_implies(rec):
    got = {o.check: o for o in C.run_checks([rec])}
    assert got["seal_integrity"].outcome == "pass"
    assert got["logging"].outcome == "pass", got["logging"].problems
    assert got["single_pass"].outcome == "pass"  # an alternative examined, and three fits
    assert got["robustness"].outcome == "pass"
    assert got["null_decoy"].outcome == "not_checked"  # not a decoy
    # The fixture's fact readings were answered from its truth, and it declares no answer.
    known = got["known_answer"]
    assert known.outcome == "not_checked" and known.problems == []
    assert "no known answer is declared" in known.unchecked[0]


# ── end to end under Estimate ────────────────────────────────────────────────

NHANES_TRUTH = "_tt_tmp_nhanes.csv"
WIDER = {"kind": "set_exclusions", "rules": [
    {"column": "kcal", "low": 600, "high": 4500, "reason": "implausible intake"}]}


@pytest.fixture(scope="module")
def estimate_e2e(tmp_path_factory) -> dict[str, Any]:
    """The NHANES journey under Estimate (sugar and glucose, ``test_fit_lock_repair``'s drive, its
    truth ``truths.FIXTURE_TRUTHS``' own), every decision with a note: a preview of dropping the
    exclusion rule, the linear fit, Fit pressed (the lock), then the rule changed with the
    estimates in view, a second fit and its curve, and the export."""
    from turbotab.core.tests.acceptance.server_drive import settle_forms
    from turbotab.server.tests.conftest import wait_for
    from turbotab.server.tests.test_fit_lock_repair import _inference_drive, press

    folder = tmp_path_factory.mktemp("swarm_e2e_estimate")
    run_dir = folder / "runs" / "e2e-nhanes-1"
    start = _now()
    with analyst_server(run_dir / "home") as client:
        driver = ScriptedDriver(client)
        drive = _inference_drive(driver)
        pid = drive.pid
        preview = driver.post(f"/api/projects/{pid}/preview",
                              json={"kind": "set_exclusions", "rules": []})
        assert preview.status_code == 200, preview.text
        drive.decide({"kind": "select_models", "models": ["linear"]})
        wait_for(driver, pid, {"fit": "fresh"}, timeout=400)
        assert press(driver, pid)["locked"] is True
        wait_for(driver, pid, {"fit": "fresh", "mattered": "fresh"}, timeout=400)
        assert driver.get(f"/api/projects/{pid}/results").status_code == 200
        drive.decide(WIDER)
        settle_forms(drive)
        wait_for(driver, pid, {"fit": "fresh", "mattered": "fresh"}, timeout=400)
        assert driver.get(f"/api/projects/{pid}/export").status_code == 200
        finish(run_dir, driver, pid, start,
               {"id": "nhanes", "control": "known_answer", "fixture": NHANES_TRUTH})
        path, ok = A.assemble(run_dir, client=client, corpus=folder / "corpus")
        base = f"/api/projects/{pid}"
        served = {name: client.get(f"{base}{route}").json() for name, route in (
            ("view", ""), ("fits", "/fits"), ("mattered", "/mattered"))}
    assert ok, path.read_text("utf-8")[:2000]
    return {"rec": validate(json.loads(path.read_text("utf-8"))), "served": served,
            "notes": driver.notes}


def test_under_estimate_the_result_is_the_fit_the_lock_was_recorded_on(estimate_e2e):
    rec, served = estimate_e2e["rec"], estimate_e2e["served"]
    lock = next(r for r in served["view"]["decisions"] if r["decision"]["kind"] == "lock_plan")
    changed = next(r for r in served["view"]["decisions"] if r["after_estimates"])
    assert changed["decision"]["rules"][0]["low"] == WIDER["rules"][0]["low"]
    r = rec.result
    assert (r.lock, r.plan_digest) == (lock["seq"], lock["decision"]["digest"])
    fits = served["fits"]["records"]
    locked = next(f for f in fits if f["id"] == r.attempt)
    assert changed["seq"] not in locked["decisions"]
    assert r.effects == {f: rows for f, rows in locked["effects"].items() if rows}
    later = [f for f in fits if changed["seq"] in (f["decisions"] or ())]
    assert later and all(f["id"] != r.attempt for f in later)  # the secondary analysis is not it
    mine = {d.seq: d.reason for d in rec.decisions if d.source == "analyst"}
    assert mine == estimate_e2e["notes"]
    own = [d for d in rec.decisions if d.source == "app"]  # TurboTab's split and its lock
    assert [d.kind for d in own] == ["set_split", "lock_plan"]
    assert all(d.reason is None for d in own)


def test_under_estimate_a_choice_the_curve_varies_carries_its_departures(estimate_e2e):
    rec, curve = estimate_e2e["rec"], estimate_e2e["served"]["mattered"]["curve"]
    rules = [d for d in rec.decisions if d.kind == "set_exclusions"]
    first, wider = rules[0], rules[-1]
    assert wider.phase == "after_estimates"
    assert first.materiality.source is None and f"#{wider.seq}" in first.materiality.why
    assert wider.materiality.source == "mattered"
    keys = [c["key"] for c in curve["choices"] if c["decision"] == "set_exclusions"]
    departures = [s for s in curve["specs"] + curve["not_estimated"]
                  if not s["primary"] and any(s["picks"][k] != "recorded" for k in keys)]
    assert [d["choice"] for d in wider.materiality.departures] == keys
    assert [d["spec"] or {"reason": d["not_estimated"]} for d in wider.materiality.departures] == [
        s if "estimate" in s else {"reason": s["reason"]} for s in departures]
    assert wider.materiality.primary == next(s for s in curve["specs"] if s["primary"])
    # Both options it set aside were examined: dropping the rule (previewed) and the old rule (fit).
    assert sorted((e.how, e.alternative) for e in wider.examined) == [("fit", True),
                                                                     ("preview", True)]
    adjusted = [d for d in rec.decisions if d.kind == "set_adjustment"][-1]
    varied = [c["key"] for c in curve["choices"] if c["decision"] == "set_adjustment"]
    assert adjusted.materiality.source == "mattered"
    assert sorted({d["choice"] for d in adjusted.materiality.departures}) == sorted(varied)


def test_under_estimate_every_check_gives_what_the_runs_construction_implies(estimate_e2e):
    got = {o.check: o for o in C.run_checks([estimate_e2e["rec"]])}
    assert got["seal_integrity"].outcome == "pass", got["seal_integrity"].problems
    assert got["logging"].outcome == "pass", got["logging"].problems
    assert got["single_pass"].outcome == "pass"  # the rule's removal previewed before the lock
    assert got["robustness"].outcome == "pass"
    known = got["known_answer"]
    assert known.outcome == "not_checked" and known.problems == []  # kcal's unit and days declared

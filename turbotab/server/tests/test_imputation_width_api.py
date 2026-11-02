"""The imputation model's width rule at the answer and at the Fit press, over HTTP (INBOX: Sullivan
2021's runs 22 and 26 locked a multiple imputation that Fit then refused, after the lock).

``smcfcs`` registers the check when it is imported, and fits run in spawned worker processes, so a
server process holds the check only if serving imports it. The drive therefore runs in a fresh
process that imports only what the server imports: a test process that had imported ``smcfcs``
itself (as ``core/tests/test_imputation_width.py`` does) would pass whether the server held the
check or not. It drives the 150-row wide table of that file (about 90 model columns; every
expected word is from its pandas counts) through the real server under inference: the roles,
each one proposed below high confidence confirmed on its own, then multiple imputation, which is
refused when it is answered; then complete cases, the questions up to the models, and multiple
imputation written to the log as a log made before this check holds it (runs 22 and 26), so Fit is
pressed on it: the press is refused and no lock is recorded.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[3]
ROLES = {"id": "identifier", "x": "exposure", **{f"c{j:02d}": "covariate" for j in range(1, 85)},
         "g": "covariate", "smoke": "covariate"}
LEAD = "Multiple imputation cannot run on these data: "


def drive(table: str, home: str) -> None:
    """In a fresh process: the drive the test reads, printed as one line of JSON."""
    import time

    from turbotab.server.tests import conftest as c

    out: dict[str, Any] = {}
    with c.make_client(Path(home), "local", 1, "http://127.0.0.1") as client:
        pid = c.open_by_path(client, Path(table))
        # the causal answers the adjustment card asks: every covariate a confounder
        c.declare(pid, {"exposure:y": "x", **{f"adjust:{k}": "yes,yes,no" for k, r in ROLES.items()
                                              if r == "covariate"}}, fixture="wide.csv")
        c.wait_for(client, pid, {"ingest": "fresh"})

        def post(body: dict[str, Any], walk: bool = True) -> Any:
            if walk:
                c.prepare(client, pid, body, timeout=300)
            return client.post(f"/api/projects/{pid}/decisions", json=body)

        for body in ({"kind": "set_lens", "lenses": ["clinical"]},
                     {"kind": "set_target", "column": "y"},
                     {"kind": "set_purpose", "purpose": "inference"},
                     {"kind": "set_roles", "roles": ROLES}):
            answered = post(body)
            assert answered.status_code == 200, (body["kind"], answered.text[:600])
        for col in ("x", "g", "smoke"):  # proposed below high confidence: one press each
            answered = post({"kind": "confirm_readings", "items": [
                {"reading": "role", "column": col, "value": ROLES[col]}]}, walk=False)
            assert answered.status_code == 200, (col, answered.text[:600])
        mi = {"kind": "set_missing", "strategy": "multiple_imputation"}
        answered = post(mi)
        out["set_missing"] = {"status": answered.status_code, "body": answered.json()}
        # complete cases (recorded over multiple imputation where the server took it), and on to
        # the models, so the press below is on what the log holds
        answered = post({"kind": "set_missing", "strategy": "complete_case"})
        assert answered.status_code == 200, answered.text[:600]
        answered = post({"kind": "select_models", "models": ["linear"]})
        assert answered.status_code == 200, answered.text[:600]
        from turbotab.core.decisions import SetMissing

        service = client.app.state.service
        service.log(pid).append(SetMissing(strategy="multiple_imputation"))  # as runs 22 and 26
        service.engine.on_decision(pid)
        end = time.monotonic() + 120
        while any(s["status"] in ("open", "waiting")
                  for s in client.get(f"/api/projects/{pid}").json()["interview"]):
            assert time.monotonic() < end, "the questions never settled after the log's answer"
            time.sleep(0.05)
        pressed = client.post(f"/api/projects/{pid}/fit")
        out["fit"] = {"status": pressed.status_code, "body": pressed.json()}
        out["kinds"] = [r.decision.kind for r in service.log(pid).records()]
    # the sampler's module came from the server's own imports, never from this test's
    out["test_module"] = "turbotab.core.tests.test_imputation_width" in sys.modules
    print(json.dumps(out))


@pytest.mark.xfail(strict=True, reason=(
    "decisions.py's tail does not import turbotab.core.methods.smcfcs, so a server process never "
    "registers the check (open issue: one line after the row_floor import; drop this mark with it)"))
def test_the_server_refuses_the_150_row_imputation_at_the_answer_and_at_the_fit_press(tmp_path):
    """Multiple imputation on 150 rows of about 90 model columns: refused when it is answered,
    with the fit's words and complete cases as the way forward; and a log that holds it (made
    before this check) is refused at the Fit press, which records no lock."""
    from turbotab.core.tests.test_imputation_width import _refusal_words, wide

    table = wide(150)
    path = tmp_path / "wide.csv"
    table.to_csv(path, index=False)
    code = f"from {__name__} import drive; drive({str(path)!r}, {str(tmp_path / 'home')!r})"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO,
                          timeout=540)
    assert done.returncode == 0, done.stderr[-3000:]
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out["test_module"] is False  # the core tests' module, which imports smcfcs, never ran
    words = LEAD + _refusal_words(table)
    answered = out["set_missing"]
    assert answered["status"] == 409, answered["body"]
    error = answered["body"]["error"]
    assert error["code"] == "imputation_too_wide" and error["message"] == words
    assert [e["label"] for e in error["exits"]] == ["Complete cases, with their assumption stated"]
    assert error["exits"][0]["decision"]["strategy"] == "complete_case"
    pressed = out["fit"]
    assert pressed["status"] == 409, pressed["body"]
    assert pressed["body"]["error"]["code"] == "imputation_too_wide"
    assert pressed["body"]["error"]["message"] == words
    assert "lock_plan" not in out["kinds"]

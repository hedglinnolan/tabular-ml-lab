"""GET /results under Estimate serves no effect for a family whose fit gives no coefficient table
(the methods lead's ruling of 2026-10-23 on the shrunk effect): a number is right or it is refused.

Three short Estimate journeys (``turbotab.core.reference.journeys``) on the walking table the spline
rule's replay writes (``test_replay_spline_rule.walking_bp`` at its seed: 150 rows, 30 blanks in
``chronic_pain``, so the complete cases are 120), each with the plan locked by Fit:

* **ridge alone.** Ridge declares a shrunk table with no intervals (``ridge.py``,
  ``InferenceDecl(table="shrunk_no_intervals")``), so the effects stage refuses: no chosen family has
  a coefficient table. Its effect slot serves no rows, and in their place the effects stage's own
  refusal, as the stage's status serves it (its code, words and ways forward); the words are the
  ones the refusal's construction gives (``stages.effects.no_table_refusal``: the linear model is
  the first family offered). None of the ridge fit's coefficients, as the fit route serves them,
  appears anywhere in the families served.
* **the linear model alone.** Its effect is unchanged: the rows of ``walks_daily`` adjusted for
  ``chronic_pain`` and ``age`` (the walking table's declared truth: a confounder and a cause of the
  outcome; ``age`` a straight line), whose estimate is statsmodels' least squares on the 120
  complete cases, and whose standard error, interval and p-value are statsmodels' HC3 with
  t(120 − 4); no refusal beside it.
* **the linear model and ridge.** The effects stage fits the linear model and gives no refusal, so
  the linear model's effect is the one above and ridge's slot is empty, with no refusal either.

What the results serve counts as shown: the linear model's effect makes the lock stand; ridge's
slot, empty, shows nothing, so the results leave the lock as it was.
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from turbotab.core.tests import test_replay_spline_rule as R

EXPOSURE, OUTCOME = "walks_daily", "sbp_mmhg"
# The effects stage's refusal as its construction words it, the linear model offered first
# (``test_no_coefficient_table_refusal.EFFECTS_WORDS``).
NO_TABLE_WORDS = ("None of the chosen model families has a coefficient table, so there is no "
                  "effect to report; choose the linear model.")


def _locked_journey(models: list[str]) -> dict[str, Any]:
    """The walking table under Estimate with ``models`` chosen, Fit pressed (the plan locked), and
    what the routes serve then: GET /results, the project's view and GET /stages/fit."""
    from turbotab.core.reference import journeys as J
    from turbotab.core.tests.acceptance.server_drive import Drive, local_server, press_fit
    from turbotab.core.tests.truths import FIXTURE_TRUTHS

    with tempfile.TemporaryDirectory(prefix="tt-no-table-") as folder:
        path = Path(folder) / R.FIXTURE
        R.walking_bp(R.SEED).to_csv(path, index=False)
        spec = J.Journey("no-table", "clinical", "inference", "?", str(path), "test",
                         lambda: path,
                         lambda: J.truth(dict(FIXTURE_TRUTHS[R.FIXTURE]), fixture=path.name),
                         target=OUTCOME, lenses=("clinical",), exposure=EXPOSURE,
                         roles=dict(R.ROLES),
                         answers={"missing": R.COMPLETE_CASES,
                                  "models": {"kind": "select_models", "models": models}})
        run = J.Run(spec)
        with local_server(Path(folder) / "home") as client:
            run.client = J.Recording(client, run)
            r = client.post("/api/projects", json={"path": str(path)})
            assert r.status_code == 200, r.text
            run.pid = pid = r.json()["id"]
            truth = spec.truth()
            truth.guess, truth.ranked = J._server_guess(client, pid), J._server_ranked(client, pid)
            run.drive = Drive(run.client, pid, truth)
            run.drive.exposure = EXPOSURE
            run.drive.artifact("ingest", timeout=300)
            assert J.follow(run), run.notes
            assert press_fit(client, pid)
            end = time.monotonic() + 240
            while True:
                view = client.get(f"/api/projects/{pid}").json()
                stages = view["stages"]
                if (stages["fit"]["status"] == "fresh"
                        and stages["effects"]["status"] in ("fresh", "error")):
                    break
                assert time.monotonic() < end, {s: stages[s]["status"] for s in ("fit", "effects")}
                time.sleep(0.1)
            assert view["state"]["plan_locked"] is True
            response = client.get(f"/api/projects/{pid}/results")
            assert response.status_code == 200, response.text
            # What the results served counts as shown: the lock stands only on an estimate shown.
            shown = client.app.state.service._shown_lock(pid)
            [lock] = [r["id"] for r in view["decisions"] if r["decision"]["kind"] == "lock_plan"]
            fit = client.get(f"/api/projects/{pid}/stages/fit").json()["artifact"]
            return {"results": response.json(), "view": view, "fit": fit, "shown": shown,
                    "lock": lock}


@pytest.fixture(scope="module")
def journeys() -> dict[str, dict[str, Any]]:
    return {"ridge": _locked_journey(["ridge"]), "linear": _locked_journey(["linear"]),
            "both": _locked_journey(["linear", "ridge"])}


def _ols_hc3() -> dict[str, float]:
    """``walks_daily``'s row of statsmodels' least squares on the complete cases, adjusted for
    ``chronic_pain`` and ``age``, with HC3 standard errors and t(n − 4) intervals."""
    import statsmodels.api as sm

    frame = R.complete(R.SEED)
    X = sm.add_constant(frame[[EXPOSURE, "chronic_pain", "age"]].astype(float))
    fitted = sm.OLS(frame[OUTCOME].astype(float), X).fit(cov_type="HC3", use_t=True)
    low, high = fitted.conf_int().loc[EXPOSURE]
    return {"estimate": float(fitted.params[EXPOSURE]), "se": float(fitted.bse[EXPOSURE]),
            "ci_low": float(low), "ci_high": float(high), "p": float(fitted.pvalues[EXPOSURE]),
            "df": float(fitted.df_resid), "n": float(len(frame))}


def _numbers(obj: Any) -> list[float]:
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, dict):
        return [x for v in obj.values() for x in _numbers(v)]
    if isinstance(obj, list):
        return [x for v in obj for x in _numbers(v)]
    return []


def _family(document: dict[str, Any], name: str) -> dict[str, Any]:
    [family] = [f for f in document["families"] if f["family"] == name]
    return family


def test_a_family_with_no_coefficient_table_serves_the_effects_stages_refusal_not_its_rows(
        journeys):
    got = journeys["ridge"]
    document, status = got["results"], got["view"]["stages"]["effects"]
    # The effects stage refused, in its own words: those its construction gives.
    stage_refusal = status["refusal"]
    assert status["status"] == "error" and stage_refusal is not None
    assert stage_refusal["code"] == "no_coefficient_table"
    assert stage_refusal["message"] == NO_TABLE_WORDS
    assert [e["decision"] for e in stage_refusal["exits"]][0] == {
        "kind": "select_models", "models": ["ridge", "linear"]}
    # The fit computed ridge's shrunk rows of the exposure (the fit route serves them)...
    [ridge_fit] = [m for m in got["fit"]["models"] if m["family"] == "ridge"]
    shrunk = [r["estimate"] for r in ridge_fit["coefficients"] or [] if r["feature"] == EXPOSURE]
    assert len(shrunk) == 1 and shrunk[0] is not None and ridge_fit["inference"] is None
    # ...and the results serve none of them: the effects stage's refusal in their place.
    family = _family(document, "ridge")
    assert family["effect"] is None
    assert family["refusal"] == stage_refusal
    assert shrunk[0] not in _numbers(document["families"])
    assert document["estimand"] is not None and document["withheld"] is None
    assert got["shown"] is None  # no estimate shown, so none makes the lock stand


def test_a_family_with_a_coefficient_table_serves_its_least_squares_effect_unchanged(journeys):
    document = journeys["linear"]["results"]
    assert journeys["linear"]["shown"] == journeys["linear"]["lock"]  # shown under the lock
    family = _family(document, "linear")
    assert family["refusal"] is None
    [row] = [r for r in family["effect"] if r["feature"] == EXPOSURE]
    truth = _ols_hc3()
    assert truth["n"] == R.N - R.BLANKS[R.SEED] == 120 and truth["df"] == 116
    for k in ("estimate", "se", "ci_low", "ci_high", "p", "df"):
        assert row[k] == pytest.approx(truth[k], rel=1e-9, abs=1e-12), k
    # A family beside it with no table: the stage fits the linear model and refuses nothing, so
    # the linear model's effect stands and ridge's slot is empty, with no refusal to serve.
    both = journeys["both"]
    assert both["view"]["stages"]["effects"]["refusal"] is None
    assert _family(both["results"], "linear")["effect"] == family["effect"]
    ridge = _family(both["results"], "ridge")
    assert ridge["effect"] is None and ridge["refusal"] is None
    [ridge_fit] = [m for m in both["fit"]["models"] if m["family"] == "ridge"]
    shrunk = [r["estimate"] for r in ridge_fit["coefficients"] if r["feature"] == EXPOSURE]
    assert shrunk and shrunk[0] not in _numbers(both["results"]["families"])
    assert np.isfinite(shrunk[0]) and shrunk[0] != pytest.approx(truth["estimate"], abs=1e-6)
    json.dumps(both["results"])  # served as JSON whole

"""Candidate 10 over HTTP: under Predict, once the held-out rows are opened with a final family
named, the served fit's and GET /results' declared result is the held-out score the opening
recorded for that family (audit WP16, RO-05), where it stayed null with "will be the result".

dietary_recalls.csv, 25% held out, linear and elastic net fitted (linear first in fit order) and
elastic net declared final at the opening. The reference is the fit's sealed frame on disk, read
here with pandas, at the opening: the result's estimate is elastic net's held-out score on the
primary metric there, and its interval is elastic net's served held-out interval. After a change
made after the opening (an exclusion) the fit is computed again: the result keeps the score read at
the opening and gives no interval, while the new fit's held-out score on disk differs from it.
"""
from __future__ import annotations

import time

import pandas as pd
import pytest

from turbotab.core.graph import artifact_dir
from turbotab.core.tests.acceptance.server_drive import press_fit
from turbotab.core.tests.test_sentence_precision import table
from turbotab.server.tests.conftest import prepare, wait_for
from turbotab.server.tests.test_seal_api import decide, sealed_project


def disk_scores(client, pid: str) -> dict[str, dict[str, float]]:
    """The held-out scores the fresh fit holds, read from its sealed frame on disk."""
    view = wait_for(client, pid, {"fit": "fresh"}, timeout=180)
    cache = client.app.state.service.workspace.cache_dir(pid)
    frame = pd.read_parquet(artifact_dir(cache, "fit", view["stages"]["fit"]["key"]) / "frames"
                            / "sealed_scores.parquet")
    out: dict[str, dict[str, float]] = {}
    for row in frame.itertuples(index=False):
        out.setdefault(row.family, {})[row.metric] = float(row.value)
    return out


def fit_key(client, pid: str) -> str:
    return wait_for(client, pid, {"fit": "fresh"}, timeout=180)["stages"]["fit"]["key"]


def refitted(client, pid: str, before: str, timeout: float = 180.0) -> str:
    """Wait for a fresh fit at another key than ``before`` (the status may read fresh at the old
    key for a moment after the decision)."""
    end = time.monotonic() + timeout
    while True:
        key = fit_key(client, pid)
        if key != before:
            return key
        assert time.monotonic() < end, "no new fit"
        time.sleep(0.1)


def served(client, pid: str) -> tuple[dict, dict]:
    fit = client.get(f"/api/projects/{pid}/stages/fit").json()["artifact"]
    document = client.get(f"/api/projects/{pid}/results").json()
    return fit, document


@pytest.mark.slow  # three fits on the dietary recalls, one after the opening: about 2.5 minutes
def test_after_the_opening_results_say_the_score_the_opening_recorded_for_the_final_family(client):
    pid = sealed_project(client)
    decide(client, pid, {"kind": "select_models", "models": ["linear", "elastic_net"]})
    wait_for(client, pid, {"fit": "fresh"}, timeout=180)
    assert press_fit(client, pid)
    fit, document = served(client, pid)
    assert fit["result"]["basis"] == "holdout" and fit["result"]["estimate"] is None
    assert fit["result"]["sentence"].endswith("will be the result.")  # sealed: nothing to report

    prepare(client, pid, {"kind": "open_seal", "family": "elastic_net"})
    wait_for(client, pid, {"fit": "fresh"}, timeout=180)
    at_disk = disk_scores(client, pid)
    decide(client, pid, {"kind": "open_seal", "family": "elastic_net"})
    fit, document = served(client, pid)
    primary = fit["primary_metric"]
    score = at_disk["elastic_net"][primary]
    assert score != at_disk["linear"][primary]  # the first family in fit order is not the result
    assert [f["family"] for f in document["families"]] == ["linear", "elastic_net"]
    assert document["result"] == fit["result"]
    result = document["result"]
    assert result["basis"] == "holdout" and result["family"] == "elastic_net"
    assert result["estimate"] == score
    interval = next(f for f in document["families"] if f["family"] == "elastic_net")[
        "intervals"][primary]
    assert interval["ci_low"] is not None
    assert (result["ci_low"], result["ci_high"]) == (interval["ci_low"], interval["ci_high"])
    assert f"is the result: {table(score)} on " in result["sentence"]
    assert "will be" not in result["sentence"]

    # A change after the opening fits again: the result stays the score the opening recorded,
    # with no interval, and the score shown now is the new fit's.
    opened_key = fit_key(client, pid)
    rule = {"kind": "set_exclusions",
            "rules": [{"column": "age", "low": 40, "reason": "adults over 40"}]}
    assert decide(client, pid, rule)["decisions"][-1]["post_seal"] is True
    refitted(client, pid, opened_key)
    now = disk_scores(client, pid)
    assert now["elastic_net"][primary] != score  # the scenario discriminates
    fit, document = served(client, pid)
    assert document["changed_after_seal"] is True and document["at_opening"]["current"] is False
    assert document["result"] == fit["result"]
    result = document["result"]
    assert result["family"] == "elastic_net" and result["estimate"] == score
    assert result["ci_low"] is None and result["ci_high"] is None
    assert result["sentence"].endswith("; the opening recorded no interval on it.")

"""GET …/methods says the comparisons' repeats the fit ran, as the rows stand (the large-table
ruling of 2026-10-23, ``voice.restated_repeats``), through the real server.

The split's sentence says the repeats on the rows as they stand when it is answered: 5 from 5,000
training rows, 10 below. An exclusion answered after it moves the training rows across 5,000, and
the fit then runs the other count. The references: the training rows counted here from the table
and ``RULE`` (the ages below 40 it leaves out, counted on the frame), and the repeats the fit itself
reports (``comparison.repeats``). The service's count is read three ways: from the fresh fit, from
the fresh split with the fit read as stale, and with both read as stale, when the rows analyzed now
bound it or the rule is said. The table is the synthetic diet table of
``core/tests/acceptance/test_export.py``, one row per ``pid``, at 5,200 rows.
"""
from __future__ import annotations

import time
from typing import Any

import pytest

from turbotab.core.tests.acceptance.server_drive import local_server
from turbotab.core.tests.acceptance.test_export import _diet, _open_diet

COMPARED = ("models were compared with each other and with the no-predictor baseline on `5`-fold "
            "cross-validation repeated ")
RULE = {"column": "age", "low": 40, "reason": "adults of 40 or more"}


def _fresh(drive: Any, timeout: float = 300.0) -> dict[str, Any]:
    """The view once the split and the fit are fresh on the answers as they stand."""
    end = time.monotonic() + timeout
    while True:
        view = drive.view()
        statuses = {s: view["stages"][s]["status"] for s in ("cohort", "split", "fit")}
        assert "error" not in statuses.values(), view["stages"]["fit"]
        if set(statuses.values()) == {"fresh"}:
            return view
        assert time.monotonic() < end, statuses
        time.sleep(0.1)


def _split_line(client: Any, pid: str) -> str:
    r = client.get(f"/api/projects/{pid}/methods")
    assert r.status_code == 200, r.text[:600]
    [line] = [x for x in r.json()["lines"] if x["kind"] == "set_split"]
    return line["sentence"]


@pytest.fixture(scope="module")
def walks(tmp_path_factory) -> dict[str, Any]:
    """Two projects on the 5,200-row table: the split answered with no exclusion (5,200 train),
    then ``RULE`` answered; and the split answered under ``RULE``, then no exclusion."""
    folder = tmp_path_factory.mktemp("repeats_methods")
    frame = _diet(5_200, seed=52)
    csv = folder / "diet.csv"
    frame.to_csv(csv, index=False)
    kept = int((frame["age"] >= RULE["low"]).sum())
    out: dict[str, Any] = {"kept": kept}
    with local_server(folder / "home") as client:
        for name, then, now in (("drops", [], [RULE]), ("restores", [RULE], [])):
            drive = _open_diet(client, csv, "prediction")
            if then:
                drive.decide({"kind": "set_exclusions", "rules": then})
            drive.answer("split", {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})
            drive.answer("models", {"kind": "select_models", "models": ["linear"]})
            view = _fresh(drive)
            said = [r["sentence"] for r in view["decisions"]
                    if r["decision"]["kind"] == "set_split"][-1]
            before = (_split_line(client, drive.pid), drive.artifact("fit")["comparison"]["repeats"])
            drive.decide({"kind": "set_exclusions", "rules": now})
            view = _fresh(drive)
            out[name] = {"said": said, "before": before,
                         "after": (_split_line(client, drive.pid),
                                   drive.artifact("fit")["comparison"]["repeats"]),
                         "record": [r["sentence"] for r in view["decisions"]
                                    if r["decision"]["kind"] == "set_split"][-1],
                         "seen": _as_if_stale(client, drive.pid)}
    return out


def _as_if_stale(client: Any, pid: str) -> dict[tuple[str, ...], Any]:
    """The service's repeats for the methods text, and the methods text's split line, with the fit
    read as not fresh (so the split drawn on the answers as they stand counts them), and with the
    split too (so neither is drawn, and the rule is said)."""
    service = client.app.state.service
    real = service.engine.status
    analyzed = client.get(f"/api/projects/{pid}/stages/cohort").json()["artifact"]["n_final"]
    out: dict[tuple[str, ...], Any] = {}
    for stale in ((), ("fit",), ("fit", "split")):
        def status(project: str, stale: tuple[str, ...] = stale) -> dict[str, Any]:
            return {k: v.model_copy(update={"status": "stale"}) if k in stale else v
                    for k, v in real(project).items()}

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(service.engine, "status", status)
            line = next(x.sentence for x in service.methods(pid).lines if x.kind == "set_split")
            out[stale] = (service._comparisons_run(pid, analyzed), line)
    return out


@pytest.mark.slow
def test_an_exclusion_after_the_split_moves_the_repeats_the_methods_text_says(walks):
    """5,200 rows train when the split is answered: 5 repeats, said, run, and in the methods text.
    ``RULE`` then leaves fewer than 5,000 (counted here): the fit runs 10, and the methods text
    says 10, while the Record keeps the 5 said."""
    assert walks["kept"] < 5_000
    walk = walks["drops"]
    assert f"{COMPARED}`5` times, by" in walk["said"]
    assert walk["before"] == (walk["said"], 5)  # with no later answer, the sentence as said
    line, ran = walk["after"]
    assert ran == 10
    assert line == walk["said"].replace(f"{COMPARED}`5` times", f"{COMPARED}`10` times")
    assert walk["record"] == walk["said"]
    _the_service_counts_from_the_fit_or_the_split_or_says_the_rule(walk, 10, walks["kept"])


@pytest.mark.slow
def test_an_exclusion_withdrawn_after_the_split_moves_the_repeats_back(walks):
    """Under ``RULE`` fewer than 5,000 rows train when the split is answered: 10 repeats, said and
    run; the exclusion withdrawn, 5,200 train, the fit runs 5, and the methods text says 5."""
    walk = walks["restores"]
    assert f"{COMPARED}`10` times, by" in walk["said"]
    assert walk["before"] == (walk["said"], 10)
    line, ran = walk["after"]
    assert ran == 5
    assert line == walk["said"].replace(f"{COMPARED}`10` times", f"{COMPARED}`5` times")
    assert walk["record"] == walk["said"]
    _the_service_counts_from_the_fit_or_the_split_or_says_the_rule(walk, 5, 5_200)


def _the_service_counts_from_the_fit_or_the_split_or_says_the_rule(walk: dict[str, Any],
                                                                   ran: int, analyzed: int) -> None:
    """The fit's count where it is fresh; the same count from the split alone; and, with neither
    fresh, the rows analyzed now (``analyzed``): fewer than 5,000, so at most those train and 10
    is said; 5,000 or more, so the rule, by row (one row per ``pid``, so nothing groups them)."""
    counted = {"comparison_repeats": ran}
    line = walk["after"][0]
    assert walk["seen"][()] == ({"set_split": counted, "set_validation": counted}, line)
    assert walk["seen"][("fit",)] == walk["seen"][()]
    unknown = {"comparison_repeats": None, "n_analyzed": analyzed, "grouped": False}
    said = "`10` times" if analyzed < 5_000 else "`10` times (`5` where `5,000` or more rows train)"
    assert walk["seen"][("fit", "split")] == (
        {"set_split": unknown, "set_validation": unknown},
        line.replace(f"{COMPARED}`{ran}` times", f"{COMPARED}{said}"))

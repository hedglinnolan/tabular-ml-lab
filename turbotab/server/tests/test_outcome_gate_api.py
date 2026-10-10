"""The outcome beside a column waits for its gate, on every route (CROSSWALK disagreement 2,
"the outcome beside a column"; question 1, ruled 2026-10-08; calm/FOUNDATION §3 rule 6).

Under Estimate and Describe (the engine's ``inference``), and with the goal unanswered (the
strictest case), no route serves the outcome beside another column before the plan's lock; under
Predict none does before the held-out rows are drawn. The sweep proves it without knowing what any
route computes: two projects are opened on files that differ only in which row holds which
outcome value (a permutation of the outcome, so the outcome alone and every other column alone are
the same in both). Every GET route the app mounts (each stage, each finding's evidence, the data
reads) and the consequence previews are called on both at the same point of the same journey, and
every response must be the same in both, once the project's own id and times are set aside. A
statistic of the outcome beside a column (a row window holding both, a mean of the outcome by a
column's levels, a correlation, a pairwise view) differs between the two; the outcome alone does
not. The routes that record are not called: reading records nothing.

Each journey then checks the outcome alone at its own gate: refused with its line before it, and
once open drawn on the rows kept so far (Estimate) or the training rows (Predict), carrying the
``view_outcome`` record. After the draw (Predict) and the lock (Estimate) the outcome beside a
column opens, and under Predict the sweep then tells the twins apart: its positive control.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pytest

from turbotab.core import outcome_gate
from turbotab.core.tests.acceptance.server_drive import Drive, open_project
from turbotab.core.tests.truths import Truth, fixture_truth
from turbotab.server.tests.conftest import SAMPLES
from turbotab.server.tests.test_seal_api import EXEMPT, mounted

CLINICAL = SAMPLES / "clinical_risk.csv"
OUTCOME = "readmit_30d"
ROLES = {"encounter_id": "identifier", "age": "covariate", "sex": "covariate",
         "charlson_index": "covariate", "prior_admissions_12mo": "covariate",
         "albumin_g_dl": "exposure", "creatinine_mg_dl": "covariate",
         "hemoglobin_g_dl": "covariate", "sodium_mmol_l": "covariate",
         "length_of_stay_days": "covariate"}
EXPOSURE = "albumin_g_dl"
KEPT = 20  # the first rows keep their outcome, so a route quoting the first values is unchanged
# The draw's cells (the split's preview): which rows are held out, one cell per row. A draw
# stratified by the outcome depends on it only through its balance, and on no other column, so it
# is no statistic of the outcome beside a column; the held-out rows themselves stay sealed.
DRAW_CELLS = (".seal.hold", ".seal.unit")
# Fields that differ between any two projects whatever their data: ids, times, fingerprints.
VOLATILE = {"id", "pid", "key", "at", "created", "created_at", "updated", "updated_at", "modified",
            "fingerprint", "sha256", "decision_id", "job_id", "seq_at", "path", "source_path",
            "started", "finished", "elapsed", "seconds", "duration", "eta", "progress",
            "timestamp", "time", "when", "recorded_at", "hash", "digest", "version_key"}


def twin_files(tmp_path: Path) -> tuple[Path, Path]:
    """``clinical_risk.csv`` as it is, and with its outcome permuted after the first rows: the
    same name, so the fixture's declared truth answers both."""
    frame = pd.read_csv(CLINICAL, dtype=str, keep_default_na=False)
    assert (frame[OUTCOME] != "").all()  # no blank outcome: Who's in keeps the same rows in both
    twin = frame.copy()
    rest = twin[OUTCOME].iloc[KEPT:].to_numpy()
    twin.loc[twin.index[KEPT:], OUTCOME] = np.random.default_rng(7).permutation(rest)
    assert (twin[OUTCOME] != frame[OUTCOME]).sum() > 50
    out = []
    for name, f in (("as_is", frame), ("permuted", twin)):
        folder = tmp_path / name
        folder.mkdir()
        f.to_csv(folder / CLINICAL.name, index=False)
        out.append(folder / CLINICAL.name)
    return out[0], out[1]


def opened(client: Any, path: Path) -> Drive:
    truth = fixture_truth(CLINICAL.name)
    truth = Truth({**truth, **{f"role:{c}": r for c, r in ROLES.items()},
                   f"exposure:{OUTCOME}": EXPOSURE,
                   **{f"adjust:{c}": "no,yes,no" for c, r in ROLES.items() if r == "covariate"}},
                  fixture=CLINICAL.name)
    drive = open_project(client, path, truth)
    drive.exposure = EXPOSURE
    drive.decide({"kind": "set_lens", "lenses": ["clinical"]})
    drive.reach("target")
    drive.decide({"kind": "set_target", "column": OUTCOME})
    drive.answer("task", {"kind": "set_task", "column": OUTCOME, "task": "binary"})
    drive.answer("event", {"kind": "set_event", "column": OUTCOME, "level": "1"})
    drive.reach("purpose")
    return drive


def whos_in(drive: Drive, purpose: str) -> None:
    drive.decide({"kind": "set_purpose", "purpose": purpose})
    drive.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit"})
    drive.reach("roles")
    drive.decide_roles(ROLES)
    drive.answer("exclusions", {"kind": "set_exclusions", "rules": []})
    drive.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
    drive.reach("split")


def settle(drive: Drive, timeout: float = 300.0) -> None:
    end = time.monotonic() + timeout
    while True:
        stages = drive.view()["stages"].values()
        if not any(s["status"] in ("queued", "running", "stale") and s.get("held") != "fit"
                   for s in stages):
            return
        assert time.monotonic() < end, [s["stage"] for s in stages if s["status"] != "fresh"]
        time.sleep(0.1)


def body(response: Any) -> Any:
    try:
        return response.json()
    except ValueError:
        return {"bytes": len(response.content), "type": response.headers.get("content-type")}


def calls(client: Any, drive: Drive, tmp_path: Path, fids: list[str]
          ) -> Iterator[tuple[str, str, Any]]:
    """Every route the app mounts, as (template, label, response): each GET with each value of its
    parameters (every stage, the outcome and another column, every finding), the consequence
    previews of the decisions a person could preview now, and nothing that records."""
    service = client.app.state.service
    pid = drive.pid
    stages = [s.name for s in service.engine.graph.stages()]
    columns = [OUTCOME, EXPOSURE, "sex"]
    values = {"pid": [pid], "stage": stages, "name": columns, "fid": fids or ["no_such_finding"],
              "jid": ["no-such-job"]}
    queries: dict[str, list[dict[str, Any]]] = {
        "/api/fs/list": [{"path": str(tmp_path)}],
        "/api/projects/{pid}/table": [{}, {"columns": OUTCOME}, {"columns": f"{EXPOSURE},{OUTCOME}"},
                                      {"limit": 500}],
        "/api/projects/{pid}/columns": [{}, {"names": OUTCOME}, {"query": "readmit"}],
    }
    previews = [{"kind": "set_purpose", "purpose": p} for p in ("inference", "prediction")] + [
        {"kind": "set_target", "column": OUTCOME},
        {"kind": "set_event", "column": OUTCOME, "level": "0"},
        {"kind": "set_task", "column": OUTCOME, "task": "binary"},
        {"kind": "set_roles", "roles": ROLES},
        {"kind": "set_exclusions", "rules": [{"column": "age", "low": 18, "high": 80,
                                              "reason": "adults"}]},
        {"kind": "set_missing", "strategy": "impute"},
        {"kind": "set_missing", "strategy": "complete_case"},
        {"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5},
        {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5},
        {"kind": "set_exposure_form", "column": EXPOSURE, "form": "spline"},
        {"kind": "select_models", "models": ["linear"]},
        {"kind": "select_models", "models": ["linear", "elastic_net", "boosted_trees"]},
        {"kind": "set_estimand", "exposure": EXPOSURE, "measure": "risk_difference"},
        {"kind": "set_model_sequence", "exposure": EXPOSURE, "model_1": ["age", "sex"]},
        {"kind": "set_modification", "modifier": "sex"},
        {"kind": "set_categorical", "columns": ["sex"]},
        {"kind": "set_levers", "forms": "rule"},
        {"kind": "set_selection", "method": "elastic_net"},
        {"kind": "set_intended_use", "use": "risk_estimation"},
        {"kind": "set_explain", "curves": "ale"},
        {"kind": "set_multiplicity", "method": "bh"},
        {"kind": "set_sensitivity", "analyses": [{"label": "adults", "rules": [
            {"column": "age", "low": 18, "high": 80, "reason": "adults"}]}]},
        {"kind": "respond_diagnostic", "exposure": EXPOSURE, "check": "influence",
         "action": "without_influential"},
        {"kind": "set_causal", "exposure": EXPOSURE, "method": "dml_plr"},
        {"kind": "set_measurement_error", "method": "regression_calibration",
         "exposures": [EXPOSURE]},
        {"kind": "set_updating", "method": "shrinkage"},
    ] + [{"kind": "apply_repair", "finding_id": f, "option": "set_missing"} for f in fids[:3]]
    for template, methods in sorted(mounted(client).items()):
        if template in EXEMPT:
            continue
        paths = [template]
        for name, options in values.items():
            if "{" + name + "}" in template:
                paths = [p.replace("{" + name + "}", str(v)) for p in paths for v in options]
        for path in paths:
            if "GET" in methods:
                for q in queries.get(template, [{}]):
                    yield template, f"GET {path} {q}", client.get(path, params=q)
            if "POST" in methods and template == "/api/projects/{pid}/preview":
                for d in previews:
                    yield template, f"POST {path} {d['kind']} {json.dumps(d)[:80]}", \
                        client.post(path, json=d)


def normalized(obj: Any, pid: str, where: str = "") -> Any:
    if isinstance(obj, dict):
        return {k: normalized(v, pid, f"{where}.{k}") for k, v in obj.items()
                if k not in VOLATILE and not k.endswith(("_seconds", "_bytes", "_at"))
                and not f"{where}.{k}".endswith(DRAW_CELLS)}
    if isinstance(obj, list):
        return [normalized(v, pid, where) for v in obj]
    if isinstance(obj, str):
        if re.match(r"Fitting (?:it|these) takes ", obj):
            return None  # said only when the shelf's measured seconds pass a threshold
        s = obj.replace(pid, "<pid>")
        s = re.sub(r"\d{4}-\d\d-\d\dT[\d:.]+(?:[+-]\d\d:\d\d|Z)?", "<time>", s)
        s = re.sub(r"\d\d:\d\d:\d\d(?:\.\d+)?", "<time>", s)
        s = re.sub(r"\b[0-9a-f]{10,64}\b", "<hex>", s)
        # A family's measured time on this machine (the shelf), not a statistic of the data.
        return re.sub(r"about \d+ (?:second|minute|hour)s?|under a second", "<duration>", s)
    return obj


def differences(a: Any, b: Any, where: str = "") -> Iterator[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            yield from differences(a.get(k), b.get(k), f"{where}.{k}")
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            yield from differences(x, y, f"{where}[{i}]")
    elif a != b:
        yield f"{where}: {json.dumps(a)[:120]} != {json.dumps(b)[:120]}"


def sweep(client: Any, a: Drive, b: Drive, tmp_path: Path) -> list[str]:
    """Every route called on both twins; the calls whose responses differ, with where."""
    settle(a)
    settle(b)
    found = client.get(f"/api/projects/{a.pid}/stages/findings").json().get("artifact") or {}
    fids = sorted(f["id"] for f in found.get("findings") or [])
    left = list(calls(client, a, tmp_path, fids))
    right = list(calls(client, b, tmp_path, fids))
    assert [t for t, _, _ in left] == [t for t, _, _ in right]
    called = {t for t, _, _ in left}
    gets = {t for t, m in mounted(client).items() if "GET" in m} - EXEMPT
    assert gets <= called, gets - called  # every GET route the app mounts, and the previews
    assert "/api/projects/{pid}/preview" in called and len(gets) >= 20
    out = []
    for (template, label, x), (_, _, y) in zip(left, right):
        assert x.status_code < 500, (label, x.text[:400])
        assert y.status_code < 500, (label, y.text[:400])
        diff = list(differences(normalized(body(x), a.pid), normalized(body(y), b.pid)))
        if x.status_code != y.status_code:
            diff.insert(0, f"status {x.status_code} != {y.status_code}")
        if diff:
            out.append(f"{label}: " + "; ".join(diff[:4]))
    return out


def twins(client: Any, tmp_path: Path) -> tuple[Drive, Drive]:
    first, second = twin_files(tmp_path)
    return opened(client, first), opened(client, second)


def get(drive: Drive, route: str, **params: Any) -> Any:
    return drive.c.get(f"/api/projects/{drive.pid}/{route}", params=params)


def outcome_summary(drive: Drive) -> dict[str, Any]:
    [mine] = get(drive, "columns", names=OUTCOME).json()
    return mine


def assert_the_outcome_alone_waits(drive: Drive, line: str) -> None:
    """Before its gate: no distribution of the outcome, its summary only what the outcome card
    shows, and the line saying when it opens."""
    refused = get(drive, f"columns/{OUTCOME}/histogram")
    assert refused.status_code == 409, refused.text[:400]
    assert refused.json()["error"]["code"] == "outcome_not_yet"
    assert refused.json()["error"]["message"] == line
    mine = outcome_summary(drive)
    assert mine["withheld"] == line and mine["record"] is None
    assert [mine[k] for k in ("mean", "std", "q25", "median", "q75")] == [None] * 5
    assert (mine["n"], mine["n_missing"], mine["min"], mine["max"]) == (480, 0, 0, 1)
    other = get(drive, f"columns/{EXPOSURE}/histogram")
    assert other.status_code == 200 and other.json()["record"] is None  # a column alone is open


def assert_the_window_leaves_the_outcome_out(drive: Drive, line: str) -> None:
    for params in ({}, {"columns": OUTCOME}, {"columns": f"{EXPOSURE},{OUTCOME}"}):
        window = get(drive, "table", **params).json()
        assert OUTCOME not in window["columns"], params
        assert window["withheld"] == {OUTCOME: line}, params
    assert get(drive, "table", columns=EXPOSURE).json()["withheld"] == {}


def test_no_route_serves_the_outcome_beside_a_column_with_the_goal_unanswered(client, tmp_path):
    a, b = twins(client, tmp_path)
    leaks = sweep(client, a, b, tmp_path)
    assert not leaks, "\n".join(leaks)
    # An unanswered goal is the strictest case: Estimate's gates, the goal its way out.
    assert_the_outcome_alone_waits(a, outcome_gate.ALONE_AFTER_WHOS_IN)
    assert_the_window_leaves_the_outcome_out(a, outcome_gate.BESIDE_AFTER_FIT)
    exits = get(a, f"columns/{OUTCOME}/histogram").json()["error"]["exits"]
    assert [e["label"] for e in exits] == ["Choose the goal"]


def test_no_route_serves_the_outcome_beside_a_column_under_predict_before_the_draw(client,
                                                                                   tmp_path):
    a, b = twins(client, tmp_path)
    for d in (a, b):
        whos_in(d, "prediction")
    leaks = sweep(client, a, b, tmp_path)
    assert not leaks, "\n".join(leaks)
    assert_the_outcome_alone_waits(a, outcome_gate.AFTER_THE_DRAW)
    assert_the_window_leaves_the_outcome_out(a, outcome_gate.AFTER_THE_DRAW)

    # The draw opens both views, on the training rows: the held-out rows' outcome stays blank,
    # and the sweep now tells the twins apart (its positive control).
    for d in (a, b):
        d.decide({"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5})
    leaks = sweep(client, a, b, tmp_path)
    assert any("/table" in leak for leak in leaks), leaks
    sealed = a.sealed()
    window = get(a, "table", limit=480).json()
    assert window["withheld"] == {OUTCOME: outcome_gate.HELD_OUT_SEALED}
    y = pd.read_csv(CLINICAL)[OUTCOME].tolist()
    at = window["columns"].index(OUTCOME)
    assert [r[at] for r in window["rows"]] == [None if i in sealed else v for i, v in enumerate(y)]
    histogram = get(a, f"columns/{OUTCOME}/histogram").json()
    assert histogram["rows"] == "training" and histogram["record"]["view"] == "distribution"
    assert sum(histogram["counts"]) == 480 - len(sealed)
    mine = outcome_summary(a)
    trained = [v for i, v in enumerate(y) if i not in sealed]
    assert mine["withheld"] is None and mine["rows"] == "training"
    assert mine["mean"] == pytest.approx(np.mean(trained))
    assert mine["n"] == len(trained) and mine["record"] == histogram["record"]
    # The record a client posts on opening it is the brief's own (decision:view_outcome).
    taken = a.post(histogram["record"])
    assert taken.status_code == 200, taken.text[:400]
    record = a.view()["decisions"][-1]["decision"]
    assert (record["kind"], record["view"], record["target"], record["rows"]) == (
        "view_outcome", "distribution", OUTCOME, "training")


def test_no_route_serves_the_outcome_beside_a_column_under_estimate_before_the_lock(client,
                                                                                    tmp_path):
    a, b = twins(client, tmp_path)
    for d in (a, b):
        whos_in(d, "inference")
        d.answer_plan(EXPOSURE)
        d.decide({"kind": "select_models", "models": ["linear"]})
        assert d.view()["state"]["plan_locked"] is None
    leaks = sweep(client, a, b, tmp_path)
    assert not leaks, "\n".join(leaks)
    assert_the_window_leaves_the_outcome_out(a, outcome_gate.BESIDE_AFTER_FIT)
    # The outcome alone opened after Who's in, on the rows kept so far.
    histogram = get(a, f"columns/{OUTCOME}/histogram").json()
    assert histogram["rows"] == "analyzed" and histogram["record"]["view"] == "distribution"
    assert sum(histogram["counts"]) == 480
    assert outcome_summary(a)["mean"] == pytest.approx(pd.read_csv(CLINICAL)[OUTCOME].mean())

    # Fit locks the plan, and the outcome beside a column opens.
    assert a.press_fit()
    assert a.view()["state"]["plan_locked"] is True
    window = get(a, "table", columns=f"{EXPOSURE},{OUTCOME}", limit=480).json()
    assert window["withheld"] == {} and window["columns"] == [EXPOSURE, OUTCOME]
    assert [r[1] for r in window["rows"]] == pd.read_csv(CLINICAL)[OUTCOME].tolist()

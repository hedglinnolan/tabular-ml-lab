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

Before the outcome alone opens, every response is also walked for a statistic of the outcome's
distribution (a mean, a spread, a quartile, a histogram): only the outcome card's fields may be
served then, on every route alike.

Each journey then checks the outcome alone at its own gate: refused with its line before it, and
once open served only after its look is recorded for the rows it reads (the refusal hands back the
``view_outcome`` to post), on the rows kept so far (Estimate) or the training rows (Predict).
After the draw (Predict) and the lock (Estimate) the outcome beside a column opens the same way,
and under Predict the sweep then tells the twins apart: its positive control.
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
# The draw's cells in the split's own preview: which rows are held out, one cell per row. A draw
# stratified by the outcome depends on it only through its balance, and on no other column, so it
# is no statistic of the outcome beside a column; the held-out rows themselves stay sealed. Set
# aside there only: the same cells anywhere else count.
DRAW_CELLS = (".seal.hold", ".seal.unit")
SPLIT_PREVIEW = '"kind": "set_split"'
# The statistics of a distribution: none of the outcome's is served before the outcome alone opens.
SPREAD = ("mean", "std", "q25", "median", "q75")
# Fields that differ between any two projects whatever their data: ids, times, fingerprints.
VOLATILE = {"id", "pid", "key", "at", "created", "created_at", "updated", "updated_at", "modified",
            "fingerprint", "sha256", "decision_id", "job_id", "seq_at", "path", "source_path",
            "started", "finished", "elapsed", "seconds", "duration", "eta", "progress",
            "timestamp", "time", "when", "recorded_at", "hash", "digest", "version_key"}


def twin_files(tmp_path: Path) -> tuple[Path, Path]:
    """``clinical_risk.csv`` as it is, and with its outcome permuted: the same name, so the
    fixture's declared truth answers both. Every row may move, the first ones too, so a route
    quoting the first rows' values (a column's sample) is caught."""
    frame = pd.read_csv(CLINICAL, dtype=str, keep_default_na=False)
    assert (frame[OUTCOME] != "").all()  # no blank outcome: Who's in keeps the same rows in both
    twin = frame.copy()
    twin[OUTCOME] = np.random.default_rng(7).permutation(twin[OUTCOME].to_numpy())
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


def whos_in(drive: Drive, purpose: str, rules: list[dict[str, Any]] | None = None) -> None:
    drive.decide({"kind": "set_purpose", "purpose": purpose})
    drive.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit"})
    drive.reach("roles")
    drive.decide_roles(ROLES)
    drive.answer("exclusions", {"kind": "set_exclusions", "rules": rules or []})
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


def normalized(obj: Any, pid: str, where: str = "", draw: bool = False) -> Any:
    """``obj`` with what differs between any two projects set aside; ``draw``: the response is the
    split's own preview, whose draw cells are set aside too."""
    if isinstance(obj, dict):
        return {k: normalized(v, pid, f"{where}.{k}", draw) for k, v in obj.items()
                if k not in VOLATILE and not k.endswith(("_seconds", "_bytes", "_at"))
                and not (draw and f"{where}.{k}".endswith(DRAW_CELLS))}
    if isinstance(obj, list):
        return [normalized(v, pid, where, draw) for v in obj]
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


def spread_of_the_outcome(obj: Any, where: str = "") -> Iterator[str]:
    """Where ``obj`` holds a statistic of the outcome's distribution: a summary naming it with a
    mean, spread or quartile, a histogram of it, or a distribution view of it."""
    if isinstance(obj, dict):
        if OUTCOME in (obj.get("name"), obj.get("column")):
            for k in SPREAD:
                if obj.get(k) is not None:
                    yield f"{where}.{k}"
            if obj.get("counts") or obj.get("edges") or obj.get("kind") == "distribution":
                yield f"{where} (a histogram)"
        for k, v in obj.items():
            yield from spread_of_the_outcome(v, f"{where}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from spread_of_the_outcome(v, f"{where}[{i}]")


def spread_served(client: Any, drive: Drive, tmp_path: Path) -> list[str]:
    """Every route called on ``drive``; where any serves a statistic of the outcome's
    distribution."""
    settle(drive)
    out = []
    for _, label, response in calls(client, drive, tmp_path, findings(client, drive)):
        if response.status_code == 200:
            out += [f"{label}: {w}" for w in spread_of_the_outcome(body(response))]
    return out


def findings(client: Any, drive: Drive) -> list[str]:
    found = client.get(f"/api/projects/{drive.pid}/stages/findings").json().get("artifact") or {}
    return sorted(f["id"] for f in found.get("findings") or [])


def sweep(client: Any, a: Drive, b: Drive, tmp_path: Path) -> list[str]:
    """Every route called on both twins; the calls whose responses differ, with where."""
    settle(a)
    settle(b)
    fids = findings(client, a)
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
        draw = template == "/api/projects/{pid}/preview" and SPLIT_PREVIEW in label
        diff = list(differences(normalized(body(x), a.pid, draw=draw),
                                normalized(body(y), b.pid, draw=draw)))
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
    assert [mine[k] for k in SPREAD] == [None] * 5
    assert (mine["n"], mine["n_missing"], mine["min"], mine["max"]) == (480, 0, 0, 1)
    other = get(drive, f"columns/{EXPOSURE}/histogram")
    assert other.status_code == 200 and other.json()["rows"] is None  # a column alone is open


def assert_the_window_leaves_the_outcome_out(drive: Drive, line: str) -> None:
    for params in ({}, {"columns": OUTCOME}, {"columns": f"{EXPOSURE},{OUTCOME}"}):
        window = get(drive, "table", **params).json()
        assert OUTCOME not in window["columns"], params
        assert window["withheld"] == {OUTCOME: line} and window["record"] is None, params
    assert get(drive, "table", columns=EXPOSURE).json()["withheld"] == {}


def open_the_distribution(drive: Drive) -> dict[str, Any]:
    """The outcome's distribution as a client opens it: refused until recorded, the record taken
    from the refusal and posted, then served."""
    refused = get(drive, f"columns/{OUTCOME}/histogram")
    assert refused.status_code == 409, refused.text[:400]
    error = refused.json()["error"]
    assert (error["code"], error["message"]) == ("outcome_not_recorded", outcome_gate.NOT_RECORDED)
    [way] = error["exits"]
    record = way["decision"]
    assert outcome_summary(drive)["record"] == record  # the summary hands back the same record
    assert drive.post(record).status_code == 200
    served = get(drive, f"columns/{OUTCOME}/histogram")
    assert served.status_code == 200, served.text[:400]
    return served.json()


def open_the_window(drive: Drive, **params: Any) -> dict[str, Any]:
    """A row window with the outcome as a client opens it once the gate is open: the outcome left
    out until the look is recorded, the record taken from the window and posted."""
    first = get(drive, "table", **params).json()
    assert OUTCOME not in first["columns"]
    assert first["withheld"] == {OUTCOME: outcome_gate.NOT_RECORDED}
    record = first["record"]
    assert (record["kind"], record["view"]) == ("view_outcome", "table")
    assert drive.post(record).status_code == 200
    window = get(drive, "table", **params).json()
    assert OUTCOME in window["columns"] and window["record"] is None
    return window


def views_recorded(drive: Drive, view: str) -> list[dict[str, Any]]:
    return [r["decision"] for r in drive.view()["decisions"]
            if r["decision"]["kind"] == "view_outcome" and r["decision"]["view"] == view]


def test_no_route_serves_the_outcome_beside_a_column_with_the_goal_unanswered(client, tmp_path):
    a, b = twins(client, tmp_path)
    leaks = sweep(client, a, b, tmp_path)
    assert not leaks, "\n".join(leaks)
    # Nor its distribution: only the outcome card's fields, on every route alike.
    spread = spread_served(client, a, tmp_path)
    assert not spread, "\n".join(spread)
    # An unanswered goal is the strictest case: Estimate's gates, the goal its way out.
    assert_the_outcome_alone_waits(a, outcome_gate.ALONE_AFTER_WHOS_IN)
    assert_the_window_leaves_the_outcome_out(a, outcome_gate.BESIDE_AFTER_FIT)
    exits = get(a, f"columns/{OUTCOME}/histogram").json()["error"]["exits"]
    assert [e["label"] for e in exits] == ["Choose the goal"]
    # A finding about the outcome shows none of its values by row, nor its distribution.
    shown = get(a, f"findings/positive_class__{OUTCOME}/evidence").json()
    assert shown["views"] == [] and shown["note"] == outcome_gate.OWN_VIEW


def test_no_route_serves_the_outcome_beside_a_column_under_predict_before_the_draw(client,
                                                                                   tmp_path):
    a, b = twins(client, tmp_path)
    for d in (a, b):
        whos_in(d, "prediction")
    leaks = sweep(client, a, b, tmp_path)
    assert not leaks, "\n".join(leaks)
    spread = spread_served(client, a, tmp_path)
    assert not spread, "\n".join(spread)
    assert_the_outcome_alone_waits(a, outcome_gate.AFTER_THE_DRAW)
    assert_the_window_leaves_the_outcome_out(a, outcome_gate.AFTER_THE_DRAW)

    # The draw opens both views, on the training rows, once each look is recorded: the held-out
    # rows' outcome stays blank, and the sweep now tells the twins apart (its positive control).
    for d in (a, b):
        d.decide({"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5})
        settle(d)
        open_the_window(d, columns=OUTCOME)
    leaks = sweep(client, a, b, tmp_path)
    assert any("/table" in leak for leak in leaks), leaks
    sealed = a.sealed()
    window = get(a, "table", limit=480).json()
    assert window["withheld"] == {OUTCOME: outcome_gate.HELD_OUT_SEALED}
    y = pd.read_csv(CLINICAL)[OUTCOME].tolist()
    at = window["columns"].index(OUTCOME)
    assert [r[at] for r in window["rows"]] == [None if i in sealed else v for i, v in enumerate(y)]
    histogram = open_the_distribution(a)
    assert histogram["rows"] == "training"
    assert sum(histogram["counts"]) == 480 - len(sealed)
    mine = outcome_summary(a)
    trained = [v for i, v in enumerate(y) if i not in sealed]
    assert mine["withheld"] is None and mine["rows"] == "training" and mine["record"] is None
    assert mine["mean"] == pytest.approx(np.mean(trained))
    assert mine["n"] == len(trained)
    # The records are the brief's own (decision:view_outcome), each naming the rows it read.
    [record] = views_recorded(a, "distribution")
    assert (record["target"], record["rows"]) == (OUTCOME, "training") and record["rows_key"]
    [table] = views_recorded(a, "table")
    assert table["rows_key"] == record["rows_key"]


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
    # The outcome alone opened after Who's in, on the rows kept so far, once recorded.
    histogram = open_the_distribution(a)
    assert histogram["rows"] == "analyzed" and sum(histogram["counts"]) == 480
    assert outcome_summary(a)["mean"] == pytest.approx(pd.read_csv(CLINICAL)[OUTCOME].mean())
    # Recorded, it is still the outcome alone: the twins agree.
    open_the_distribution(b)
    leaks = sweep(client, a, b, tmp_path)
    assert not leaks, "\n".join(leaks)

    # Fit locks the plan, and the outcome beside a column opens, once recorded, labeled
    # exploratory.
    assert a.press_fit()
    assert a.view()["state"]["plan_locked"] is True
    window = open_the_window(a, columns=f"{EXPOSURE},{OUTCOME}", limit=480)
    assert window["withheld"] == {} and window["columns"] == [EXPOSURE, OUTCOME]
    assert window["labels"] == {OUTCOME: outcome_gate.EXPLORATORY}
    assert [r[1] for r in window["rows"]] == pd.read_csv(CLINICAL)[OUTCOME].tolist()


def test_the_outcome_is_never_read_on_rows_the_current_draw_has_not_settled(client, tmp_path,
                                                                            monkeypatch):
    """Under Predict the outcome opens only on the current draw's training rows: while the split
    recomputes (a new draw, or the first), nothing of the outcome is served, and a redraw needs a
    new look, so no row the new draw holds out is ever shown (the gate fails closed)."""
    first, _ = twin_files(tmp_path)
    d = opened(client, first)
    whos_in(d, "prediction")
    y = pd.read_csv(CLINICAL)[OUTCOME].tolist()
    service = client.app.state.service

    def shown(window: dict[str, Any]) -> set[int]:
        if OUTCOME not in window["columns"]:
            return set()
        at = window["columns"].index(OUTCOME)
        return {i for i, r in enumerate(window["rows"]) if r[at] is not None}

    windows = []
    d.decide({"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5})
    windows.append(get(d, "table", columns=OUTCOME, limit=480).json())  # the split computing
    settle(d)

    # Whatever the timing, a split not fresh is no seal known: the outcome waits for it, though
    # an earlier artifact of the split is on disk.
    real = service.engine.status

    def computing(pid: str) -> dict[str, Any]:
        stages = dict(real(pid))
        stages["split"] = stages["split"].model_copy(update={"status": "running"})
        return stages

    with monkeypatch.context() as m:
        m.setattr(service.engine, "status", computing)
        window = get(d, "table", columns=OUTCOME, limit=480).json()
        assert OUTCOME not in window["columns"]
        assert window["withheld"] == {OUTCOME: outcome_gate.BEING_DRAWN}
        refused = get(d, f"columns/{OUTCOME}/histogram").json()["error"]
        assert refused["message"] == outcome_gate.BEING_DRAWN
        points = get(d, "stages/explore").json()["artifact"] or {}
        for f in points.get("findings") or []:
            if f.get("kind") == "outcome_relationship":
                assert f["points"] == [] and f["detail"] == outcome_gate.BEING_DRAWN

    window = open_the_window(d, columns=OUTCOME, limit=480)
    assert shown(window) and not shown(window) & d.sealed()
    open_the_distribution(d)

    # A redraw: while it computes, and once settled until the new look is recorded, the outcome is
    # left out; the old draw's training rows are never read for the new one.
    d.decide({"kind": "set_split", "holdout": 0.3, "seed": 5, "folds": 5})
    windows.append(get(d, "table", columns=OUTCOME, limit=480).json())
    histogram = get(d, f"columns/{OUTCOME}/histogram")
    assert histogram.status_code == 409
    settle(d)
    sealed = d.sealed()
    assert len(sealed) > 96
    windows.append(get(d, "table", columns=OUTCOME, limit=480).json())
    assert windows[-1]["withheld"] == {OUTCOME: outcome_gate.NOT_RECORDED}
    for w in windows:
        assert not shown(w) & sealed, w["withheld"]
    window = open_the_window(d, columns=OUTCOME, limit=480)
    assert shown(window) == set(range(480)) - sealed
    assert [r[0] for r in window["rows"]] == [None if i in sealed else v for i, v in enumerate(y)]


def test_a_split_recorded_under_estimate_is_no_draw_under_predict(client, tmp_path):
    """Switching to Predict after Who's in under Estimate keeps the split TurboTab recorded there
    (no rows held out): no draw the person made. The held-out rows' question opens again, and the
    outcome's views, the explore stage's relationship points with them, wait for the draw."""
    a, b = twins(client, tmp_path)
    for d in (a, b):
        whos_in(d, "inference")
        assert d.view()["state"]["split"]["holdout"] == 0.0
        d.decide({"kind": "set_purpose", "purpose": "prediction"})
        assert d.view()["state"]["split"] is not None
    leaks = sweep(client, a, b, tmp_path)
    assert not leaks, "\n".join(leaks)
    assert_the_outcome_alone_waits(a, outcome_gate.AFTER_THE_DRAW)
    assert_the_window_leaves_the_outcome_out(a, outcome_gate.AFTER_THE_DRAW)
    lines = [line for stage in client.get(f"/api/projects/{a.pid}/quest").json()["stages"]
             for line in stage["lines"] if line["key"] == "split"]
    assert [line["status"] for line in lines] == ["open"]
    explore = get(a, "stages/explore").json()["artifact"] or {}
    for f in explore.get("findings") or []:
        if f.get("kind") == "outcome_relationship":
            assert f["points"] == [] and f["detail"] == outcome_gate.AFTER_THE_DRAW

    # The person's own draw under Predict opens them.
    a.decide({"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5})
    settle(a)
    assert open_the_distribution(a)["rows"] == "training"


def test_each_look_at_the_outcome_is_recorded_for_the_rows_it_reads(client, tmp_path):
    """Under Estimate the outcome alone reads the rows kept so far, which an answer still open
    before the lock can move (an exclusion re-answered): each set of rows is its own look, served
    only once recorded, so every look the person took is in the log as a forking path. After the
    lock the outcome beside a column reads only the rows analyzed, labeled exploratory."""
    first, _ = twin_files(tmp_path)
    d = opened(client, first)
    young = [{"column": "age", "low": 18, "high": 60, "reason": "60 or under"}]
    older = [{"column": "age", "low": 61, "high": 120, "reason": "over 60"}]
    whos_in(d, "inference", young)
    settle(d)
    histogram = open_the_distribution(d)
    assert sum(histogram["counts"]) == 137 and histogram["counts"][-1] == 21

    d.decide({"kind": "set_exclusions", "rules": older})
    settle(d)
    histogram = open_the_distribution(d)  # refused until this set of rows is recorded too
    assert sum(histogram["counts"]) == 343 and histogram["counts"][-1] == 91
    looks = views_recorded(d, "distribution")
    assert [look["n_rows"] for look in looks] == [137, 343]
    assert len({look["rows_key"] for look in looks}) == 2

    d.answer_plan(EXPOSURE)
    d.decide({"kind": "select_models", "models": ["linear"]})
    assert d.press_fit()
    window = open_the_window(d, columns=f"age,{OUTCOME}", limit=480)
    age = pd.read_csv(CLINICAL)["age"].tolist()
    assert all((r[1] is None) == (a <= 60) for r, a in zip(window["rows"], age))
    assert window["withheld"] == {OUTCOME: outcome_gate.NOT_ANALYZED}
    assert window["labels"] == {OUTCOME: outcome_gate.EXPLORATORY}

"""P8 (W1 part 2 ruling 9) · GET …/methods/sections: the methods in the export bundle's own
sections and words, so the manuscript rail and the downloaded methods can never differ.

Under prediction the sections are TRIPOD+AI's, under inference the bundle's own order (STROBE's
with STROBE-nut's); each line carries its decision record's id, and ``count`` counts the lines. The
references: the bundle's ``methods.json`` from the zip ``GET …/export`` hands over, for a project
taken to its export; the record's own decisions (their ids and kinds) and the section each kind
declares (``export.methods.SECTION_OF``), for projects the export still refuses. The projects are
the synthetic diet table of ``core/tests/acceptance/test_export.py``, opened through the real
server.
"""
from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path
from typing import Any

import pytest

from turbotab.core.export import methods as M
from turbotab.core.tests.acceptance.server_drive import local_server
from turbotab.core.tests.acceptance.test_export import _diet, _open_diet


def _sections(c: Any, pid: str) -> dict[str, Any]:
    r = c.get(f"/api/projects/{pid}/methods/sections")
    assert r.status_code == 200, r.text[:600]
    return r.json()


def _wait_fresh(drive: Any, stages: tuple[str, ...], timeout: float = 300.0) -> None:
    end = time.monotonic() + timeout
    while True:
        statuses = drive.view()["stages"]
        if all(statuses[s]["status"] == "fresh" for s in stages):
            return
        for s in stages:
            assert statuses[s]["status"] != "error", statuses[s]
        assert time.monotonic() < end, {s: statuses[s]["status"] for s in stages}
        time.sleep(0.1)


def _check_grouping(served: dict[str, Any], view: dict[str, Any], purpose: str) -> None:
    """The guideline the purpose names; every section in that guideline's order; each record line
    in the section its decision's kind declares, with that decision's id; the lines counted."""
    guideline = "TRIPOD+AI" if purpose == "prediction" else "STROBE-nut"
    layout = M.TRIPOD_SECTIONS if purpose == "prediction" else M.STROBE_SECTIONS
    order = [key for key, _, _ in layout] + ["after", "reproducibility"]
    assert served["guideline"] == guideline
    keys = [s["key"] for s in served["sections"]]
    assert keys == sorted(keys, key=order.index) and keys[-1] == "reproducibility"
    kinds = {d["id"]: d["decision"]["kind"] for d in view["decisions"]}
    which = 1 if purpose == "prediction" else 0
    lines = [(s["key"], e) for s in served["sections"] for e in s["entries"]]
    recorded = [(key, e) for key, e in lines if e["source"] == "record"]
    assert recorded, served
    for key, e in recorded:
        assert kinds[e["record_id"]] == e["kind"], e
        if key != "after" and e["kind"] not in M.EVENTS:
            assert key == M.SECTION_OF.get(e["kind"], ("other", "other"))[which], (key, e["kind"])
    assert all(e["record_id"] is None for _, e in lines if e["source"] != "record")
    assert served["count"] == len(lines)


def test_an_unknown_project_has_no_methods_sections(client):
    r = client.get("/api/projects/p0000000000/methods/sections")
    assert r.status_code == 404 and r.json()["error"]["code"] == "unknown_project"


@pytest.mark.parametrize("purpose", ["prediction", "inference"])
def test_the_sections_are_the_purposes_guidelines_while_the_export_still_refuses(
        client, tmp_path: Path, purpose: str):
    """Before any fit, the export refuses; the route serves the methods the bundle would hold, in
    TRIPOD+AI's sections under prediction and STROBE's under inference (the guard both ways)."""
    from turbotab.core.export.bundle import contents

    csv = tmp_path / "diet.csv"
    _diet().to_csv(csv, index=False)
    drive = _open_diet(client, csv, purpose)
    assert client.get(f"/api/projects/{drive.pid}/export").status_code == 409
    served = _sections(client, drive.pid)
    _check_grouping(served, drive.view(), purpose)
    service = client.app.state.service
    built = contents(service.export_source(drive.pid), gated=False).methods
    assert served["sections"] == [s.model_dump(mode="json") for s in built.sections]


# About 30 s on a developer machine: the diet table's prediction journey to its export.
def test_the_sections_are_the_exported_bundles_methods_word_for_word(tmp_path: Path):
    """The diet table under prediction, a fifth of its rows held out, the linear model fitted,
    Fit pressed, the final model declared and the held-out rows opened, then exported: the route
    serves the zip's ``methods.json`` sections exactly (keys, titles, items, every line's words,
    source, kind and record id), before the export and after it."""
    csv = tmp_path / "diet.csv"
    _diet().to_csv(csv, index=False)
    with local_server(tmp_path / "home") as client:
        drive = _open_diet(client, csv, "prediction")
        drive.answer("split", {"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5})
        drive.answer("energy_adjustment", {"kind": "set_energy_adjustment", "method": "none"})
        drive.answer("models", {"kind": "select_models", "models": ["linear"]})
        _wait_fresh(drive, ("cohort", "design", "fit"))
        assert drive.press_fit()
        refused = client.get(f"/api/projects/{drive.pid}/export")
        assert refused.status_code == 409, refused.text[:400]
        opening = next(e["decision"] for e in refused.json()["error"]["exits"]
                       if (e["decision"] or {}).get("family") == "linear")
        drive.decide(opening)
        _wait_fresh(drive, ("cohort", "design", "fit"))
        before = _sections(client, drive.pid)
        r = client.get(f"/api/projects/{drive.pid}/export")
        assert r.status_code == 200, r.text[:600]
        after = _sections(client, drive.pid)
        view = drive.view()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        bundle = json.loads(z.read("turbotab-export/methods.json"))
    for served in (before, after):
        assert served["sections"] == bundle["sections"]
        assert (served["guideline"], served["seen_from"]) == (bundle["guideline"],
                                                              bundle["seen_from"])
        _check_grouping(served, view, "prediction")
    # The opening closes a prediction's methods, last among the analysis's lines, with its record id.
    analysis = next(s for s in before["sections"] if s["key"] == "analysis")
    opened = next(d["id"] for d in reversed(view["decisions"])
                  if d["decision"]["kind"] == "open_seal")
    assert analysis["entries"][-1]["kind"] == "open_seal"
    assert analysis["entries"][-1]["record_id"] == opened

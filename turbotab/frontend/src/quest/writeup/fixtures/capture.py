"""Captures what Write-up reads, as the server serves it, for two small projects built by hand: a
prediction (two families compared on every row, so the result is the selection-corrected score)
and an estimate (Table 2, its appendix with no adjustment term to list, so its file says so in
place of rows, and its sensitivity; the plan locked, one decision made after the estimates were
seen). Each project's decisions go through the real decision log, and
its stage artifacts (the fit or the effects, the participant flow, the lineage) through the real
export (``turbotab.core.export.bundle``):

* ``zip``: the bytes GET …/export serves (``bundle.contents(source).zip()``), base64;
* ``checklist``: GET …/checklist (``bundle.live_checklist``), with every fitted family's scores
  shown, as the Results screen shows them;
* ``plan``: GET …/plan (``plan_lock.plan_export``);
* ``waiting``: the same project while its result stage is still computing: the live checklist then
  (it names what the export waits for) and the 409 body the export answers with
  (``gate.refusal``, as ``server/errors.py`` serves a refusal).

``truth`` is read here independently of the frontend's readers: each file's name, as the zip
stores it, and its SHA-256 from Python's ``zipfile``; each results table's title, caption, header
and cells from the engine's ``Table`` models (the cells as the Markdown writes them); each figure's
title and description from its SVG by ``xml.etree``; each checklist item's heading (its label and
topic) from the bundle's own checklist file.

The artifacts are made by hand to exercise the reading, not to agree with one another as one
analysis would (the participant flow's 90 rows and Table 2's 3 are different inputs).

    OMP_NUM_THREADS=2 TURBOTAB_WORKERS=2 scripts/py turbotab/frontend/src/quest/writeup/fixtures/capture.py

writes ``bundle-predict.json`` and ``bundle-estimate.json`` beside this file; nothing in them is
written by hand.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import sys
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))  # the repository's root: `turbotab`

from turbotab.core import decisions as d  # noqa: E402
from turbotab.core import plan_lock  # noqa: E402
from turbotab.core.decisions import DecisionLog  # noqa: E402
from turbotab.core.export import bundle, gate, tables  # noqa: E402
from turbotab.core.tests import test_export as TE  # noqa: E402

HERE = Path(__file__).resolve().parent
SVG = "{http://www.w3.org/2000/svg}"

COHORT = {"n_final": 90, "steps": [
    {"key": "loaded", "label": "Rows in the table", "n": 100, "dropped": 0},
    {"key": "missing", "label": "Complete cases", "n": 90, "dropped": 10,
     "reason": "a value missing"}]}
DESIGN = {"matrix": {"n_rows": 90, "n_cols": 2}, "lineage": {"nodes": [
    {"id": "raw:fiber", "column": "fiber", "lane": "raw", "label": "fiber"},
    {"id": "raw:age", "column": "age", "lane": "raw", "label": "age"},
    {"id": "mx:fiber", "column": "fiber", "lane": "matrix", "label": "fiber"},
    {"id": "mx:age", "column": "age", "lane": "matrix", "label": "age"}], "links": [
    {"source": "raw:fiber", "target": "mx:fiber", "operation": "kept"},
    {"source": "raw:age", "target": "mx:age", "operation": "kept"}]}}
# Table 2's effects with no adjustment term to list, so the bundle writes its appendix with no rows.
EFFECTS = {**TE.EFFECTS, "families": [{**f, "appendix": []} for f in TE.EFFECTS["families"]]}
FIT = {"primary_metric": "rmse", "metric_labels": {"rmse": "RMSE", "r2": "R²"},
       "n_train": 90, "n_holdout": 0,
       "models": [
           {"family": "linear", "label": "Linear model", "cv": {
               "rmse": {"estimate": 12.5, "ci_low": 11.0, "ci_high": 14.0, "se": 0.7},
               "r2": {"estimate": 0.21}}},
           {"family": "elastic_net", "label": "Elastic net", "cv": {
               "rmse": {"estimate": 12.25, "ci_low": 10.9, "ci_high": 13.6, "se": 0.68},
               "r2": {"estimate": 0.24}}}],
       "result": {"basis": "selection_corrected", "family": "elastic_net", "metric": "rmse",
                  "estimate": 12.75, "ci_low": 11.2, "ci_high": 14.3,
                  "sentence": "The declared result is the selection-corrected RMSE of the elastic "
                              "net."}}


def _log(folder: Path, purpose: str) -> DecisionLog:
    log = DecisionLog(folder / "decisions.jsonl")
    for x in (d.SetLens(lenses=["dietary"]), d.SetTarget(column="glucose"),
              d.SetTask(column="glucose", task="regression"), d.SetPurpose(purpose=purpose),
              d.SetRoles(roles={"fiber": "exposure", "age": "covariate"}),
              d.SetMissing(strategy="complete_case"),
              d.SetSplit(holdout=0.0, seed=0, folds=5)):
        log.append(x, sentence=TE._say)
    if purpose == "prediction":
        log.append(d.SelectModels(models=["linear", "elastic_net"]), sentence=TE._say)
        return log
    log.append(d.SelectModels(models=["linear"]), sentence=TE._say)
    log.append(d.validate({"kind": "lock_plan"}, {"state": log.state()}), sentence=TE._say)
    log.append(d.SetExclusions(rules=[d.ExclusionRule(column="age", low=18, reason="adults")]),
               sentence=TE._say, after_estimates=True)
    return log


def _statuses(computing: str | None) -> dict[str, Any]:
    return {s: SimpleNamespace(status="running" if s == computing else "fresh", key=f"k-{s}",
                               error=None, missing=[], refusal=None)
            for s in ("cohort", "design", "fit", "effects")}


def _source(log: DecisionLog, purpose: str, *, computing: str | None = None) -> Any:
    artifacts = {"cohort": COHORT, "design": DESIGN,
                 **({"fit": FIT} if purpose == "prediction" else {"effects": EFFECTS})}
    return TE._source(log.records(), purpose=purpose, artifacts=artifacts,
                      statuses=_statuses(computing))


def _figure(svg: bytes) -> dict[str, str]:
    root = ET.fromstring(svg)
    return {"title": root.findtext(f"{SVG}title") or "", "desc": root.findtext(f"{SVG}desc") or ""}


def _table(t: Any, files: dict[str, bytes]) -> dict[str, Any]:
    """A table's words from its model; with no rows, the line its file says instead (read off the
    file: the model holds none)."""
    headers = tables.headers_for(t)
    kinds = {c.key: c.kind for c in t.columns}
    [md] = [b.decode("utf-8") for n, b in files.items() if n.endswith(f"/results/{t.name}.md")]
    return {"file": f"results/{t.name}.md", "title": t.title, "caption": t.caption,
            "header": [headers[s] for s in t.shown] if t.rows else [],
            "rows": [[tables._cell(row, s, kinds) for s in t.shown] for row in t.rows],
            "empty": None if t.rows else md.strip().splitlines()[-1].strip("_")}


def capture(purpose: str) -> dict[str, Any]:
    folder = Path(tempfile.mkdtemp())
    log = _log(folder, purpose)
    source = _source(log, purpose)
    assert gate.missing(source) == [], [m.message for m in gate.missing(source)]
    made = bundle.contents(source)  # gated, as the route builds it
    served = made.zip()
    with zipfile.ZipFile(io.BytesIO(served)) as z:
        files = {n: z.read(n) for n in z.namelist()}
    [stem] = [n for n in files if n.endswith(".md") and "/checklist/" in n]
    # Each item's heading in the bundle's checklist file: its label (TRIPOD+AI's with its scope,
    # STROBE's with its table) and its topic, in the items' order.
    headings = [line[4:] for line in files[stem].decode("utf-8").splitlines()
                if line.startswith("### ")]
    families = [m["family"] for m in FIT["models"]] if purpose == "prediction" else []
    checklist = bundle.live_checklist(source, families).model_dump(mode="json")
    computing = "fit" if purpose == "prediction" else "effects"
    early = _source(log, purpose, computing=computing)
    found = gate.missing(early)
    assert found, "the export waits for the result stage"
    return {
        "purpose": purpose,
        "zip": base64.b64encode(served).decode("ascii"),
        "checklist": checklist,
        "plan": json.loads(plan_lock.plan_export(log.records())),
        "waiting": {"checklist": bundle.live_checklist(early, families).model_dump(mode="json"),
                    "refusal": gate.refusal(found).to_dict()},
        "truth": {
            "files": {n: hashlib.sha256(b).hexdigest() for n, b in sorted(files.items())},
            "tables": [_table(t, files) for t in sorted(made.tables, key=lambda t: t.name)],
            "figures": {n: _figure(b) for n, b in sorted(files.items()) if n.endswith(".svg")},
            "checklist_headings": headings,
        },
    }


def main() -> None:
    for purpose, name in (("prediction", "bundle-predict.json"),
                          ("inference", "bundle-estimate.json")):
        out = capture(purpose)
        (HERE / name).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", "utf-8")
        print(name, len(out["truth"]["files"]), "files,", len(out["truth"]["tables"]), "tables")


if __name__ == "__main__":
    main()

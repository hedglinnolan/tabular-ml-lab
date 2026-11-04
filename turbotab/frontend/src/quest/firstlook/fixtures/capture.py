"""What the server serves First look's gallery, captured at each of the outcome's gates
(turbotab/core/outcome_gate.py; calm/FOUNDATION §5 rule 6), for the gallery's tests
(src/quest/firstlook/firstlook.test.tsx). Writes ``served.json`` beside this file.

The journeys are the outcome gate's own (turbotab/server/tests/test_outcome_gate_api.py): the
clinical sample (``turbotab/sample_data/clinical_risk.csv``, outcome ``readmit_30d``) driven
through the real HTTP API in-process.

* Predict: First look before Who's in (the outcome's gate shut until the draw), with the draw
  posted with ``decide_now`` then (refused: the held-out rows are drawn last); after the draw
  (each outcome view open, its look not recorded: every route hands back the record to post);
  after each record a route handed back is posted (the distribution's, one relationship's, the
  row window's); then one lever option posted with ``decide_now``.
* Estimate: after Who's in (the outcome alone open, its look not recorded; the outcome beside a
  column and Explore's relationships waiting for the lock).

Reading records nothing; each record posted is the one a route served, as the gallery posts it.
Run from the repository root (about a minute):

    PYTHONPATH=. OMP_NUM_THREADS=2 TURBOTAB_WORKERS=2 scripts/py \\
        turbotab/frontend/src/quest/firstlook/fixtures/capture.py
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("TURBOTAB_CLAIMS_STRICT", "1")

from turbotab.core.tests.acceptance.server_drive import local_server  # noqa: E402
from turbotab.server.tests import test_outcome_gate_api as gate  # noqa: E402

OUT = Path(__file__).resolve().parent / "served.json"
ROWS = 8  # the row window the gallery reads (firstlook/model.ts ROWS_SHOWN)
BESIDE = 11  # the columns beside the outcome in it (firstlook/model.ts COLUMNS_BESIDE)
BINS = 20  # the gallery's histograms (firstlook/data.ts BINS)


def get(drive: Any, path: str, **query: Any) -> dict[str, Any]:
    r = drive.c.get(f"/api/projects/{drive.pid}/{path}", params=query)
    return {"status": r.status_code, "body": r.json()}


def window_columns(columns: list[dict[str, Any]]) -> list[str]:
    others = [c["name"] for c in columns if c["name"] != gate.OUTCOME]
    return [gate.OUTCOME, *others[:BESIDE]]


def moment(drive: Any) -> dict[str, Any]:
    """Every read the gallery makes, as served now: the view and the quest log, the columns, the
    explore and target_info stages, the row window, and each number column's histogram (the
    outcome's among them, refused until its look is recorded)."""
    columns = get(drive, "columns")
    names = window_columns(columns["body"])
    numbers = [c["name"] for c in columns["body"] if c["dtype"] in ("numeric", "integer")]
    return {
        "view": drive.view(),
        "quest": get(drive, "quest")["body"],
        "columns": columns,
        "explore": get(drive, "stages/explore"),
        "target_info": get(drive, "stages/target_info"),
        "table": get(drive, "table", offset=0, limit=ROWS, columns=",".join(names)),
        "histograms": {name: get(drive, f"columns/{name}/histogram", bins=BINS) for name in numbers},
    }


def post(drive: Any, body: dict[str, Any], **query: Any) -> dict[str, Any]:
    r = drive.c.post(f"/api/projects/{drive.pid}/decisions", json=body, params=query)
    answer = r.json()
    if r.status_code == 200:  # the view: only the record it added is kept
        answer = {"recorded": answer["decisions"][-1]}
    return {"status": r.status_code, "body": answer}


def outcome_summary(m: dict[str, Any]) -> dict[str, Any]:
    return next(c for c in m["columns"]["body"] if c["name"] == gate.OUTCOME)


def predict(client: Any, folder: Path) -> dict[str, Any]:
    first, _ = gate.twin_files(folder)
    d = gate.opened(client, first)
    d.decide({"kind": "set_purpose", "purpose": "prediction"})
    gate.settle(d)
    out: dict[str, Any] = {"first_visit": moment(d)}
    # "Decide now" refused: the held-out rows are drawn last, so not while Who's in is open
    # (core/sequence.py `decide_now_refusal`).
    draw = {"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5}
    out["decide_now_refused"] = {"decision": draw, "answer": post(d, draw, decide_now="true")}
    assert out["decide_now_refused"]["answer"]["status"] == 409, out["decide_now_refused"]

    d.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit"})
    d.reach("roles")
    d.decide_roles(gate.ROLES)
    d.answer("exclusions", {"kind": "set_exclusions", "rules": []})
    d.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
    d.reach("split")
    d.decide({"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5})
    gate.settle(d)
    drawn = moment(d)
    out["drawn"] = drawn

    explore = drawn["explore"]["body"]["artifact"]
    relationship = next(f for f in explore["findings"] if f["kind"] == "outcome_relationship")
    posts = {
        "distribution": post(d, outcome_summary(drawn)["record"]),
        "relationship": post(d, relationship["record"]),
        "table": post(d, drawn["table"]["body"]["record"]),
    }
    gate.settle(d)
    out["posts"] = posts
    out["recorded"] = moment(d)

    lever = next(f for f in explore["findings"] if f["lever"] and any(
        o["decision"] for o in f["lever"]["options"]))
    option = next(o for o in lever["lever"]["options"] if o["decision"])
    out["decide_now"] = {"finding": lever["id"], "option": option["key"],
                         "answer": post(d, option["decision"], decide_now="true")}
    gate.settle(d)
    return out


def estimate(client: Any, folder: Path) -> dict[str, Any]:
    first, _ = gate.twin_files(folder)
    d = gate.opened(client, first)
    gate.whos_in(d, "inference")
    gate.settle(d)
    return {"whos_in": moment(d)}


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with local_server(root / "home") as client:
            (root / "predict").mkdir()
            (root / "estimate").mkdir()
            served = {
                "meta": {"source": "turbotab/sample_data/clinical_risk.csv", "outcome": gate.OUTCOME,
                         "script": "turbotab/frontend/src/quest/firstlook/fixtures/capture.py",
                         "rows": ROWS, "beside": BESIDE, "bins": BINS},
                "predict": predict(client, root / "predict"),
                "estimate": estimate(client, root / "estimate"),
            }
    OUT.write_text(json.dumps(served, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

"""Re-record the routing value check's result, with provenance.

The only writer of `docs/turbotab/data/routing-value-check.json`. It is a
script and not a test for the reason `T0-PREREG-003` records: the file's
`passes_under_literal_reading: false` is called a permanent record in
`T0-PREREG-001`'s note, and while the suite rewrote it on every run it was
permanent only for as long as the code that computed it kept computing it.

The suite now **compares** and fails on divergence. This **re-records**, and
refuses to overwrite silently: `--out` is required, and pointing it at the
recorded result needs `--replace` plus a stated reason, which is written into
the file so the replacement carries its own justification.

`VALUE_CHECK_ADJUDICATION.md` is the authority. This file is its evidence.

Usage
-----
    turbotab/.venv/Scripts/python scripts/rerecord_routing_value_check.py \
        --out docs/turbotab/data/routing-value-check-l9c.json \
        --measured-at <commit>

    # replacing the recorded result, which needs a reason on the record
    ... --out docs/turbotab/data/routing-value-check.json --replace \
        --reason "adjudicated in VALUE_CHECK_ADJUDICATION.md ..."
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RECORDED = ROOT / "docs" / "turbotab" / "data" / "routing-value-check.json"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--measured-at", default=None,
                        help="the commit this result was computed at.")
    parser.add_argument("--replace", action="store_true",
                        help="allow writing over the recorded result.")
    parser.add_argument("--reason", default="",
                        help="why the recorded result is being replaced. "
                             "Written into the file.")
    args = parser.parse_args(argv)

    out = args.out.resolve()
    if out == RECORDED.resolve() and not args.replace:
        parser.error(
            f"{args.out} is the recorded result. Write the new one beside it, "
            "or pass --replace with --reason and adjudicate in "
            "docs/turbotab/VALUE_CHECK_ADJUDICATION.md.")
    if args.replace and not args.reason:
        parser.error("--replace requires --reason; a replacement with no stated "
                     "cause is the drift this script exists to stop.")
    if out.exists() and not args.replace:
        parser.error(f"{args.out} already exists. Pick a path that does not.")

    from tests.integration.test_routing_value_check import (
        ADJUDICATED, ADJUDICATED_AT, BASELINE, DATASETS, PREREG_AT,
        _asked_coverage, _ratio_from, _run_guided, _surfaced_coverage)
    from turbotab import measure

    baseline = {m["dataset"]: m for m in measure.read_baseline(ADJUDICATED)}
    frozen = {m["dataset"]: m for m in measure.read_baseline(BASELINE)}

    rows, failures, strict = [], [], []
    for name, path, target in DATASETS:
        g = _run_guided(name, path, target)
        gm = g.to_dict(args.measured_at)["metrics"]
        surfaced, asked = _surfaced_coverage(g), _asked_coverage(g)
        n_now = len(g.required)
        frozen_keys = [r["key"] for r in frozen[name]["required"]]
        n_frozen = len(frozen_keys)
        raised = {q.covers for q in g.questions if q.covers}
        raised_asked = {q.covers for q in g.questions if q.covers and not q.skipped}
        sf = sum(1 for k in frozen_keys if k in raised) / n_frozen
        af = sum(1 for k in frozen_keys if k in raised_asked) / n_frozen
        stamp = args.measured_at or ADJUDICATED_AT

        rows.append({
            "dataset": name,
            "classic": {**baseline[name]["metrics"],
                        "coverage_ratio": _ratio_from(baseline[name], ADJUDICATED_AT)},
            "classic_frozen": {**frozen[name]["metrics"],
                               "coverage_ratio": _ratio_from(frozen[name], PREREG_AT)},
            "guided": {**gm, "surfaced_coverage": round(surfaced, 4),
                       "asked_coverage": round(asked, 4),
                       "surfaced_ratio": f"{round(surfaced * n_now)}/{n_now} @ {stamp}",
                       "asked_ratio": f"{round(asked * n_now)}/{n_now} @ {stamp}"},
            "guided_under_frozen_denominator": {
                "n_required": n_frozen,
                "surfaced_coverage": round(sf, 4), "asked_coverage": round(af, 4),
                "surfaced_ratio": f"{round(sf * n_frozen)}/{n_frozen} @ {PREREG_AT}",
                "asked_ratio": f"{round(af * n_frozen)}/{n_frozen} @ {PREREG_AT}",
            },
        })
        dc = gm["deferral_closes"]
        possible = len([r for r in g.required if r.key.startswith("repair::")]) > 1
        if dc is not None and abs(dc - 1.0) > 1e-9:
            failures.append(f"{name}: deferral_closes = {dc}, must be exactly 1.0")
        elif dc is None and possible:
            failures.append(f"{name}: deferrals were possible but none occurred")
        elif dc is None:
            strict.append(f"{name}: deferral_closes = None (nothing deferrable on "
                          "this dataset; fails only under the literal reading of "
                          "the prereg)")

    payload = {
        "schema_version": "1.0", "prereg": "VALUE_CHECK_PREREG.md",
        "adjudication": "VALUE_CHECK_ADJUDICATION.md",
        "measured_at": args.measured_at,
        "classic_reference": ADJUDICATED.name,
        "classic_frozen_reference": BASELINE.name,
        "rows": rows,
        "verdict": {
            "passes": not failures, "failures": failures,
            "literal_reading_only_failures": strict,
            "passes_under_literal_reading": not (failures or strict),
        },
    }
    if args.reason:
        payload["replaced_because"] = args.reason

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    print(f"  passes={payload['verdict']['passes']} "
          f"passes_under_literal_reading="
          f"{payload['verdict']['passes_under_literal_reading']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

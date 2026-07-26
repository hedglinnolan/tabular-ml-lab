#!/usr/bin/env python3
"""Ledger tooling for the TurboTab transition.

`data/findings.json` is the source of truth. `FINDINGS_LEDGER.md` is generated
from it and should never be hand-edited — edit the JSON, then regenerate.

Usage
-----
    python docs/turbotab/tools/ledger.py stats
    python docs/turbotab/tools/ledger.py next --n 15
    python docs/turbotab/tools/ledger.py next --n 15 --area STATE
    python docs/turbotab/tools/ledger.py set FIND-ID --status FIXED --note "..." --test test_foo
    python docs/turbotab/tools/ledger.py regen
    python docs/turbotab/tools/ledger.py check          # exits 1 if the schema is violated

Status values
-------------
    UNVERIFIED   not yet re-checked against the current baseline (the default)
    OPEN         confirmed to still exist
    PARTIAL      partly addressed; `note` must say what remains
    FIXED        no longer reproducible; `test` must name a regression test
    NOT-A-DEFECT the finding was wrong; `note` must say why
    WONTFIX      real but deliberately not fixed; `note` must give the reason
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "findings.json"
OUT = ROOT / "FINDINGS_LEDGER.md"

TERMINAL = {"FIXED", "NOT-A-DEFECT", "WONTFIX"}
VALID = {"UNVERIFIED", "OPEN", "PARTIAL"} | TERMINAL

AREA_NAME = {
    "TIER0": "Verified against main",
    "STATE": "Application state / lockbox",
    "CONTRACT": "Stage-boundary contracts",
    "MINE": "Silent-failure landmines",
    "SWEEP": "Completeness sweep",
    "TEST": "Migration safety net",
    "COACH": "Coach to Router",
    "MODELS": "Models / training / eval",
    "RECORD": "Record / narrative / export",
    "PAGES": "Page-layer extraction",
    "PREP": "Features / preprocessing",
    "MISC": "Other",
}
SEV_ORDER = {"critical": 0, "landmine": 1, "high": 2, "invariant": 3, "medium": 4, "low": 5}


def load() -> list[dict]:
    return json.loads(DATA.read_text())


def save(rows: list[dict]) -> None:
    DATA.write_text(json.dumps(rows, indent=1))


def norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


# ---------------------------------------------------------------- commands
def cmd_stats(rows, _args) -> int:
    by_status = Counter(r["status"] for r in rows)
    done = sum(by_status[s] for s in TERMINAL)
    print(f"total {len(rows)}   closed {done}   remaining {len(rows) - done}")
    print()
    for s in ["UNVERIFIED", "OPEN", "PARTIAL", "FIXED", "NOT-A-DEFECT", "WONTFIX"]:
        if by_status.get(s):
            print(f"  {s:14} {by_status[s]:4}")
    print()
    unver = [r for r in rows if r["status"] == "UNVERIFIED"]
    if unver:
        print("UNVERIFIED by area:")
        for a, n in Counter(r["area"] for r in unver).most_common():
            print(f"  {AREA_NAME.get(a, a):32} {n:4}")
    return 0


def cmd_next(rows, args) -> int:
    """Emit the next batch to work, highest severity first, as JSON."""
    pool = [r for r in rows if r["status"] == "UNVERIFIED"]
    if args.area:
        pool = [r for r in pool if r["area"] == args.area.upper()]
    pool.sort(key=lambda r: SEV_ORDER.get(r["sev"], 9))
    print(json.dumps(pool[: args.n], indent=1))
    return 0


def cmd_set(rows, args) -> int:
    hits = [r for r in rows if r["id"] == args.id]
    if not hits:
        print(f"no such finding: {args.id}", file=sys.stderr)
        return 1
    r = hits[0]
    status = args.status.upper()
    if status not in VALID:
        print(f"invalid status {status}; valid: {sorted(VALID)}", file=sys.stderr)
        return 1
    if status == "FIXED" and not args.test:
        print("FIXED requires --test naming a regression test", file=sys.stderr)
        return 1
    if status in {"PARTIAL", "NOT-A-DEFECT", "WONTFIX"} and not args.note:
        print(f"{status} requires --note", file=sys.stderr)
        return 1
    r["status"] = status
    if args.note:
        r["note"] = norm(args.note)
    if args.test:
        r["test"] = norm(args.test)
    if args.evidence:
        r["verified_ev"] = norm(args.evidence)
    save(rows)
    print(f"{args.id} -> {status}")
    return 0


def cmd_check(rows, _args) -> int:
    """Schema guard. Non-zero exit means the ledger broke its own rules."""
    bad = []
    seen = set()
    for r in rows:
        if r["id"] in seen:
            bad.append(f"{r['id']}: duplicate id")
        seen.add(r["id"])
        if r["status"] not in VALID:
            bad.append(f"{r['id']}: invalid status {r['status']!r}")
        if r["status"] == "FIXED" and not r.get("test"):
            bad.append(f"{r['id']}: FIXED without a named regression test")
        if r["status"] in {"PARTIAL", "NOT-A-DEFECT", "WONTFIX"} and not r.get("note"):
            bad.append(f"{r['id']}: {r['status']} without a note")
    for b in bad:
        print("FAIL " + b, file=sys.stderr)
    if bad:
        print(f"\n{len(bad)} violation(s)", file=sys.stderr)
        return 1
    print(f"ok — {len(rows)} findings, schema clean")
    return 0


def cmd_regen(rows, _args) -> int:
    by_status = Counter(r["status"] for r in rows)
    done = sum(by_status[s] for s in TERMINAL)

    md: list[str] = []
    md.append("# TurboTab findings ledger\n")
    md.append(
        "> This is the **TurboTab transition** ledger. It is not the app's own defect\n"
        "> ledger — that lives at `docs/FINDINGS_LEDGER.md` and still has an open tail on\n"
        "> the multi-file import path. See `TRANSITION_PLAN.md` §05 before touching that code.\n"
    )
    md.append(
        "\n**Generated from `data/findings.json` — do not hand-edit.**\n"
        "Update the JSON via `tools/ledger.py set`, then `tools/ledger.py regen`.\n"
    )
    md.append("\n## Governing rule\n")
    md.append(
        "> The app may be **silent**, and it may **refuse**, but it must never\n"
        "> **assert something false**.\n"
    )
    md.append("\nNothing is closed without a regression test named after it.\n")
    md.append(f"\n## Progress\n\n**{done} of {len(rows)} closed.**\n")
    md.append("\n| Status | Count |\n|---|---:|")
    for s in ["UNVERIFIED", "OPEN", "PARTIAL", "FIXED", "NOT-A-DEFECT", "WONTFIX"]:
        if by_status.get(s):
            md.append(f"| `{s}` | {by_status[s]} |")

    order = ["OPEN", "PARTIAL", "UNVERIFIED", "FIXED", "NOT-A-DEFECT", "WONTFIX"]
    for status in order:
        group = [r for r in rows if r["status"] == status]
        if not group:
            continue
        md.append(f"\n---\n\n## {status} — {len(group)}\n")
        by_area: dict[str, list[dict]] = {}
        for r in group:
            by_area.setdefault(r["area"], []).append(r)
        for area in sorted(by_area, key=lambda a: -len(by_area[a])):
            sub = sorted(by_area[area], key=lambda r: SEV_ORDER.get(r["sev"], 9))
            md.append(f"\n### {AREA_NAME.get(area, area)} — {len(sub)}\n")
            md.append("| ID | Sev | Finding | Evidence | Action / Note |")
            md.append("|---|---|---|---|---|")
            for r in sub:
                tail = r.get("note") or r.get("act") or ""
                if r.get("test"):
                    tail = f"**test:** `{r['test']}` — {tail}" if tail else f"**test:** `{r['test']}`"
                md.append(
                    f"| `{r['id']}` | {r['sev']} | {norm(r['item'])[:200]} "
                    f"| `{norm(r.get('ev'))[:110]}` | {norm(tail)[:180]} |"
                )

    OUT.write_text("\n".join(md) + "\n")
    print(f"wrote {OUT.relative_to(ROOT.parent.parent)} — {done}/{len(rows)} closed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats")
    p_next = sub.add_parser("next")
    p_next.add_argument("--n", type=int, default=15)
    p_next.add_argument("--area")
    p_set = sub.add_parser("set")
    p_set.add_argument("id")
    p_set.add_argument("--status", required=True)
    p_set.add_argument("--note")
    p_set.add_argument("--test")
    p_set.add_argument("--evidence")
    sub.add_parser("regen")
    sub.add_parser("check")
    args = ap.parse_args()

    rows = load()
    return {
        "stats": cmd_stats,
        "next": cmd_next,
        "set": cmd_set,
        "regen": cmd_regen,
        "check": cmd_check,
    }[args.cmd](rows, args)


if __name__ == "__main__":
    raise SystemExit(main())

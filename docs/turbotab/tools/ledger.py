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
    python docs/turbotab/tools/ledger.py add --id FIND-ID --area STATE --sev high \
        --item "..." --status OPEN --evidence "file:line"
    python docs/turbotab/tools/ledger.py regen
    python docs/turbotab/tools/ledger.py check          # exits 1 if the schema is violated

`add` exists because work discovered during a build is a finding too, and until
L9 the tool could only *dispose* of rows the audit passes had written. A defect
found while fixing another one, with no way into the ledger, is the same silence
the ledger exists to remove — so it gets a door, through the tool, with the
schema guard applied.

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
    # Filed from drives of the Guided door. Present in the data since the L1
    # adjudication and absent from this table until `add` started validating
    # against it — the map was a display convenience and is now a schema.
    "GUIDED": "Guided-door drive feedback",
    "MISC": "Other",
}
SEV_ORDER = {"critical": 0, "landmine": 1, "high": 2, "invariant": 3, "medium": 4, "low": 5}


def load() -> list[dict]:
    return json.loads(DATA.read_text(encoding="utf-8"))


def save(rows: list[dict]) -> None:
    # encoding is explicit: the default is locale-dependent (cp1252 on Windows),
    # and write_text truncates before it raises, so a failure destroys the file.
    DATA.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")


def clip(text: str, limit: int) -> str:
    """Truncate at a word boundary, never mid-word.

    A hard slice cut "Analysis" one and two characters short, leaving a partial
    word in a tracked artifact — and, because the remaining stem is the one the
    British-spelling checker looks for, a spelling failure nobody had written.
    The generator was the bug, not the prose.
    """
    text = text or ""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    if space > limit * 0.6:          # keep most of the budget, lose the part word
        cut = cut[:space]
    return cut.rstrip(" ,;:—-") + "…"


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


def cmd_add(rows, args) -> int:
    """Append a finding discovered during a build.

    Same guards as `set`, applied before the row exists rather than after: a
    `FIXED` row needs its test named at birth, and a `PARTIAL` needs its note.
    Ids are unique and never reused — a reused id is a rewritten history.
    """
    if any(r["id"] == args.id for r in rows):
        print(f"{args.id} already exists; use `set` to change it", file=sys.stderr)
        return 1
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
    if args.sev not in SEV_ORDER:
        print(f"invalid severity {args.sev!r}; valid: {sorted(SEV_ORDER)}",
              file=sys.stderr)
        return 1
    if args.area not in AREA_NAME:
        print(f"invalid area {args.area!r}; valid: {sorted(AREA_NAME)}",
              file=sys.stderr)
        return 1

    rows.append({
        "id": args.id,
        "area": args.area,
        "sev": args.sev,
        "item": norm(args.item),
        "detail": norm(args.detail),
        "ev": norm(args.evidence),
        "act": norm(args.act),
        "status": status,
        "note": norm(args.note),
        "verified_against": norm(args.verified_against),
        "test": norm(args.test),
        "verified_ev": norm(args.evidence),
    })
    save(rows)
    print(f"added {args.id} [{status}]")
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
    # The JSON can be perfect while the generated markdown is empty or stale.
    # A guard that only reads the source cannot see a destroyed artifact.
    if not OUT.exists():
        print("FAIL FINDINGS_LEDGER.md missing — run regen", file=sys.stderr)
        return 1
    md = OUT.read_text(encoding="utf-8")
    if len(md) < 1024:
        print(f"FAIL FINDINGS_LEDGER.md is {len(md)} bytes — regen truncated it", file=sys.stderr)
        return 1
    missing = [r["id"] for r in rows if f"`{r['id']}`" not in md]
    if missing:
        print(f"FAIL {len(missing)} findings absent from the markdown "
              f"(e.g. {', '.join(missing[:5])}) — run regen", file=sys.stderr)
        return 1
    print(f"ok — {len(rows)} findings, schema clean, markdown current ({len(md):,} bytes)")
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
                    f"| `{r['id']}` | {r['sev']} | {clip(norm(r['item']), 200)} "
                    f"| `{clip(norm(r.get('ev')), 110)}` | {clip(norm(tail), 180)} |"
                )

    body = "\n".join(md) + "\n"
    tmp = OUT.with_suffix(".md.tmp")
    tmp.write_text(body, encoding="utf-8")   # write aside, then swap: never truncate the real file
    tmp.replace(OUT)
    if OUT.stat().st_size < 1024:
        raise SystemExit("regen produced a suspiciously small ledger — aborting")
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
    p_add = sub.add_parser("add")
    p_add.add_argument("--id", required=True)
    p_add.add_argument("--area", required=True)
    p_add.add_argument("--sev", required=True)
    p_add.add_argument("--item", required=True)
    p_add.add_argument("--status", default="OPEN")
    p_add.add_argument("--detail", default="")
    p_add.add_argument("--evidence", default="")
    p_add.add_argument("--act", default="")
    p_add.add_argument("--note", default="")
    p_add.add_argument("--test", default="")
    p_add.add_argument("--verified-against", dest="verified_against", default="")
    sub.add_parser("regen")
    sub.add_parser("check")
    args = ap.parse_args()

    rows = load()
    return {
        "stats": cmd_stats,
        "next": cmd_next,
        "set": cmd_set,
        "add": cmd_add,
        "regen": cmd_regen,
        "check": cmd_check,
    }[args.cmd](rows, args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        # piping into `head` closes stdout early; not an error worth a traceback
        try:
            sys.stdout.close()
        finally:
            raise SystemExit(0)

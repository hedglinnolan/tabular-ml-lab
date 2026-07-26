#!/usr/bin/env python3
"""The feature register, as data — because the hand-edited version got destroyed.

The register existed to turn silence about a capability into a failure. Then a
branch merge blind-copied an older FEATURE_PARITY.md over the newer one and the
whole Data & Target register vanished — silently, because a hand-maintained
markdown table has no guard. The ledger survived the same merge because it is
JSON worked through a tool. So the register now works the same way.

`data/register.json` is the source of truth. `FEATURE_REGISTER.md` is generated.

Usage
-----
    python docs/turbotab/tools/register.py stats
    python docs/turbotab/tools/register.py add --id x --step data-target \\
        --capability "..." --classic "..." --state classic-only --reason "..."
    python docs/turbotab/tools/register.py set x --state both --reason "..."
    python docs/turbotab/tools/register.py regen
    python docs/turbotab/tools/register.py check     # exits 1 on violation

States
------
    core          extracted into the shared core (an implementation fact)
    both          exposed in Classic and Guided
    classic-only  Classic has it, Guided does not — a claim to be justified, never a shrug
    guided-only   Guided has it, Classic does not — a debt owed back to Classic

Rules `check` enforces
----------------------
    - every id unique, every state valid
    - `classic-only` and `guided-only` require a reason
    - every step listed in BUILT_STEPS has at least one row — a step with no
      rows is exactly the silence the register exists to prevent
    - the generated markdown exists, is non-trivial, and contains every id
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "register.json"
OUT = ROOT / "FEATURE_REGISTER.md"

STATES = {"core", "both", "classic-only", "guided-only"}
NEED_REASON = {"classic-only", "guided-only"}

# Guided steps that have been built. Building a new step means adding it here,
# which is what makes "no rows for this step" a checkable failure.
BUILT_STEPS = ["data-target", "cross-step"]

STEP_NAME = {
    "data-target": "Data & Target (Classic: pages/01, Step 4)",
    "cross-step": "Cross-step infrastructure",
    "explore": "Explore / EDA (Classic: pages/02)",
    "features": "Feature engineering & selection (Classic: pages/03, 04)",
    "preprocess": "Preprocess (Classic: pages/05)",
    "train": "Train & compare (Classic: pages/06)",
    "explain": "Explainability & sensitivity (Classic: pages/07, 08)",
    "report": "Report & export (Classic: pages/10)",
}
STATE_ORDER = {"guided-only": 0, "both": 1, "core": 2, "classic-only": 3}


def load() -> list[dict]:
    return json.loads(DATA.read_text(encoding="utf-8"))


def save(rows: list[dict]) -> None:
    DATA.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")


def norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def cmd_stats(rows, _a) -> int:
    print(f"total {len(rows)}")
    for st, n in Counter(r["state"] for r in rows).most_common():
        print(f"  {st:14} {n:3}")
    for step in BUILT_STEPS:
        n = sum(1 for r in rows if r["step"] == step)
        print(f"  step {step:18} {n:3} rows")
    return 0


def cmd_add(rows, a) -> int:
    if any(r["id"] == a.id for r in rows):
        print(f"duplicate id: {a.id}", file=sys.stderr)
        return 1
    if a.state not in STATES:
        print(f"invalid state {a.state}; valid: {sorted(STATES)}", file=sys.stderr)
        return 1
    if a.state in NEED_REASON and not a.reason:
        print(f"{a.state} requires --reason", file=sys.stderr)
        return 1
    rows.append(dict(id=a.id, step=a.step, capability=norm(a.capability),
                     classic=norm(a.classic), state=a.state, reason=norm(a.reason)))
    save(rows)
    print(f"added {a.id} [{a.state}]")
    return 0


def cmd_set(rows, a) -> int:
    hits = [r for r in rows if r["id"] == a.id]
    if not hits:
        print(f"no such row: {a.id}", file=sys.stderr)
        return 1
    r = hits[0]
    if a.state:
        if a.state not in STATES:
            print(f"invalid state {a.state}", file=sys.stderr)
            return 1
        r["state"] = a.state
    if a.reason is not None:
        r["reason"] = norm(a.reason)
    if r["state"] in NEED_REASON and not r.get("reason"):
        print(f"{r['state']} requires a reason", file=sys.stderr)
        return 1
    save(rows)
    print(f"{a.id} -> {r['state']}")
    return 0


def cmd_check(rows, _a) -> int:
    bad = []
    seen = set()
    for r in rows:
        if r["id"] in seen:
            bad.append(f"{r['id']}: duplicate id")
        seen.add(r["id"])
        if r["state"] not in STATES:
            bad.append(f"{r['id']}: invalid state {r['state']!r}")
        if r["state"] in NEED_REASON and not norm(r.get("reason")):
            bad.append(f"{r['id']}: {r['state']} without a reason")
    for step in BUILT_STEPS:
        if not any(r["step"] == step for r in rows):
            bad.append(f"step {step}: BUILT but has no register rows — the silence this file exists to prevent")
    if not OUT.exists():
        bad.append("FEATURE_REGISTER.md missing — run regen")
    else:
        md = OUT.read_text(encoding="utf-8")
        if len(md) < 512:
            bad.append(f"FEATURE_REGISTER.md is {len(md)} bytes — regen truncated it")
        else:
            missing = [r["id"] for r in rows if f"`{r['id']}`" not in md]
            if missing:
                bad.append(f"{len(missing)} rows absent from the markdown (e.g. {', '.join(missing[:4])}) — run regen")
    for b in bad:
        print("FAIL " + b, file=sys.stderr)
    if bad:
        print(f"\n{len(bad)} violation(s)", file=sys.stderr)
        return 1
    print(f"ok — {len(rows)} rows, register clean, markdown current")
    return 0


def cmd_regen(rows, _a) -> int:
    c = Counter(r["state"] for r in rows)
    md: list[str] = []
    md.append("# TurboTab feature register\n")
    md.append("**Generated from `data/register.json` — do not hand-edit.** "
              "Update via `tools/register.py`, then `regen`. Rationale and rules: `FEATURE_PARITY.md`.\n")
    md.append("| State | Meaning | Count |\n|---|---|---:|")
    md.append(f"| `both` | exposed in Classic and Guided | {c.get('both', 0)} |")
    md.append(f"| `core` | extracted into the shared core | {c.get('core', 0)} |")
    md.append(f"| `classic-only` | a claim to be justified, never a shrug | {c.get('classic-only', 0)} |")
    md.append(f"| `guided-only` | a debt owed back to Classic | {c.get('guided-only', 0)} |")
    for step in BUILT_STEPS + sorted({r["step"] for r in rows} - set(BUILT_STEPS)):
        sub = [r for r in rows if r["step"] == step]
        if not sub:
            continue
        md.append(f"\n## {STEP_NAME.get(step, step)} — {len(sub)}\n")
        md.append("| ID | Capability | Classic | State | Reason |")
        md.append("|---|---|---|---|---|")
        for r in sorted(sub, key=lambda r: (STATE_ORDER.get(r["state"], 9), r["id"])):
            md.append(f"| `{r['id']}` | {norm(r['capability'])} | {norm(r.get('classic'))} "
                      f"| **{r['state']}** | {norm(r.get('reason'))} |")
    body = "\n".join(md) + "\n"
    tmp = OUT.with_suffix(".md.tmp")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(OUT)
    print(f"wrote {OUT.name} — {len(rows)} rows")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats")
    p = sub.add_parser("add")
    for f in ("--id", "--step", "--capability", "--state"):
        p.add_argument(f, required=True)
    p.add_argument("--classic", default="")
    p.add_argument("--reason", default="")
    p = sub.add_parser("set")
    p.add_argument("id")
    p.add_argument("--state")
    p.add_argument("--reason")
    sub.add_parser("regen")
    sub.add_parser("check")
    a = ap.parse_args()
    rows = load()
    return {"stats": cmd_stats, "add": cmd_add, "set": cmd_set,
            "regen": cmd_regen, "check": cmd_check}[a.cmd](rows, a)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        try:
            sys.stdout.close()
        finally:
            raise SystemExit(0)

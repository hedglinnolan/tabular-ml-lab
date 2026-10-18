"""The paper's participant flow from the prep's tables, by the plan file's exclusion rules.

The engine counts its flow within the rows whose outcome is measured, so the paper's flow over
the whole stacked sample (39,156 → 16,913, before any outcome) is not one the engine can show
(an engine gap the report names). This counts it outside the engine, with pandas: the prep's
files merged on ``SEQN``, then each of the plan file's exclusion rules in its order, as the
engine reads a range rule (``low <= value <= high``; a blank value excluded unless the rule says
``missing: keep``). It reads no engine code and changes nothing the runs use.

    scripts/py research/replication/sullivan2021/engine/check_flow.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PLAN = json.loads((HERE / "sullivan2021.json").read_text())


def keeps(frame: pd.DataFrame, rule: dict) -> pd.Series:
    x = pd.to_numeric(frame[rule["column"]], errors="coerce")
    inside = pd.Series(True, index=frame.index)
    if rule.get("low") is not None:
        inside &= x >= rule["low"]
    if rule.get("high") is not None:
        inside &= x <= rule["high"]
    if rule.get("missing", "exclude") == "keep":
        return x.isna() | inside
    return x.notna() & inside


def main() -> int:
    t = PLAN["table"]
    frame = pd.read_parquet(REPO / t["main"])
    for name in t["joins"]:
        frame = frame.merge(pd.read_parquet(REPO / t["join_folder"] / f"{name}.parquet"),
                            on=t["on"], how="left")
    paper = [n for _, n in PLAN["paper"]["flow"]]
    print(f"{'step':12} {'ours':>8} {'paper':>8}")
    print(f"{'start':12} {len(frame):8,} {paper[0]:8,}")
    # The paper's steps, in its order; its two recall lines are our two status rules and its
    # only-water line our two zero-energy rules.
    steps = [("age", ["age"]), ("pregnant", ["pregnant"]), ("rating", ["rating"]),
             ("recall1", ["recall1"]), ("recall2", ["recall2"]), ("water", ["water1", "water2"])]
    rules = {e["key"]: e["rule"] for e in PLAN["exclusions"]}
    out = 0
    for (label, keys), p in zip(steps, paper[1:-1]):
        before = len(frame)
        for k in keys:
            frame = frame[keeps(frame, rules[k])]
        dropped = len(frame) - before
        out |= dropped != p
        print(f"{label:12} {dropped:8,} {p:8,}")
    print(f"{'analytic':12} {len(frame):8,} {paper[-1]:8,}")
    return 1 if out or len(frame) != paper[-1] else 0


if __name__ == "__main__":
    sys.exit(main())

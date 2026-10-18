"""The Sullivan 2021 replication's numbers, gathered from the runs into ``results.json``.

Reads every run's capture (``research/data/derived/engine/runs/<name>.json``, written by
``run.py``) and writes, per run: the exposure's coefficient on the raw DBQ700 codes as the export's
Table 2 reports it, and ``-β`` beside the paper's β (SULLIVAN_PLAN ruling 3), its SE, df, p and the
design-based n (the analysis rows with a positive weight); the paper's printed numbers and whether
each matches at the paper's printed precision; for a variant, the run it varies (the same outcome
and model with the plan file's stated answers) and its shift in β and SE; the analytic rows' CSV
(its SHA-256 and row count, for the verifier); the answers kept through a "keep it, recorded" exit;
and the run's seconds. Nothing here refits or changes a number: each is read from the captures.

    scripts/py research/replication/sullivan2021/engine/results.py
"""
from __future__ import annotations

import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))
from check_rows import refit  # noqa: E402

PLAN = json.loads((HERE / "sullivan2021.json").read_text())
RUNS = REPO / "research" / "data" / "derived" / "engine" / "runs"
TEMPLATE = REPO / "research" / "data" / "derived" / "engine" / "template" / "template.json"
OUT = HERE / "results.json"
LADDER_MODELS = {"M0", "M1", "M2", "M3", "M4"}


def rounded(x: float, places: int) -> float:
    """``x`` at ``places`` decimals, halves away from zero (as a printed table rounds)."""
    q = Decimal(1).scaleb(-places)
    return float(Decimal(repr(x)).quantize(q, rounding=ROUND_HALF_UP))


def paper_for(outcome: str, model: str) -> dict[str, Any] | None:
    """The paper's printed numbers for this outcome and model, with their printed decimals."""
    p = PLAN["paper"]
    if model == "table4":
        row = p["table4"].get(outcome)
        places = p["precision"]["table4"].get(outcome)
        if row is None:
            return None
        return {"beta": row["beta"], "se": row["se"], "n": row["n"], "p": row["p"],
                "places": places, "source": "Table 4"}
    key = "M4" if model == "M4_table4" else model
    if key in LADDER_MODELS and outcome in p["ladder"]:
        cell = p["ladder"][outcome][key]
        return {"beta": cell[0], "se": cell[1], "n": None,
                "p": cell[2] if len(cell) > 2 else None,
                "places": p["precision"]["ladder"][outcome], "source": "Appendix Table 2"}
    return None


def base_of(run: dict[str, Any]) -> str | None:
    """The run a variant varies: the same outcome and model, the plan file's answers."""
    if not run.get("variant"):
        return None
    model, outcome = run["model"], run["outcome"]
    if model == "table4":
        return f"t4-{outcome}"
    if model == "M4_table4":
        return f"m4t4-{outcome}"
    return f"{model.lower()}-{outcome}"


def gather() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for path in sorted(RUNS.glob("*.json")):
        d = json.loads(path.read_text())
        run = d["run"]
        rows = d.get("table2") or []
        primary = next((r for r in rows if r.get("primary")), None)
        crude = next((r for r in rows if not r.get("primary")), None)
        fit = (d.get("fit") or {}).get(PLAN["model_family"]) or {}
        variant = PLAN["variants"].get(run.get("variant") or "", {})
        entry: dict[str, Any] = {
            "group": run["group"], "outcome": run["outcome"], "model": run["model"],
            "variant": run.get("variant"), "about": variant.get("about"),
            "alternative": bool(variant.get("alternative")),
            "export": d["export"].get("status"), "seconds": d["captured"]["seconds"],
            "commit": d["captured"].get("commit"),
            "kept_by_exit": [n for n in d.get("notes") or [] if "blocked and recorded" in n],
            "stopped": [n for n in d.get("notes") or [] if n.startswith("stopped")],
            "guessed": d.get("guessed") or [], "app_ranked": d.get("app_ranked") or [],
            "rows": {k: (d.get("rows") or {}).get(k) for k in ("file", "sha256", "rows")},
            "n_analyzed": fit.get("n"),
        }
        if primary is not None:
            entry.update({
                "beta_raw": primary["estimate"], "beta": -primary["estimate"],
                "se": primary["se"], "df": primary["df"], "p": primary["p"],
                "ci": [-primary["ci_high"], -primary["ci_low"]], "n": int(primary["n"]),
                "adjusted_for": primary["adjusted_for"]})
        if crude is not None:
            entry["unadjusted_same_rows"] = {"beta": -crude["estimate"], "se": crude["se"],
                                             "n": int(crude["n"])}
        paper = paper_for(run["outcome"], run["model"])
        if paper is not None and primary is not None:
            bp, sp = paper["places"]
            entry["paper"] = {k: paper[k] for k in ("beta", "se", "n", "p", "source")}
            entry["at_printed_precision"] = {
                "beta": rounded(entry["beta"], bp), "se": rounded(entry["se"], sp),
                "beta_matches": rounded(entry["beta"], bp) == paper["beta"],
                "se_matches": None if paper["se"] is None else
                rounded(entry["se"], sp) == paper["se"],
                "n_matches": None if paper["n"] is None else entry["n"] == paper["n"]}
        entry["base"] = base_of(run)
        if primary is not None and (d.get("rows") or {}).get("file"):
            entry["rows_refit"] = refit(path.stem)  # check_rows.py: the CSV refit by numpy
        out[path.stem] = entry
    for name, e in out.items():
        b = out.get(e.get("base") or "")
        if b and "beta" in e and "beta" in b:
            places = (e.get("paper") or {}) and paper_for(e["outcome"], e["model"])["places"]
            unit = 10.0 ** -places[0] if places else None
            e["shift"] = {"beta": e["beta"] - b["beta"], "se": e["se"] - b["se"],
                          "n": e["n"] - b["n"],
                          "beta_in_printed_units": None if unit is None else
                          (e["beta"] - b["beta"]) / unit,
                          "printed_beta_changes": None if not places else
                          rounded(e["beta"], places[0]) != rounded(b["beta"], places[0]),
                          "printed_se_changes": None if not places else
                          rounded(e["se"], places[1]) != rounded(b["se"], places[1])}
    return out


def main() -> int:
    runs = gather()
    template = json.loads(TEMPLATE.read_text()) if TEMPLATE.exists() else {}
    seconds = [r["seconds"] for r in runs.values()]
    doc = {
        "about": ("The Sullivan 2021 replication through TurboTab's Estimate lane (results.py, from "
                  "run.py's captures). beta is -1 x the engine's coefficient on the raw DBQ700 "
                  "codes (NHANES codes excellent = 1, the paper poor = 1); n is the design-based n, "
                  "the analysis rows with a positive weight; n_analyzed counts the zero-weight "
                  "rows too. rows names each run's analytic-rows CSV under research/data/derived/"
                  "engine/ (ignored by git), with its SHA-256 and row count."),
        "plan": "research/replication/sullivan2021/engine/sullivan2021.json",
        "template": {"joined": template.get("joined"), "join_seconds": template.get("seconds"),
                     "join_seconds_total": round(sum((template.get("seconds") or {}).values()), 1)},
        "totals": {"runs": len(runs), "exports_200": sum(r["export"] == 200 for r in runs.values()),
                   "seconds": sum(seconds), "median_seconds": sorted(seconds)[len(seconds) // 2]
                   if seconds else None,
                   "fitted_models": sum(2 for r in runs.values() if "beta" in r)},
        "runs": runs,
    }
    OUT.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}: {len(runs)} runs, {sum(seconds)} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

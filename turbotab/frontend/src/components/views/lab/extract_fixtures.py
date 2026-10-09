"""Cut the /lab/views fixtures for the table, forest and page views out of the captured engine
journeys (src/mocks/fixtures/m3-*.json): each artifact at its last captured version, unchanged.

    python3 src/components/views/lab/extract_fixtures.py   # from turbotab/frontend
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURES = HERE.parents[2] / "mocks" / "fixtures"
OUT = HERE / "fixtures.json"

# Table 1's characteristics: the analysis's own columns. table1FromProfile (adapters.ts) leaves the
# outcome out whatever this list says: the profile covers the whole file, not the rows analyzed.
TABLE1_COLUMNS = ["age", "gender", "bmi", "waist", "kcal", "sugar", "protein", "carb", "fat_total"]


def merge(base, patch):
    """src/mocks/m3.ts applyPatch: objects merge key by key, anything else is replaced whole."""
    if isinstance(patch, dict) and isinstance(base, dict):
        out = dict(base)
        for k, v in patch.items():
            if isinstance(v, dict) and v.get("$del") == 1 and len(v) == 1:
                out.pop(k, None)
            elif isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = merge(out[k], v)
            else:
                out[k] = v
        return out
    return patch


def latest(journey: str, stage: str):
    packed = json.loads((FIXTURES / f"m3-{journey}.json").read_text())["artifacts"][stage]
    value = packed["base"]
    for patch in packed["patches"]:
        value = merge(value, patch)
    return value


def main() -> None:
    profile = latest("nhanes-inference", "profile")
    fit = latest("nhanes-inference", "fit")
    out = {
        "meta": {
            "source": "src/mocks/fixtures/m3-nhanes-inference.json and m3-time-varying.json (captured 2026-10-05)",
            "script": "src/components/views/lab/extract_fixtures.py",
            "note": "Each artifact at its last captured version, unchanged; the profile keeps Table 1's columns only.",
        },
        "nhanes": {
            "effects": latest("nhanes-inference", "effects"),
            "profile": {**profile, "columns": [c for c in profile["columns"] if c["name"] in TABLE1_COLUMNS]},
            "cohort": latest("nhanes-inference", "cohort"),
            "concerns": fit["models"][0]["concerns"],
        },
        "timeVarying": {"effects": latest("time-varying", "effects")},
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

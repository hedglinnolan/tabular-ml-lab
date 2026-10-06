"""Capture real Explore stacks into `explore-stack.html`'s data island.

`GUIDED-149`, Part C of L45. The product owner judges the bound by looking at it,
so what he looks at has to be **what the server actually serves** rather than a
hand-typed mock. This drives the real API against real fixtures, partitions with
the real `turbotab.attention.stack` at each bound, and writes the result between
two markers in the prototype.

Half generated, half hand-assembled — the same arrangement `COPY_DECK.md` uses,
and for the same reason. The prose, the layout and the questions being asked are
hand-written and are the point; the numbers are generated because a hand-typed
count is a claim with nothing behind it, and a prototype that quietly disagreed
with the app would be worse than no prototype.

    venv/bin/python docs/turbotab/prototypes/capture_explore_stack.py

Re-run it after anything that changes what a fixture produces. The page states
the capture's fixtures and bounds in its own header, so a stale capture is
visible rather than silent.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PAGE = Path(__file__).resolve().parent / "explore-stack.html"
BEGIN = "/* ⟦CAPTURE-BEGIN⟧ */"
END = "/* ⟦CAPTURE-END⟧ */"

#: The bounds the page compares. Two is `PRODUCT_VISION.md` §08's interruption
#: budget — included precisely because the loop's claim is that it is the wrong
#: number for a stack, and an argument that never shows the option it rejects is
#: not an argument. Five ships. Eight is the other side of the median.
BOUNDS = (2, 5, 8)

#: `GUIDED-097`'s fixture rule wants at least two lenses; this takes five, and
#: each one is here because it is a different SHAPE of stack rather than a
#: different domain.
#:
#: `(fixture, lens, target, why it is in the set)`
CASES = [
    ("clinical_labs.csv", "clinical", "readmitted",
     "The table that produced the finding. 21 served, 13 in the Explore stack, "
     "one of them the only critical."),
    ("metabolomics_untargeted.csv", "metabolomics", "responder",
     "Four criticals. The case where the bound does not decide the size of the "
     "stack, because nothing that gates a decision may be collapsed."),
    ("nhanes_kilojoules.csv", "dietary", "DR1TKCAL",
     "Six findings and no critical — one over a bound of five. This is the "
     "panel where a bound reads as arbitrary if it is going to."),
    ("survey_instrument.csv", "survey", None,
     "Three findings. Under every bound here, so the stack is complete and "
     "says so."),
    ("longitudinal_visits.csv", None, None,
     "One finding, and no lens answered. The smallest stack any fixture in "
     "this repository produces."),
]

#: THE POPULATION THE BOUND WAS MEASURED ON — every CSV in
#: `turbotab/sample_data/`, driven the same way, so the median in
#: `attention.BOUND_BECAUSE` is re-derivable rather than quoted. A fixture whose
#: companion names a lens is driven under it; the four with no companion are
#: driven with no lens, which is a state a real project reaches (the lens
#: question is answerable with *none of these*). A target is set where one is
#: named, and the Explore stack does not depend on it.
#:
#: `(fixture, lens, target)`
POPULATION = [
    ("clinical_labs.csv", "clinical", "readmitted"),
    ("clinic_visits.csv", "clinical", "outcome"),
    ("clinical_risk.csv", "clinical", None),
    ("clinical_longitudinal.csv", "clinical", None),
    ("leaky_sepsis.csv", "clinical", None),
    ("nhanes_kilojoules.csv", "dietary", "DR1TKCAL"),
    ("nhanes_dietary.csv", "dietary", None),
    ("nhanes_partial_design.csv", "dietary", None),
    ("dietary_recalls.csv", "dietary", None),
    ("metabolomics_untargeted.csv", "metabolomics", "responder"),
    ("survey_sentinels.csv", "survey", "sought_support"),
    ("survey_instrument.csv", "survey", None),
    ("genomics_expression.csv", "genomics", "condition"),
    ("multiclass_stage.csv", None, None),
    ("wide_assay.csv", None, None),
    ("longitudinal_visits.csv", None, "outcome"),
]

#: SAID OUT LOUD, because a substitution nobody mentions reads as a set that
#: covered everything. The loop prompt asks for "a table with two" and no
#: fixture in `turbotab/sample_data/` produces exactly two Explore findings —
#: the small end runs 1, 3, 3, 3, 3. One and three are shown instead. Part D's
#: probe drives a synthetic two, where a fixture is not required.
NOT_CAPTURED = (
    "No fixture in turbotab/sample_data/ produces exactly two Explore "
    "findings; the small end of the population is 1 · 3 · 3 · 3 · 3. The "
    "one-finding and three-finding tables are shown instead, and the probe "
    "drives a synthetic two."
)


def _drive(client, fixture: str, lens, target):
    from turbotab import attention as A

    data = ROOT / "turbotab" / "sample_data" / fixture
    with data.open("rb") as handle:
        pid = client.post("/project", files={
            "file": (fixture, handle, "text/csv")}).json()["id"]
    notes = []
    if lens:
        r = client.post(f"/project/{pid}/decision",
                        json={"kind": "set_lens", "payload": {"lens": [lens]}})
        if r.status_code != 200:
            notes.append(f"set_lens refused: {r.status_code}")
    if target:
        r = client.post(f"/project/{pid}/decision",
                        json={"kind": "set_target",
                              "payload": {"column": target}})
        if r.status_code != 200:
            notes.append(f"set_target refused: {r.status_code}")
    project = client.get(f"/project/{pid}").json()

    ordered = A.explore_findings(project["findings"])
    cards = [{
        "id": f["id"],
        "title": f.get("title") or "",
        "detail": (f.get("detail") or "")[:340],
        "severity": f.get("severity") or "",
        "confidence": f.get("confidence") or "",
        "source_label": f.get("source_label") or "",
        "subject": (f.get("shape") or {}).get("subject_line") or "",
        "has_chips": bool((f.get("shape") or {}).get("has_chips")),
        "columns": [str(c) for c in (f.get("affected_columns") or [])][:5],
        "badge": ((f.get("evidence") or {}).get("evidence_status") or ""),
    } for f in ordered]

    return {
        "fixture": fixture,
        "lens": lens or "",
        "target": target or "",
        "n_served_total": len(project["findings"]),
        "cards": cards,
        "notes": notes,
        "stacks": {str(b): A.stack(project["findings"], bound=b)
                   for b in BOUNDS},
    }


def main() -> int:
    os.environ.setdefault("PYTHONHASHSEED", "0")
    from fastapi.testclient import TestClient

    from turbotab import api, attention as A

    client = TestClient(api.app)
    payload = {
        "bounds": list(BOUNDS),
        "ships": A.BOUND,
        "bound_because": A.BOUND_BECAUSE,
        "not_captured": NOT_CAPTURED,
        "cases": [],
        "population": [],
    }

    # THE MEASUREMENT ITSELF, not a quotation of it. The bound's whole
    # justification is a median over these sixteen tables, and a prototype that
    # asserted the median while showing five of them would be asking the product
    # owner to take the load-bearing number on trust.
    for fixture, lens, target in POPULATION:
        case = _drive(client, fixture, lens, target)
        payload["population"].append({
            "fixture": fixture,
            "lens": lens or "",
            "n": len(case["cards"]),
            "n_gating": sum(1 for c in case["cards"]
                            if c["severity"] in A.NEVER_COLLAPSED),
            "collapsed_at_ships": len(
                case["stacks"][str(A.BOUND)]["collapsed"])
            if str(A.BOUND) in case["stacks"] else None,
        })
    sizes = sorted(row["n"] for row in payload["population"])
    mid = len(sizes) // 2
    median = (sizes[mid] if len(sizes) % 2
              else (sizes[mid - 1] + sizes[mid]) / 2)
    # An integral median renders as an integer. "median 5.0" on the page would
    # be the page saying something slightly untrue about its own arithmetic.
    payload["median"] = int(median) if float(median).is_integer() else median
    payload["sizes"] = sizes
    payload["n_collapsing"] = sum(1 for row in payload["population"]
                                  if row["collapsed_at_ships"])
    print(f"  population: {len(sizes)} tables, sizes {sizes}, "
          f"median {payload['median']}, "
          f"{payload['n_collapsing']} collapse anything at bound {A.BOUND}")
    if str(A.BOUND) not in [str(b) for b in BOUNDS]:
        print(f"! the shipping bound {A.BOUND} is not among {BOUNDS}; "
              f"the page would default to a bound the build does not use")
        return 2

    for fixture, lens, target, why in CASES:
        case = _drive(client, fixture, lens, target)
        case["why"] = why
        payload["cases"].append(case)
        print(f"  {fixture:<32} lens={lens or '—':<13} "
              f"explore={len(case['cards']):>3}  "
              + "  ".join(f"b{b}:{len(case['stacks'][str(b)]['collapsed'])}↓"
                          for b in BOUNDS)
              + ("  " + "; ".join(case["notes"]) if case["notes"] else ""))

    text = PAGE.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        print(f"! {PAGE.name} has no capture markers; nothing written")
        return 2
    block = (BEGIN + "\nvar CAPTURE = "
             + json.dumps(payload, indent=1, ensure_ascii=False)
             + ";\n" + END)
    text = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), lambda _: block,
                  text, flags=re.S)
    PAGE.write_text(text, encoding="utf-8")
    print(f"wrote {len(payload['cases'])} cases × {len(BOUNDS)} bounds "
          f"into {PAGE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

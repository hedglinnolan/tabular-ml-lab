#!/usr/bin/env python3
"""Dump real `/recipes` responses for `recipe-lattice.html`. Deterministic.

    venv/bin/python docs/turbotab/prototypes/capture_recipes.py

**Why the capture script is committed and not just its output**, which is the
same argument `sample_data/make_fixtures.py` makes: a payload whose provenance
is lost is a payload nobody can refresh, and the next person who changes
`resolve()` has no way to tell whether the prototype still describes the app.

**And why the prototype is fed a capture rather than a mockup.** `GUIDED-074`
is about a lattice the engine already computes and never draws. A prototype
that faked the resolution would teach the wrong thing twice — wrong about what
the app does, and wrong about whether the drawing is worth building.

Three captures, chosen because they differ in the two ways the lattice can:

* `no_pack` — a 396-column assay panel under `other`, so the CORE table alone
  resolves and no pack row is in the lattice.
* `metabolomics` — the SAME table under the metabolomics lens, so the pack's two
  rows enter the precedence lattice and the scale cells re-resolve. **The delta
  between these two is the motion the prototype is testing**, and it is on one
  table so nothing else changes underneath it.
* `dietary` — a different table where the scale divergence is NOT material, so
  three variant questions are suppressed rather than raised. The assay captures
  suppress none.

The dietary lens was the obvious third capture and is not here: it contributes
PRIORS rather than recipe rows, so `pack_defaults` is empty under it and the
lattice is identical to the core one. That is legitimate and it is also why the
pack delta is demonstrated on metabolomics.

Each capture also carries `candidates`: for every (model, operation) cell, every
default row that matched, with the specificity rank the engine gave it, so the
page displays a resolution rather than re-deriving one in JavaScript.

**That used to be computed here**, by reaching into `recipes._matches` and
`recipes._SPECIFICITY`. `GUIDED-074`'s port moved it to `recipes.candidates`
and `/recipes` serves it, so this script now captures the field like every
other one. Two implementations of a precedence rule drift into two plausible
answers and nothing on screen distinguishes them — the reason the ranking
belongs beside `resolve`, which is the function whose rule it is.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = ROOT / "turbotab" / "sample_data"


def capture(fixture: str, target: str, lens, steps, picks) -> dict:
    from fastapi.testclient import TestClient

    from turbotab import api, recipes as rec

    state = rec.snapshot()
    try:
        rec._install_core()
        from turbotab import packs
        packs.unload_for_test()

        client = TestClient(api.app)
        with open(FIXTURES / fixture, "rb") as fh:
            pid = client.post("/project", files={
                "file": (fixture, fh, "text/csv")}).json()["id"]

        def decide(what, **payload):
            response = client.post(f"/project/{pid}/decision",
                                   json={"kind": what, "payload": payload})
            if response.status_code >= 400:
                raise SystemExit(f"{what} refused: {response.text[:300]}")

        decide("set_lens", lens=lens)
        decide("set_target", column=target)
        for what, payload in steps:
            decide(what, **payload)
        decide("set_eligibility", answer="everyone")
        decide("seal")

        shelf = client.get(f"/project/{pid}/models").json()
        available = {m["key"] for g in shelf.get("groups", [])
                     for m in g["models"]}
        chosen = [k for k in picks if k in available]
        decide("select_models", models=chosen)
        decide("set_preparation_mode", mode="per_model")

        payload = client.get(f"/project/{pid}/recipes").json()
        payload["_fixture"] = fixture
        payload["_lens"] = lens
        return payload
    finally:
        rec.restore(state)
        from turbotab import packs
        packs.unload_for_test()


DIETARY_STEPS = [
    ("set_grain", {"answer": "people_repeat", "group_col": "participant_id"}),
    ("set_repeat_kind", {"kind": "repeats"}),
    ("set_unit_of_analysis", {"unit": "person"}),
    ("set_aggregation", {"method": "mean"}),
]
DIETARY_PICKS = ["ridge", "rf", "knn_reg", "histgb_reg", "nn"]
ASSAY_PICKS = ["ridge", "rf", "knn", "histgb_clf", "logreg", "nn"]


def main() -> None:
    captures = {
        "no_pack": capture(
            "metabolomics_untargeted.csv", "responder", ["other"],
            [("set_grain", {"answer": "one_row_per_person"})], ASSAY_PICKS),
        "metabolomics": capture(
            "metabolomics_untargeted.csv", "responder", ["metabolomics"],
            [("set_grain", {"answer": "one_row_per_person"})], ASSAY_PICKS),
        "dietary": capture("dietary_recalls.csv", "hba1c", ["dietary"],
                           DIETARY_STEPS, DIETARY_PICKS),
    }
    # A `.js` assignment rather than a `.json` file, and the reason is the
    # prototype's own constraint: `file://` blocks `fetch`, so a JSON sidecar
    # could not be read without a server — which is the build step
    # `PRODUCT_VISION.md` §06 says these prototypes exist to avoid. A script tag
    # loads from disk with no server and no bundler.
    path = HERE / "recipe-lattice-data.js"
    path.write_text(
        "// GENERATED by capture_recipes.py — do not hand-edit.\n"
        "// Real `/recipes` responses. Refresh with:\n"
        "//     venv/bin/python docs/turbotab/prototypes/capture_recipes.py\n"
        "window.__RECIPE_CAPTURES = "
        + json.dumps(captures, indent=1, ensure_ascii=False) + ";\n",
        encoding="utf-8")
    for name, payload in captures.items():
        print(f"{name:14s} {len(payload['models']):2d} models × "
              f"{len(payload['operations'])} operations, "
              f"{payload['n_choices_suppressed']} suppressed, "
              f"{len(payload['pack_defaults'])} pack row(s)")
    print(f"wrote {path.relative_to(ROOT)} "
          f"({path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()

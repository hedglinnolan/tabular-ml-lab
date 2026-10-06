"""L51-D — two rows the tail was growing around, and two parked with a date.

## `GUIDED-189` — a built chip that opens onto zero cards

`ml/eda_recommender.py` raised the Explore missingness chip at `rate > 0.05`
while `ml/missingness_plan.HIGH_MISSING_SHARE` gated the cards the chip opens
onto at `0.20`. Two thresholds, **4× apart**, deciding the two halves of one
affordance — so a table whose worst column sits between them got a
solid-bordered chip whose own tooltip read *"2 columns with >5% missing
values"* and which opened onto an empty panel. Measured on the shipped
`multiclass_stage.csv`: `crp` 10.0%, `bmi` 7.1%, chip `built: true`,
`/evidence/missingness` returned `[]`.

**Neither threshold moved.** The row's own `act` says *"decide which threshold
is the real one and make the other read it, rather than moving either"*, and
that is also `AGENT_ONBOARD.md` §08 check 2 — the loop that pressured a
threshold does not get to move it. The one that **fills** the panel is the real
one, because it decides whether there is anything to look at; the chip reads it
now instead of holding a second copy. **§06.2's exception was available and was
not invoked, because it was not needed.**

## `GUIDED-196` — a controller-wide throw wearing a message's clothes

The boot's terminal `.catch(function(e){ setErr(e.message); })` could not tell
a **request that failed** from an **exception that escaped `renderAll`**, and
reported both as a sentence in the error sink. Observed live: a renderer called
a function that did not exist, `renderAll` died after `renderData`, and the
whole journey rendered as an upload step with one error line — which reads as
*the server said no*, not as *this page is broken*. `GUIDED-139` is why it
matters: one such error once killed every pull affordance in the door and
nothing said so.

A request failure is the app working. A `ReferenceError` escaping a renderer is
the app **not** working, and the honest form says which — and says that what is
on screen is incomplete, which is the part a person cannot otherwise know.
"""
# The tests here that drove the retired legacy app (turbotab/api.py, the page, the
# figure and manuscript modules, the gates) were removed with it, BLUEPRINT §9.1; git keeps them.
from __future__ import annotations

from pathlib import Path


DATA = Path(__file__).resolve().parent / "sample_data"


# ── GUIDED-189 ───────────────────────────────────────────────────────────────

def test_the_chip_and_the_panel_read_one_threshold():
    """The structural half: there is no second copy left to drift."""
    from ml import eda_recommender, missingness_plan

    source = Path(eda_recommender.__file__).read_text(encoding="utf-8")
    assert "HIGH_MISSING_SHARE" in source, (
        "the recommender no longer reads the panel's threshold, so the chip "
        "and the cards it opens onto can disagree again")
    body = source[source.index("signals.high_missing_cols"):][:400]
    assert "0.05" not in body, (
        f"the chip still holds its own literal threshold: {body[:200]!r}")
    assert missingness_plan.HIGH_MISSING_SHARE == 0.20, (
        "the panel's threshold moved. Neither threshold was supposed to — the "
        "fix was for one to READ the other, and §08 check 2 forbids moving a "
        "threshold in the loop that pressured it")


# ── GUIDED-196 ───────────────────────────────────────────────────────────────


def _routes(client, pid):
    out = {f"/project/{pid}": client.get(f"/project/{pid}").json()}
    for path in ("interview?step=data", "interview?step=explore",
                 "interview?step=features", "interview?step=preprocess",
                 "capabilities", "features", "recipes", "preprocess", "figures",
                 "draft", "manuscript", "models", "training", "instability",
                 "explain", "sensitivity", "evidence/plausibility",
                 "evidence/missingness"):
        got = client.get(f"/project/{pid}/{path}")
        out[f"/project/{pid}/{path}"] = got.json() if got.status_code == 200 else {}
    return out



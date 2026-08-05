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
from __future__ import annotations

from pathlib import Path

import pytest

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


@pytest.mark.parametrize("fixture,target", [
    ("multiclass_stage.csv", "disease_stage"),
    ("clinic_visits.csv", "outcome"),
], ids=["the row's own reproduction", "a table with real missingness"])
def test_a_built_chip_opens_onto_something(fixture, target):
    """`GUIDED-189`'s own evidence, driven on the fixture it was found on.

    The claim is an agreement, not a count: **if** the chip is raised, the
    panel it opens has cards. A fixture where neither happens satisfies it
    honestly, which is why the reproduction fixture is parametrized beside one
    that does have missingness.
    """
    from fastapi.testclient import TestClient

    from turbotab import api

    client = TestClient(api.app)
    with (DATA / fixture).open("rb") as handle:
        pid = client.post("/project", files={
            "file": (fixture, handle, "text/csv")}).json()["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": target}})

    plan = client.get(f"/project/{pid}/interview?step=explore").json()
    chips = [q for q in plan.get("questions", [])
             if str(q.get("key", "")).startswith("look::")
             and "missing" in str(q.get("key", ""))]
    cards = client.get(f"/project/{pid}/evidence/missingness").json().get("cards") or []
    built = [c for c in chips if c.get("built")]

    assert not (built and not cards), (
        f"{fixture}: the missingness chip is raised and built, and the panel "
        f"it opens has {len(cards)} cards. That is a solid-bordered control "
        f"whose own tooltip promises columns and which opens onto nothing — "
        f"`GUIDED-006`'s sentence, arriving through a threshold mismatch")


# ── GUIDED-196 ───────────────────────────────────────────────────────────────

def _project(fixture="clinical_labs.csv", target="readmitted"):
    from fastapi.testclient import TestClient

    from turbotab import api

    client = TestClient(api.app)
    with (DATA / fixture).open("rb") as handle:
        pid = client.post("/project", files={
            "file": (fixture, handle, "text/csv")}).json()["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": target}})
    return client, pid


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


def test_a_programming_error_is_not_reported_as_an_answer():
    """`GUIDED-196`, driven through a real path rather than by calling the
    classifier.

    The classifier lives inside the page's IIFE and exposing it for a test
    would be test plumbing in production code. So the throw is **provoked**:
    `GUIDED-188` records that `findingById` reads `P.findings.length` with no
    guard, so a project payload without a `findings` array makes a renderer
    throw a `TypeError` on the real render path. That is the exact shape the
    row was found on — a renderer dying part-way — and driving it also
    exercises `GUIDED-188`'s latent defect rather than only asserting about it.
    """
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")

    client, pid = _project()
    routes = _routes(client, pid)
    broken = dict(routes[f"/project/{pid}"])
    broken.pop("findings", None)
    routes[f"/project/{pid}"] = broken

    out = PH.run(
        "for (var i = 0; i < 10; i++) "
        "  await new Promise(function(r){ setTimeout(r, 0); });\n"
        "__emit({said: __harness.el('upErr') ? __harness.el('upErr').textContent : null,"
        "        band: __harness.html('refusal')});",
        routes=routes, search=f"?project={pid}")

    said = out["said"] or ""
    assert said, (
        "a renderer threw and the page said nothing at all, which is worse "
        "than saying the wrong thing")
    assert "TypeError" in said or "ReferenceError" in said, (
        f"the page reported a controller fault as an ordinary message: "
        f"{said!r}. That reads as *the server said no* for *this page is "
        f"broken*, which is `GUIDED-196`")
    assert "incomplete" in said, (
        f"the sentence does not tell the user what they can see is partial — "
        f"the one thing they cannot work out for themselves: {said!r}")
    assert "console" in (out["band"] or ""), (
        "nothing points at where the details are, so whoever is asked about it "
        "afterwards has the same nothing the user had")


def test_a_request_failure_still_reads_as_one():
    """The negative control, and it is what stops this being noise.

    A refusal from the server is the app working. If every failure started
    reading as a broken page the distinction would be gone in the other
    direction — so an ordinary 400 must still arrive as its own sentence.
    """
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")

    client, pid = _project()
    routes = _routes(client, pid)
    routes[f"POST /project/{pid}/decision"] = {
        "__status": 400, "body": {"detail": "This column is not in your table."}}

    out = PH.run(
        "var m = (__harness.html('profList') || '').match(/data-dismiss=\"([^\"]+)\"/);\n"
        "if (m) __harness.dispatch('click', __harness.target("
        "  {'data-dismiss': m[1], 'data-ac': 'find-' + m[1]}));\n"
        "for (var i = 0; i < 10; i++) "
        "  await new Promise(function(r){ setTimeout(r, 0); });\n"
        "__emit({pressed: !!m,"
        "        said: __harness.el('upErr') ? __harness.el('upErr').textContent : null});",
        routes=routes, search=f"?project={pid}")

    if not out["pressed"]:
        pytest.skip("no finding card rendered on this fixture to press")
    assert out["said"] == "This column is not in your table.", (
        f"an ordinary refusal was rewritten as a controller fault: "
        f"{out['said']!r}. The classifier has started claiming everything")


def test_every_render_entry_point_is_inside_the_boundary():
    """A boundary with a way around it is not one.

    Six call sites assign `P` and render; each must go through the guard, or a
    throw from that path is unclassified again.
    """
    page = (Path(__file__).resolve().parent / "web" / "index.html").read_text(
        encoding="utf-8")
    script = page[page.index("<script>"):page.rindex("</script>")]
    bare = [line.strip() for line in script.split("\n")
            if "renderAll();" in line and "renderAllGuarded" not in line]
    assert len(bare) == 1, (
        f"{len(bare)} bare `renderAll()` call sites: {bare}. Exactly one is "
        f"expected — the call inside `renderAllGuarded` itself")

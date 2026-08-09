"""L54-B — `GUIDED-178`'s deck, and the mechanism underneath it.

The product owner's specification, given in conversation on 2026-08-07: stacked
cards you page through showing before/after preprocessing for each model; that
place transforms at Train; and again at Compare. **One component spanning three
steps** — his standing thesis that *the steps are not the product, the
connective tissue between them is*, as a single object.

## B0 is the deliverable and this file is where it is checked

`DESIGN_LANGUAGE.md` §05.2: *"an object that is destroyed and replaced teaches
nothing, however smoothly it fades."* A deck rebuilt by `innerHTML` is
destroy-and-replace — the card for `logreg` after a re-render is a different
element that happens to look the same, so nothing can follow what became what
when face 3 reorders by score.

So the assertions below are about **node identity across a re-render**, driven
in a real DOM through `pageharness.py` rather than inferred from the source.
A source grep would prove the string `innerHTML` is absent; it would not prove
the cards survive, and surviving is the claim.

## What is NOT covered

- **Faces 2 and 3.** Specified and registered as `Pending`, not built. There is
  no reorder to drive, so the retention test moves cards by re-rendering with a
  changed model set rather than by a ranking that does not exist yet.
- **Motion.** No transition is attached. `§05` rule 5 forbids ambient movement
  and there is nothing yet for a transition to express; B0's point is that the
  nodes are retained so motion CAN be added without rewriting the renderer.
- **The other 105 `innerHTML` sites**, which are `MISC-021`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent / "sample_data"
PAGE = Path(__file__).resolve().parent / "web" / "index.html"


def _sealed(fixture="clinical_labs.csv", target="readmitted",
            group_col="patient_id"):
    """A project past the seal, because that is where per-model recipes exist."""
    from fastapi.testclient import TestClient

    from turbotab import api

    client = TestClient(api.app)
    with (DATA / fixture).open("rb") as handle:
        pid = client.post("/project", files={
            "file": (fixture, handle, "text/csv")}).json()["id"]
    for kind, payload in [
            ("set_target", {"column": target}),
            ("set_grain", {"answer": "people_repeat", "group_col": group_col}),
            ("set_repeat_kind", {"kind": "repeats"}),
            ("set_unit_of_analysis", {"unit": "record"}),
            ("set_eligibility", {"answer": "everyone"}),
            ("seal", {"fraction": 0.25})]:
        got = client.post(f"/project/{pid}/decision",
                          json={"kind": kind, "payload": payload})
        assert got.status_code == 200, (kind, got.text[:200])
    return client, pid


def _select(client, pid, n=3):
    shelf = client.get(f"/project/{pid}/models").json()
    keys = [m["key"] for g in shelf.get("groups", []) for m in g.get("models", [])]
    assert keys, "the shelf offered no models, so nothing here is driven"
    chosen = keys[:n]
    got = client.post(f"/project/{pid}/decision",
                      json={"kind": "select_models", "payload": {"models": chosen}})
    assert got.status_code == 200, got.text[:200]
    return chosen


def _routes(client, pid):
    out = {f"/project/{pid}": client.get(f"/project/{pid}").json()}
    for path in ("interview?step=data", "interview?step=explore",
                 "interview?step=features", "interview?step=preprocess",
                 "capabilities", "features", "recipes", "preprocess", "figures",
                 "draft", "manuscript", "models", "training", "instability",
                 "explain", "sensitivity", "checklist",
                 "evidence/plausibility", "evidence/missingness"):
        got = client.get(f"/project/{pid}/{path}")
        out[f"/project/{pid}/{path}"] = got.json() if got.status_code == 200 else {}
    return out


# ── B0 · the mechanism ───────────────────────────────────────────────────────

def test_the_deck_region_never_assigns_innerhtml():
    """The structural half, and it is deliberately the WEAKER of the two.

    A source check cannot show that cards survive; it can only show that the
    one construct guaranteeing they do not is absent. The behavioral test
    below is the real one.
    """
    page = PAGE.read_text(encoding="utf-8")
    # `find`, NOT `index`, and the absence is an ASSERTION rather than a
    # `ValueError`. Under a total revert the region does not exist, and
    # `str.index` raises — so the probe died on a lookup instead of reporting
    # that the deck was gone. That is `TEST-064`'s class, met in the test
    # written the same day as the row.
    start = page.find("/* ═════════ THE DECK ═════════")
    end = page.find("function renderStudy(){")
    assert start != -1 and end != -1 and start < end, (
        "the deck region is not in the page at all, so there is no "
        "node-owning renderer for the model shelf")
    region = page[start:end]
    # THE POSITIVE CONTROL. An absence assertion over a region located by
    # string search passes loudest when the search found something tiny.
    assert region.count("\n") > 100, (
        f"the deck region resolved to {region.count(chr(10))} lines; the "
        f"assertion below would be about almost nothing")
    assignments = [ln.strip() for ln in region.split("\n")
                   if re.search(r"\binnerHTML\s*=\s*[^=]", ln)
                   and not ln.lstrip().startswith(("*", "/*"))
                   and "`innerHTML =`" not in ln]
    assert not assignments, (
        f"the deck region assigns innerHTML: {assignments}. Every such write "
        f"destroys the cards and builds new ones, so nothing can follow what "
        f"became what — §05.2")
    for owning in ("createElement", "appendChild", "textContent"):
        assert owning in region, f"the deck does not use {owning} at all"


def test_a_card_survives_a_re_render_as_the_same_element():
    """**The claim.** Identity across a state change, driven in a real DOM.

    A stamp is written onto the card's element after the first render; the page
    is re-rendered; the stamp must still be there. `innerHTML` would have
    discarded the element carrying it.
    """
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")

    client, pid = _sealed()
    chosen = _select(client, pid)
    out = PH.run(
        "for (var i = 0; i < 12; i++) "
        "  await new Promise(function(r){ setTimeout(r, 0); });\n"
        "var host = __harness.el('studyModels');\n"
        "var first = host ? host.querySelectorAll('.deck-card') : [];\n"
        "for (var j = 0; j < first.length; j++) first[j].dataset.stamp = 'j' + j;\n"
        # THE RE-RENDER IS DRIVEN THROUGH A REAL CONTROL, not by calling the
        # renderer. `renderDeck` lives inside the page's IIFE and is not in
        # scope here, so the first draft's `if (typeof renderDeck ===
        # 'function') renderDeck();` never fired and this test passed without
        # re-rendering anything — TEST-059's shape, in the test written to
        # check retention. Toggling the study panel twice runs `renderStudy`,
        # which is the path a user takes.
        "__harness.dispatch('click', __harness.target({'data-study-toggle': '1'}));\n"
        "await new Promise(function(r){ setTimeout(r, 0); });\n"
        "__harness.dispatch('click', __harness.target({'data-study-toggle': '1'}));\n"
        "for (var q = 0; q < 6; q++) "
        "  await new Promise(function(r){ setTimeout(r, 0); });\n"
        "var after = host ? host.querySelectorAll('.deck-card') : [];\n"
        "var kept = 0, stamps = [];\n"
        "for (var k = 0; k < after.length; k++){\n"
        "  stamps.push(after[k].dataset.stamp || '');\n"
        "  if (after[k].dataset.stamp) kept++;\n"
        "}\n"
        "__emit({before: first.length, after: after.length, kept: kept,"
        "        stamps: stamps});",
        routes=_routes(client, pid), search=f"?project={pid}")

    assert out["before"] >= 2, (
        f"only {out['before']} cards rendered for {len(chosen)} selected "
        f"models, so retention is being checked over almost nothing")
    assert out["after"] == out["before"], (
        f"the deck had {out['before']} cards and has {out['after']} after a "
        f"re-render")
    assert out["kept"] == out["before"], (
        f"{out['before'] - out['kept']} of {out['before']} cards lost the "
        f"stamp written onto them, so they are NEW ELEMENTS. That is "
        f"destroy-and-replace: nothing can follow what became what. Stamps "
        f"after the re-render: {out['stamps']}")


def test_a_card_is_moved_rather_than_rebuilt_when_the_deck_reorders():
    """The mechanism face 3 needs, exercised now so it is not assumed later.

    Re-appending the cards in a new order must change their order while every
    element stays the one it was — which is precisely what a FLIP or a view
    transition needs and what `innerHTML` cannot give.

    **The move is performed as detach-then-append rather than as a bare
    re-append, and that is the harness's constraint rather than the design's.**
    In a browser `appendChild` on an attached node moves it; the harness's DOM
    appends a copy (`TEST-066`), so a bare re-append doubles the deck. The
    renderer does the same detach-and-refill for the same reason, and the
    cards are the same objects either way — which is the claim.
    """
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")

    client, pid = _sealed()
    _select(client, pid)
    out = PH.run(
        "for (var i = 0; i < 12; i++) "
        "  await new Promise(function(r){ setTimeout(r, 0); });\n"
        "var deck = __harness.el('studyModels').querySelector('.deck');\n"
        "var cards = Array.prototype.slice.call("
        "  deck.querySelectorAll('.deck-card'));\n"
        "for (var j = 0; j < cards.length; j++) cards[j].dataset.stamp = 'j' + j;\n"
        "var before = cards.map(function(c){ return c.getAttribute('data-model'); });\n"
        "while (deck.lastChild) deck.removeChild(deck.lastChild);\n"
        "cards.slice().reverse().forEach(function(c){ deck.appendChild(c); });\n"
        "var now = Array.prototype.slice.call(deck.querySelectorAll('.deck-card'));\n"
        "__emit({before: before,"
        "        after: now.map(function(c){ return c.getAttribute('data-model'); }),"
        "        stamps: now.map(function(c){ return c.dataset.stamp || ''; }),"
        "        count: now.length});",
        routes=_routes(client, pid), search=f"?project={pid}")

    assert out["count"] >= 2, "fewer than two cards; a reorder means nothing"
    assert out["after"] == list(reversed(out["before"])), (
        f"the deck did not reorder: {out['before']} -> {out['after']}")
    assert all(out["stamps"]), (
        f"a card lost its stamp during the reorder, so it was rebuilt rather "
        f"than moved: {out['stamps']}")


# ── face one · what it says, and where the words come from ───────────────────

def test_the_card_quotes_the_recipe_records_reason():
    """L36's rule, third surface. The page composes none of this prose."""
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")

    client, pid = _sealed()
    _select(client, pid)
    recipes = client.get(f"/project/{pid}/recipes").json()
    models = recipes.get("models") or {}
    assert models, "no per-model recipes after the seal; nothing to quote"

    out = PH.run(
        "for (var i = 0; i < 12; i++) "
        "  await new Promise(function(r){ setTimeout(r, 0); });\n"
        "__emit({text: __harness.el('studyModels').textContent});",
        routes=_routes(client, pid), search=f"?project={pid}")
    shown = " ".join((out["text"] or "").split())
    assert shown, "the deck rendered no text"

    quoted = 0
    for key, rows in models.items():
        for row in rows or []:
            reason = " ".join(str(row.get("reason") or "").split())
            if len(reason) > 40 and reason in shown:
                quoted += 1
    assert quoted >= 3, (
        f"only {quoted} of the record's reason strings appear verbatim in the "
        f"deck. A card that paraphrases is a second copy of the claim, which "
        f"L36 and L53-B have both ruled on")


def test_the_card_shows_the_capability_that_makes_this_model_differ():
    """The pedagogical payload: preprocessing is not one pipeline.

    Where the record chose a row by capability rather than by a wildcard, that
    selector is the engine's own account of why this model's pipeline differs
    from its neighbour's, so the card shows it.
    """
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")

    client, pid = _sealed()
    _select(client, pid)
    models = (client.get(f"/project/{pid}/recipes").json().get("models") or {})
    caps = sorted({str(r.get("selector")) for rows in models.values()
                   for r in (rows or [])
                   if str(r.get("selector", "")).startswith("caps:")})
    assert caps, (
        "no row in this project's recipes was chosen by a capability, so this "
        "fixture cannot show a per-model difference and the assertion below "
        "would be vacuous")

    out = PH.run(
        "for (var i = 0; i < 12; i++) "
        "  await new Promise(function(r){ setTimeout(r, 0); });\n"
        "__emit({text: __harness.el('studyModels').textContent});",
        routes=_routes(client, pid), search=f"?project={pid}")
    shown = out["text"] or ""
    assert any(c in shown for c in caps), (
        f"none of {caps} reaches the card. That selector is why this model's "
        f"pipeline differs from the one beside it, and the deck exists to "
        f"make exactly that visible")


def test_a_model_with_no_resolved_recipe_says_what_is_missing():
    """L52's vocabulary, kept: never a blank card."""
    from turbotab import reporting_checklist  # noqa: F401  (house style import)

    page = PAGE.read_text(encoding="utf-8")
    start = page.index("function deckFill(")
    body = page[start:page.index("function modelName(")]
    assert "has not resolved" in body, (
        "a model whose recipe did not resolve gets an empty card body with no "
        "sentence, which is the blank cell `GUIDED-179` was filed for")
    assert "deckClear(body)" in body, (
        "the empty branch does not clear the previous model's rows, so a card "
        "that stops resolving keeps showing the last recipe it had")


# ── B3 · faces two and three are SPECIFIED, not built ────────────────────────

def test_the_unbuilt_faces_are_registered_rather_than_described():
    """Two, never three, when testing an abstraction.

    The `Pending` shape `figure_specs.py` uses: written down, blocker named,
    nothing asserted that is not built. A pending whose blocker is unnamed is a
    wish, so both fields are required rather than encouraged.
    """
    from turbotab import deck_faces as F

    assert len(F.FACES) == 3, F.FACES
    assert [f.key for f in F.built()] == ["preprocess"], (
        f"{[f.key for f in F.built()]} claim to be built. Faces 2 and 3 are "
        f"specified this loop and not built, and a face that says otherwise is "
        f"the misreport DRIVE-009 was just corrected for")
    for face in F.pending():
        assert len(face.needs.split()) >= 25, (
            f"{face.key}'s `needs` is too thin to build from: {face.needs!r}")
        assert face.blocked_by, f"{face.key} names no blocker"
    train = next(f for f in F.pending() if f.key == "train")
    assert "cumulative-link" in train.blocked_by, (
        "the Train face does not record that the domain lens never reaches "
        "the model shelf, which is the thing that blocks its survey case")


def test_the_scroll_correction_is_recorded_where_it_can_be_read():
    """The one change made to the product owner's framing, in the source.

    He said the face transforms WHEN YOU SCROLL. §05 rule 5 forbids ambient
    animation, so the face changes because the analysis advanced and scrolling
    only takes you to where that happened. A change to someone's design that
    lives only in a report is a change they cannot find later.
    """
    from turbotab import deck_faces as F

    doc = F.__doc__ or ""
    assert "not because you scrolled" in doc, doc[:200]
    assert "rule 5" in doc and "Nothing else moves" in doc, (
        "the correction is stated without the rule it rests on, so a reader "
        "cannot tell whether it was a preference or a constitutional point")

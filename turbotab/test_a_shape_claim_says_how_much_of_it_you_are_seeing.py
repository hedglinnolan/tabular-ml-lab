"""`GUIDED-164` — three plots, five chips, and neither of them labeled.

A shape claim ("23 features with outliers") renders a sample of what it is about
and never says it is a sample. Two caps, both silent:

* `attachSkewPlots` takes `affected_columns.slice(0, 3)` — **three** plots;
* `findingCard` takes `affected_columns.slice(0, 5)` — **five** chips.

So on `wide_assay.csv` a card headed *23 features with outliers* draws three
distributions and names five columns, and nothing on it distinguishes that from
a card whose finding really is about three and five. §09's recorded-absence rule
runs in this direction too: **a truncation nobody records reads as a complete
answer**, and here it reads as one that contradicts the card's own title.

And the way out already existed. `GET /project/<id>/evidence/histograms` has
served `page`, `n_pages`, `per_page` and `n_features` since `GUIDED-005`; the
page's click delegate has handled `data-hist-page` for as long. The only control
that ever carried that attribute was the gallery's own Previous/Next — which
cannot be pressed until the gallery is open. **The one card that motivates a
per-feature gallery was the one surface with no way into it**, which is trap 1
in its ordinary form: the capability shipped, its consumer did not.

## The numbers this file drives, not the ones it was briefed with

The brief described `n_features: 26, per_page: 6, n_pages: 5` and a finding
about fifteen columns. **No fixture in `turbotab/sample_data/` produces those**
— re-derived rather than quoted forward, and the real ones are in `CASES`
below. The shape of the defect is exactly as described; the arithmetic is not.

Two fixtures of different target shape (`GUIDED-097`): `wide_assay.csv` and
`clinical_labs.csv` are classification, `metabolomics_untargeted.csv` is
regression on continuous `bmi`. They also straddle the gallery's own gate:
`wide_assay` has 46 numeric columns and pages, `metabolomics_untargeted` has
396 and is refused with a reason — so the card's press is driven into both the
built answer and the refusal.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent / "sample_data"

#: NOT COVERED, said out loud.
SHAPES_NOT_COVERED = (
    "A multiclass target. Both classification fixtures here are binary; the "
    "shape claim and the gallery read numeric FEATURE columns and never the "
    "target, so the number of classes is not on this path.",
    "Whether the gallery is on screen once opened. It renders into "
    "`#paletteBox`, further down the same section, and the page never moves "
    "the viewport (`DRIVE-006`) — so this proves the press reaches the pager "
    "and renders it, and says nothing about where the user's eye is.",
    "A finding whose `shape.has_chips` is false — a cohort-level finding with "
    "no column subject at all. It takes the other branch of the chip row, "
    "which `GUIDED-053` already covers and this change does not touch.",
    "A skew/distribution finding that falls back to `skewCandidates()` "
    "because it carries no `affected_columns`. No shipped fixture produces "
    "one, so the fallback's count is read from the same pool but is not "
    "driven here.",
)

#: fixture -> (target, task shape, finding id, columns the finding is about,
#:             gallery availability)
CASES = {
    "wide_assay.csv": dict(
        target="responder", shape="classification", finding="profile_outliers_3",
        n_columns=23, gallery=True, n_features=46, per_page=6, n_pages=8),
    "metabolomics_untargeted.csv": dict(
        target="bmi", shape="regression", finding="profile_outliers_4",
        n_columns=373, gallery=False, n_features=396, per_page=None,
        n_pages=None),
    "clinical_labs.csv": dict(
        target="readmitted", shape="classification", finding="profile_outliers_3",
        n_columns=4, gallery=True, n_features=9, per_page=6, n_pages=2),
}

_PATHS = ("interview?step=data", "interview?step=explore",
          "interview?step=features", "capabilities", "features", "recipes",
          "preprocess", "figures", "draft", "manuscript", "models", "training",
          "instability", "explain", "sensitivity", "evidence/plausibility",
          "evidence/missingness")

#: THE PRESS IS BUILT FROM THE RENDER, NEVER HAND-SPECIFIED (trap 3).
#: A synthetic `{'data-hist-page': '0'}` would drive the delegate whether or
#: not the card ever emits such a button, which is the whole defect supplied by
#: the fixture. This reads the attributes off the button the card actually drew.
_PRESS_FROM_RENDER = """
function pressed(html, attr){
  var re = /<button\\b([^>]*)>/g, m;
  while ((m = re.exec(html))){
    var raw = m[1], attrs = {}, a = /([a-zA-Z-]+)="([^"]*)"/g, k;
    while ((k = a.exec(raw))) attrs[k[1]] = k[2];
    if (attrs[attr] !== undefined) return attrs;
  }
  return null;
}
"""


def _project(fixture):
    from fastapi.testclient import TestClient

    from turbotab import api

    client = TestClient(api.app)
    with (DATA / fixture).open("rb") as handle:
        pid = client.post("/project", files={
            "file": (fixture, handle, "text/csv")}).json()["id"]
    resp = client.post(f"/project/{pid}/decision", json={
        "kind": "set_target",
        "payload": {"column": CASES[fixture]["target"]}})
    assert resp.status_code < 400, f"set_target refused: {resp.text[:200]}"
    return client, pid


def _routes(client, pid):
    out = {f"/project/{pid}": client.get(f"/project/{pid}").json()}
    for path in _PATHS:
        resp = client.get(f"/project/{pid}/{path}")
        out[f"/project/{pid}/{path}"] = (resp.json() if resp.status_code == 200
                                         else {})
    # The gallery's first page, answered from the real endpoint, because the
    # press below is what fetches it.
    gal = client.get(f"/project/{pid}/evidence/histograms?page=0")
    out[f"/project/{pid}/evidence/histograms?page=0"] = gal.json()
    return out


def _finding(client, pid, fixture):
    want = CASES[fixture]["finding"]
    for f in client.get(f"/project/{pid}").json()["findings"]:
        if f["id"] == want:
            return f
    raise AssertionError(
        f"{fixture} no longer serves `{want}`, so this file is driving a card "
        f"the fixture does not render")


def _drive(fixture, press=False):
    from turbotab import pageharness as PH

    client, pid = _project(fixture)
    finding = _finding(client, pid, fixture)
    assert len(finding["affected_columns"]) == CASES[fixture]["n_columns"], (
        f"{fixture}'s `{finding['id']}` is now about "
        f"{len(finding['affected_columns'])} columns, not "
        f"{CASES[fixture]['n_columns']}; the arithmetic below is pinned to it")
    routes = _routes(client, pid)

    body = (_PRESS_FROM_RENDER +
            "var fid = %(fid)s;\n"
            "var ev = __harness.html('ev-' + fid) || '';\n"
            "var cards = ['profList', 'profRest', 'structList']\n"
            "  .map(function(id){ return __harness.html(id) || ''; }).join('');\n"
            "var btn = pressed(ev, 'data-hist-page');\n" % {
                "fid": json.dumps(finding["id"])})
    if press:
        body += ("if (btn) __harness.dispatch('click', __harness.target(btn));\n"
                 "for (var i = 0; i < 10; i++) "
                 "await new Promise(function(r){ setTimeout(r, 0); });\n")
    body += ("__emit({ev: ev, cards: cards, btn: btn,\n"
             "        palette: __harness.html('paletteBox'),\n"
             "        calls: __harness.calls().map(function(c){\n"
             "          return c.method + ' ' + c.path; })});")

    out = PH.run(body, routes=routes, search=f"?project={pid}")
    out["pid"] = pid
    out["finding"] = finding
    return out


def _card(html, finding_id):
    """The one card's markup, cut out of the stack it was rendered into."""
    open_at = html.find(f'id="find-{finding_id}"')
    assert open_at != -1, (
        f"no card with id `find-{finding_id}` is in the Explore stack, so "
        f"nothing below is reading the finding it claims to")
    start = html.rfind("<article", 0, open_at)
    end = html.find("</article>", open_at)
    return html[start:end]


# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("fixture", sorted(CASES), ids=[
    "classification · 4 columns, nothing truncated",
    "regression · 373 columns, gallery refused",
    "classification · 23 columns, gallery pages"])
def test_the_chip_row_says_how_many_columns_of_how_many(fixture):
    """Five chips for twenty-three columns, and no way to tell.

    Asserted on both branches: a truncated row has to say it is truncated, and
    a whole row has to say it is whole, or the reader is inferring one from the
    other's silence.
    """
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")

    out = _drive(fixture)
    case = CASES[fixture]
    card = _card(out["cards"], case["finding"])

    named = re.findall(r'<span class="chip">([^<]*)</span>', card)
    columns = [c for c in named if c in out["finding"]["affected_columns"]]
    assert len(columns) == min(5, case["n_columns"]), (
        f"the card names {len(columns)} columns; the cap is five: {named}")

    count = re.search(r'data-chip-showing="(\d+)" data-chip-of="(\d+)">'
                      r'([^<]*)</span>', card)
    assert count, (
        f"the chip row names {len(columns)} of {case['n_columns']} columns and "
        f"says nothing about the other {case['n_columns'] - len(columns)}. A "
        f"card titled {out['finding']['title']!r} sitting above five column "
        f"names is a truncation the reader has to detect by arithmetic. The "
        f"row as rendered: {named}")
    assert int(count.group(1)) == len(columns), (
        f"the chip says it is showing {count.group(1)} and {len(columns)} are "
        f"rendered — trap 7, the machine-readable count disagreeing with what "
        f"is beside it")
    assert int(count.group(2)) == case["n_columns"], (
        f"the chip says the finding is about {count.group(2)} columns; the "
        f"record says {case['n_columns']}")
    assert str(case["n_columns"]) in count.group(3), (
        f"the count is in the attributes and not in the sentence a person "
        f"reads: {count.group(3)!r}")


@pytest.mark.parametrize("fixture", sorted(CASES), ids=[
    "classification · 4 columns, nothing truncated",
    "regression · 373 columns, gallery refused",
    "classification · 23 columns, gallery pages"])
def test_the_plot_block_says_how_many_distributions_of_how_many(fixture):
    """Three plots, and the block has to name the pool they came out of."""
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")

    out = _drive(fixture)
    case = CASES[fixture]
    ev = out["ev"]

    drawn = ev.count('class="sm-plot"') or ev.count('<div class="ev">')
    assert drawn == min(3, case["n_columns"]), (
        f"the evidence slot for `{case['finding']}` holds {drawn} "
        f"distributions where three are expected. It rendered: {ev[:400]!r}")

    said = re.search(r'data-plots-showing="(\d+)" data-plots-of="(\d+)">'
                     r'([^<]*)</span>', ev)
    assert said, (
        f"the block draws {drawn} of {case['n_columns']} distributions and "
        f"labels neither number, so a sample and a complete set render "
        f"identically. The slot holds: {ev[:400]!r}")
    assert int(said.group(1)) == drawn, (
        f"the block says it is showing {said.group(1)} and {drawn} are drawn")
    assert int(said.group(2)) == case["n_columns"], (
        f"the block says the claim is about {said.group(2)} columns; the "
        f"record says {case['n_columns']}")
    assert str(case["n_columns"]) in said.group(3), (
        f"the pool size is in the attributes and not in the sentence: "
        f"{said.group(3)!r}")


@pytest.mark.parametrize("fixture", sorted(CASES), ids=[
    "classification · 4 columns, nothing truncated",
    "regression · 373 columns, gallery refused",
    "classification · 23 columns, gallery pages"])
def test_pressing_the_card_reaches_the_gallery_the_card_motivates(fixture):
    """The consumer, pressed, and the pager observed answering it.

    The press is built from the button the card drew, so a card that stops
    emitting one produces no press — trap 3. The consequence in the name is
    checked twice over: the fetch the delegate issues, and what the palette
    renders when it comes back.
    """
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")

    out = _drive(fixture, press=True)
    case = CASES[fixture]

    assert out["btn"], (
        f"the card for `{case['finding']}` renders no control that reaches the "
        f"histogram pager, so the one card a per-feature gallery is FOR is the "
        f"one surface that cannot open it. The slot holds: {out['ev'][:400]!r}")
    assert out["btn"].get("data-hist-from") == case["finding"], (
        f"the press does not say which card made it: {out['btn']!r}")

    want = f"GET /project/{out['pid']}/evidence/histograms?page=0"
    assert want in out["calls"], (
        f"the press did not reach the pager. The delegate handles "
        f"`data-hist-page`; what the drive fetched was: "
        f"{[c for c in out['calls'] if 'histogram' in c][:6]}")

    palette = out["palette"] or ""
    if case["gallery"]:
        assert f"{case['n_features']} numeric features" in palette, (
            f"the gallery came back and the palette does not say what it is "
            f"showing: {palette[:400]!r}")
        assert f"of {case['n_pages']}" in palette, (
            f"the gallery renders no page count: {palette[:400]!r}")
    else:
        # The refusal is the server's sentence, and it is the right answer for
        # 396 numeric columns — the press must reach it rather than a wall of
        # plots or an empty panel.
        assert str(case["n_features"]) in palette and "50" in palette, (
            f"the gallery is refused above {case['n_features']} features and "
            f"the palette does not carry the server's reason: "
            f"{palette[:400]!r}")

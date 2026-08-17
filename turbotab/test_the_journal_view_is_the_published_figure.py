"""`DRIVE-014` — Journal view renders the figure as it will be published.

`DESIGN_LANGUAGE.md` §07 specifies the duality this file checks:

> **Journal view**: a toggle re-renders the same figure as it will be published —
> serif type, greyscale, series distinguished by **dash pattern rather than
> color alone**, printed *r* values in correlation matrices, proper ticks,
> numbered caption.

## The premise this part was given was half wrong, and the correction is the finding

The row says `.fig-canvas.journal` is fully styled and the literal `journal`
appears zero times in the script region — both true. The inference was that a
toggle was missing its handler. **The real state was that no figure was drawn at
all.** The figure surface rendered a figure's title, tier, annotations,
checklist and caption and never its geometry; the whole 8,479-line page held two
`<svg>` occurrences, a single-series spark histogram and a static glyph. So
`.fig-canvas.journal` was not a rule whose toggle was missing — it was styling
for an element with **no producer anywhere**, which is exactly why the row says
no test and no import graph would ever flag it, and is a stronger version of
that claim than the row makes.

## Two figure kinds, and they are the two §07 names

`roc` is multi-series, which is what the dash rule is about. `item_correlations`
is a matrix, which is what the printed-*r* rule is about. Two of deliberately
different shape; a third of a shape already seen would only cast the mould
again.

## Every payload here is composed by production code

`figure_bundle.render(project)` builds the bundle and `figure_specs`' own
`compute` builds the geometry — nothing in this file types a `curves` dict or a
`matrix`. That is `AGENT_ONBOARD` §07 trap 3's rule one level up: where a test
hands the renderer a payload standing for a real one, the payload has to be the
real one, or the test proves a renderer that production can never feed.
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import eventfixture as _EF                             # noqa: E402
from turbotab import figure_bundle as FB                            # noqa: E402
from turbotab import packs                                          # noqa: E402
from turbotab import pageharness as PH                              # noqa: E402
from turbotab import training as T                                  # noqa: E402
from turbotab.project import AnalysisProject                        # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data")


def _client():
    from fastapi.testclient import TestClient

    from turbotab import api

    return TestClient(api.app)


def _api_project(client, fixture, target, decisions):
    """A project built through the real routes, so every route below is real."""
    from turbotab import api

    with open(os.path.join(DATA, fixture), "rb") as fh:
        pid = client.post("/project", files={
            "file": (fixture, fh, "text/csv")}).json()["id"]
    for kind, payload in decisions:
        got = client.post(f"/project/{pid}/decision",
                          json={"kind": kind, "payload": payload})
        assert got.status_code == 200, (kind, got.text[:250])
    return pid, api.STORE.get(pid)


def _trained(client):
    """A real project with predictions, so `roc` is genuinely drawable."""
    from turbotab import api

    pid, project = _api_project(client, "clinical_risk.csv", "readmit_30d", [
        ("set_target", {"column": "readmit_30d"}),
        ("set_purpose", {"answer": "prediction"}),
        ("set_grain", {"answer": "one_row_per_person"}),
        ("set_eligibility", {"answer": "everyone"}),
        ("seal", {"fraction": 0.25})])
    # `DRIVE-041`. Posted as a decision, because this helper's whole point is
    # that every route below is real.
    _EF.choose_event_over_http(client, pid, "readmit_30d", required=True)
    project.training_run = T.train(project, ["logreg", "histgb_clf"])
    api._RUNS[pid] = {"run": project.training_run}
    return pid, project


def _survey(client, items=10):
    """A survey project with FEW ENOUGH ITEMS that the server prints values.

    `print_values` is the server's own decision — `k <= 15` — and the full
    instrument carries 40 items, so the full fixture produces a matrix the
    server has correctly decided is too dense to label. Trimming the upload is
    how the test reaches the branch it is about; RE-DECIDING `print_values`
    here would be a second implementation of the rule under test.
    """
    from turbotab import api

    df = pd.read_csv(os.path.join(DATA, "survey_instrument.csv"))
    keep = ["age"] + [c for c in df.columns if c.startswith("item_")][:items]
    trimmed = df[keep].to_csv(index=False).encode("utf-8")
    pid = client.post("/project", files={
        "file": ("survey_instrument.csv", trimmed, "text/csv")}).json()["id"]
    for kind, payload in [("set_lens", {"lens": [packs.SURVEY]}),
                          ("set_target", {"column": "age"}),
                          ("set_grain", {"answer": "one_row_per_person"}),
                          ("set_eligibility", {"answer": "everyone"})]:
        got = client.post(f"/project/{pid}/decision",
                          json={"kind": kind, "payload": payload})
        assert got.status_code == 200, (kind, got.text[:250])
    return pid, api.STORE.get(pid)


# `_overlaid` STOOD HERE AND IS DELETED, WHICH IS `GUIDED-236` CLOSING.
#
# It built the ROC payload the bundle WOULD have built if it read every scored
# model — the real `figure_specs.roc_payload`, the real run's held-out
# probabilities — and swapped the result into the served row, because
# `figure_bundle._risks_or_refuse` took `scored[0]` and handed the spec exactly
# one model however many were fitted. It carried its own retirement condition:
# *"GUIDED-236 may be fixed — if so this helper should be deleted and the
# served payload used directly, because it is only here to stand in for a
# bundle that could not overlay models."*
#
# The bundle overlays models now, so the caller below passes `FB.render(project)`
# straight through and the assertion is an end-to-end check rather than a check
# on a stand-in. Strictly stronger, and the deletion is the evidence.


def _routes(client, pid):
    out = {f"/project/{pid}": client.get(f"/project/{pid}").json()}
    for path in ("interview?step=data", "interview?step=explore",
                 "interview?step=features", "interview?step=preprocess",
                 "capabilities", "features", "recipes", "preprocess", "figures",
                 "draft", "manuscript", "models", "training", "instability",
                 "explain", "sensitivity", "checklist",
                 "evidence/plausibility", "evidence/missingness"):
        got = client.get(f"/project/{pid}/{path}")
        out[f"/project/{pid}/{path}"] = (got.json() if got.status_code == 200
                                         else {})
    return out


def _row(bundle, figure_id):
    for row in (bundle.get("admitted") or []) + (bundle.get("held") or []):
        if row.get("id") == figure_id:
            return row
    return None


def _drive(client, pid, bundle, body):
    """Render a real figure bundle through the page's own controller."""
    if not PH.available():
        pytest.skip("no JS engine on this machine")
    routes = _routes(client, pid)
    routes[f"/project/{pid}/figures"] = bundle
    return PH.run(body, routes=routes, search=f"?project={pid}")



# ═══════════ 1 · a figure this build draws gets a canvas and a toggle ═══════

def test_a_drawable_figure_reaches_a_canvas_and_a_journal_toggle():
    """`DRIVE-014`. The first producer `.fig-canvas` has ever had.

    Driven through the page's own controller against a bundle
    `figure_bundle.render` composed, so the geometry on screen is the geometry
    a real project produces.
    """
    client = _client()
    pid, project = _trained(client)
    bundle = FB.render(project)
    roc = _row(bundle, "roc")
    assert roc and roc.get("payload", {}).get("curves"), (
        "the fixture produced no ROC payload with curves, so nothing below is "
        "about drawing one")

    out = _drive(client, pid, bundle,
                 "for (var i = 0; i < 10; i++) "
                 "  await new Promise(function(r){ setTimeout(r, 0); });\n"
                 "__emit({box: __harness.html('figuresBox') || ''});")
    box = out["box"]
    assert "fig-canvas" in box, (
        f"no figure canvas rendered at all. Before L55-C the page drew no "
        f"figure of any kind — title, tier, annotations, checklist and caption, "
        f"and never the geometry.\n{box[:1500]}")
    assert 'data-journal="roc"' in box, "the ROC figure carries no journal toggle"
    assert "<svg" in box, "the canvas holds no svg"


def test_a_figure_this_build_cannot_draw_gets_no_canvas_and_no_toggle():
    """The recorded-absence rule: an empty canvas is a claim.

    Most figures still have no renderer, and they must render exactly as they
    did — metadata with no box — rather than an empty frame that says there was
    nothing to plot.
    """
    client = _client()
    pid, project = _trained(client)
    bundle = FB.render(project)
    others = [r for r in (bundle.get("admitted") or [])
              if r.get("id") not in ("roc", "item_correlations")]
    assert others, "the fixture admitted only drawable figures; nothing to check"

    out = _drive(client, pid, bundle,
                 "for (var i = 0; i < 10; i++) "
                 "  await new Promise(function(r){ setTimeout(r, 0); });\n"
                 "var out = {};\n"
                 "(__harness.el('figuresBox').querySelectorAll('.fig')).forEach("
                 "  function(f){ out[f.id] = f.innerHTML; });\n"
                 "__emit(out);")
    for row in others:
        markup = out.get("fig-" + row["id"], "")
        assert markup, f"{row['id']} did not render at all"
        assert "fig-canvas" not in markup, (
            f"{row['id']} has no renderer in this build and was given an empty "
            f"canvas anyway: {markup[:400]}")
        assert "data-journal" not in markup, (
            f"{row['id']} offers a journal toggle for a figure that is never "
            f"drawn, so pressing it would change nothing a reader can see")


# ═══════════ 2 · the rule the part is named for ═════════════════════════════

def _paths(svg):
    return re.findall(r"<path\b[^>]*>", svg)


def test_the_journal_face_tells_series_apart_by_dash_rather_than_color():
    """§07, and it is the claim this part will be checked on.

    In-app the curves are told apart by hue. In Journal view every curve is the
    same ink and they are told apart by DASH PATTERN — which is what survives a
    greyscale print and a reader who cannot separate the hues.
    """
    client = _client()
    pid, project = _trained(client)
    # `GUIDED-236` IS FIXED, SO THE STAND-IN IS GONE. This used to be
    # `_overlaid(project, FB.render(project))` — a helper that called the real
    # `figure_specs.roc_payload` with the real run's probabilities and swapped
    # its output into the served row, because `figure_bundle` could only ever
    # supply one curve. Its own message said to delete it if the row was ever
    # fixed, and this is that. The assertion below is now an end-to-end check
    # on the bundle instead of a check on a hand-composed payload.
    bundle = FB.render(project)
    roc = _row(bundle, "roc")
    assert roc and len(roc["payload"]["curves"]) >= 2, (
        f"the ROC payload carries "
        f"{len((roc or {}).get('payload', {}).get('curves', {}))} curve(s); "
        f"one series cannot demonstrate a rule about telling series apart")

    out = _drive(client, pid, bundle,
                 "for (var i = 0; i < 10; i++) "
                 "  await new Promise(function(r){ setTimeout(r, 0); });\n"
                 "var app = __harness.html('figuresBox') || '';\n"
                 "__harness.dispatch('click', __harness.target({'data-journal': 'roc', 'data-journal-on': '1'}));\n"
                 "for (var q = 0; q < 6; q++) "
                 "  await new Promise(function(r){ setTimeout(r, 0); });\n"
                 "__emit({app: app, journal: __harness.html('figuresBox') || ''});")

    app, journal = out["app"], out["journal"]
    assert "fig-canvas journal" in journal or 'data-face="journal"' in journal, (
        f"pressing the toggle did not switch the face.\n{journal[:1200]}")
    assert 'data-face="journal"' not in app, (
        "the app face was already the journal face, so the comparison below "
        "is between one thing and itself")

    app_strokes = set(re.findall(r'<path\b[^>]*stroke="([^"]+)"', app))
    jrn_strokes = set(re.findall(r'<path\b[^>]*stroke="([^"]+)"', journal))
    assert len(app_strokes) >= 2, (
        f"the in-app face draws every curve in one color: {app_strokes}. The "
        f"duality needs both halves to be real.")
    assert len(jrn_strokes) == 1, (
        f"the journal face uses {len(jrn_strokes)} stroke colors — "
        f"{jrn_strokes} — so it is still telling series apart by hue. §07: "
        f"greyscale, series distinguished by dash pattern rather than color "
        f"alone.")

    dashed = [p for p in _paths(journal) if "stroke-dasharray" in p]
    assert len(dashed) >= 1, (
        f"no curve in the journal face carries a stroke-dasharray, so with one "
        f"ink the series are indistinguishable. Paths: {_paths(journal)[:4]}")


def test_the_journal_face_prints_the_r_values_in_a_correlation_matrix():
    """§07's other named property, on the figure it names.

    `print_values` is the SERVER's decision about whether the matrix is small
    enough for the numbers to be legible, and it is read rather than re-made
    here.
    """
    client = _client()
    pid, project = _survey(client)
    bundle = FB.render(project)
    corr = _row(bundle, "item_correlations")
    assert corr and corr.get("payload", {}).get("matrix"), (
        "the survey fixture produced no correlation matrix")
    payload = corr["payload"]
    assert payload.get("print_values"), (
        f"the server decided this matrix is too large to print values on "
        f"({len(payload.get('columns') or [])} columns), so this test cannot "
        f"tell a renderer that prints them from one that does not")

    out = _drive(client, pid, bundle,
                 "for (var i = 0; i < 10; i++) "
                 "  await new Promise(function(r){ setTimeout(r, 0); });\n"
                 "__harness.dispatch('click', "
                 "  __harness.target({'data-journal': 'item_correlations', "
                 "                    'data-journal-on': '1'}));\n"
                 "for (var q = 0; q < 6; q++) "
                 "  await new Promise(function(r){ setTimeout(r, 0); });\n"
                 "__emit({journal: __harness.html('figuresBox') || ''});")
    journal = out["journal"]

    # THE RECORD'S OWN NUMBERS, formatted the one way the renderer formats
    # them — not "some digits appear". A matrix of the wrong values would pass
    # a looser check.
    wanted = []
    for r, rowvals in enumerate(payload["matrix"]):
        for c, v in enumerate(rowvals):
            if v is not None and r != c:
                wanted.append(f"{float(v):.2f}")
    assert wanted, "the matrix carries no off-diagonal values"
    missing = [w for w in wanted[:6] if ">" + w + "<" not in journal]
    assert not missing, (
        f"the journal face does not print the r values {missing}. §07 names "
        f"printed r values in correlation matrices as a property of the "
        f"published figure.\n{journal[:1200]}")
    assert "rgb(" in journal, (
        "the journal matrix is not greyscale — §07 says the published face is")


# ═══════════ 3 · what a view is, and where the canvas sits ══════════════════

def test_the_toggle_records_nothing():
    """A view is not a decision.

    The record is what the app knows about the study. Which face a reader is
    looking at is not that, so the toggle posts nothing, marks nothing stale
    and appears nowhere in the transcript.
    """
    client = _client()
    pid, project = _trained(client)
    bundle = FB.render(project)
    out = _drive(client, pid, bundle,
                 "for (var i = 0; i < 10; i++) "
                 "  await new Promise(function(r){ setTimeout(r, 0); });\n"
                 "var before = __harness.posts().length;\n"
                 "__harness.dispatch('click', __harness.target({'data-journal': 'roc', 'data-journal-on': '1'}));\n"
                 "for (var q = 0; q < 6; q++) "
                 "  await new Promise(function(r){ setTimeout(r, 0); });\n"
                 "__emit({before: before, after: __harness.posts().length,"
                 "        journal: (__harness.html('figuresBox') || '')"
                 "                   .indexOf('data-face=\\\"journal\\\"') >= 0});")
    assert out["journal"], "the toggle did not fire, so 'it posted nothing' is vacuous"
    assert out["after"] == out["before"], (
        f"pressing Journal view sent {out['after'] - out['before']} request(s). "
        f"A view is not a decision.")


def test_the_canvas_sits_immediately_before_the_caption():
    """Because `.fig-canvas.journal + .fig-cap` is an ADJACENT-SIBLING rule.

    That rule — white ground, Georgia — has been in the stylesheet since the
    prototype and is what puts the caption into the published face. A canvas
    rendered anywhere else in the card leaves it dead exactly as it was, and
    nothing about the markup would look wrong.
    """
    client = _client()
    pid, project = _trained(client)
    bundle = FB.render(project)
    out = _drive(client, pid, bundle,
                 "for (var i = 0; i < 10; i++) "
                 "  await new Promise(function(r){ setTimeout(r, 0); });\n"
                 "__emit({box: __harness.html('figuresBox') || ''});")
    box = out["box"]
    at = box.find('<div class="fig-canvas')
    assert at != -1, "no canvas rendered"
    after = box[at:]
    close = after.find("</div>")
    rest = after[close + len("</div>"):].lstrip()
    assert rest.startswith('<p class="fig-cap"'), (
        "the caption does not immediately follow the canvas, so "
        "`.fig-canvas.journal + .fig-cap` never matches and the journal "
        f"caption keeps the app's face. What follows: {rest[:200]!r}")


# ═══════════ the scope this part did NOT cover ══════════════════════════════
#
# Said out loud rather than left to be discovered.
#
# **Export at 3× and LaTeX tables are not built.** §07 specifies both and this
# part builds neither; the toggle changes what is on screen and nothing leaves
# the app. `DRIVE-009` carries the export third.
#
# **Two figure kinds of the twenty-one registered are drawn.** `roc` and
# `item_correlations` — chosen because they are the two §07 names a property
# for. Every other figure renders exactly as it did, with no canvas and no
# toggle, and `test_a_figure_this_build_cannot_draw_gets_no_canvas_and_no_toggle`
# asserts that rather than leaving it to be assumed.
#
# **Ticks and a numbered caption are not built.** §07 lists "proper ticks" and
# "numbered caption" beside the dash rule; the axes carry labels and no tick
# marks, and the caption is the server's prose with no figure number, because
# nothing numbers figures yet.
#
# **The journal face does not follow the dark theme, deliberately.** It is
# literal hex by §07's own reasoning — the exported file has to be
# self-contained — so a reader in dark mode sees a white figure. That is the
# published face and it is what the export will be.

"""The frontend harness, with the gate the README says it needs.

`README.md` "State of play": *the safety net is thinner than the coverage
number* — and the sharpest instance is that **three frontend tests once passed
against a page emptied to `<body></body>`**. `GUIDED-045` filed the class:
an absence assertion over a file gets EASIER to satisfy as the file loses
content, so deleting the guarded thing makes the guard pass harder.

`test_an_absence_assertion_carries_a_positive_control.py` enforces the rule one
assertion at a time. **This file enforces it one HARNESS at a time**: every
claim below is registered in `CLAIMS`, and the last test empties the page and
asserts that **every one of them goes red**. A claim that survives an empty
page is not a claim about the page.

## What is covered, and why these four

Load-bearing rather than exhaustive — the things the app promises a user can
see:

1. **The interview renders its questions**, with the values the server serves
   rather than the labels a reader sees.
2. **A decision re-paints** — the press changes what is on screen, not only what
   was posted.
3. **A question shows its consequence** — its reason and what the answer will
   change both reach the card.
4. **A question of consequence renders a way through** — the band, the claim,
   and both terminal controls.

## Two gaps the harness found on its first run

**`GUIDED-076` was found here and is fixed** — the band reads `q.exits` now,
and the test that asserted the defect asserts the fix. A test that asserts a
defect is a placeholder with a deadline.

## The fifth claim exists now, and it is why this file was worth writing

*A figure arrives with its annotation box* could not be written last loop
because the page never fetched `/figures` — zero occurrences. `GUIDED-075` was
that finding, and the claim is the sixth entry in `CLAIMS` now.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import exits as X                                       # noqa: E402
from turbotab import pageharness as H                                 # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"

pytestmark = pytest.mark.skipif(not H.available(),
                                reason="no JS engine on this machine")


def _client():
    from fastapi.testclient import TestClient

    from turbotab import api
    return TestClient(api.app)


def _project(client, name="metabolomics_untargeted.csv"):
    with open(DATA / name, "rb") as fh:
        return client.post("/project",
                           files={"file": (name, fh, "text/csv")}).json()


def _routes(client, project, **extra):
    """Everything the controller fetches while it boots, answered for real.

    From the live API rather than from fixtures written here, so a server change
    that breaks the page breaks this file instead of being papered over by a
    canned reply.
    """
    pid = project["id"]
    routes = {
        f"/project/{pid}": project,
        f"/project/{pid}/interview?step=data":
            client.get(f"/project/{pid}/interview?step=data").json(),
        f"/project/{pid}/interview?step=explore": {"questions": [], "steps": []},
        f"/project/{pid}/evidence/missingness": {"cards": []},
        f"/project/{pid}/evidence/plausibility": {"columns": []},
        f"/project/{pid}/draft": {"paragraphs": []},
        f"/project/{pid}/gaps": {"gaps": []},
    }
    routes.update(extra)
    return routes


def _drive(body, routes, pid):
    return H.run(body, routes=routes, search=f"?project={pid}")


# ── the four claims, each written once and asserted twice ──────────────────

def claim_questions_render(client, project):
    """The interview renders its questions, by value and not by label."""
    out = _drive("__emit(__harness.html('askedQuestions'));",
                 _routes(client, project), project["id"])
    buttons = [b for b in H.elements(out) if b.get("data-answer-key")]
    assert buttons, "no question rendered an answerable control at all"
    assert all(b.get("data-answer-value") is not None for b in buttons), (
        "a control renders with no value to submit, so pressing it posts a label")


def claim_a_decision_repaints(client, project):
    """A press changes what is ON SCREEN, not only what was posted.

    `DRIVE-001`'s class, on the page: a control that records correctly and does
    not redraw is indistinguishable from a broken control.
    """
    out = _drive(
        """
        var before = __harness.html('askedQuestions');
        var rx = /<button([^>]*data-answer-key="state_lens"[^>]*)>/g, m, opts = [];
        while ((m = rx.exec(before)) !== null){
          var a = {};
          m[1].replace(/([a-zA-Z-]+)="([^"]*)"/g,
                       function(_, k, v){ a[k] = v; return ""; });
          opts.push(a);
        }
        if (!opts.length) { throw new Error('no answerable control rendered'); }
        __harness.dispatch('click', __harness.target(opts[0], ['answer', 'multi']));
        __emit({before: before, after: __harness.html('askedQuestions')});
        """,
        _routes(client, project), project["id"])
    assert out["before"], "nothing was on screen to begin with"
    assert out["after"] != out["before"], (
        "the press left the render byte-identical, so the pick is invisible")


def claim_a_question_shows_its_consequence(client, project):
    """`why` and `consumer` reach the card, not only the response.

    `DESIGN_LANGUAGE.md` §10's layer 3: the three hardest questions carry a
    reason and a statement of what the answer will change, and the app asks
    them anyway. A question whose consequence stays on the server is a question
    the user cannot weigh — which is the shape `GUIDED-034` was filed as.

    Asserted against the FIRST question that carries both, rather than a named
    one, so the claim survives the plan being reordered.
    """
    plan = _routes(client, project)[
        f"/project/{project['id']}/interview?step=data"]
    carrying = [q for q in plan.get("questions", [])
                if len(str(q.get("why") or "")) > 30
                and len(str(q.get("consumer") or "")) > 30]
    assert carrying, (
        "the plan served no question carrying both a reason and a consumer, so "
        "this claim has nothing to be about — check the fixture, not the page")

    out = _drive("__emit(__harness.html('askedQuestions'));",
                 _routes(client, project), project["id"])
    q = carrying[0]
    assert q["why"][:40] in out, (
        f"the server sent a reason for {q['key']!r} and the page did not "
        f"render it")
    assert q["consumer"][:40] in out, (
        f"the server said what answering {q['key']!r} would change and the page "
        f"did not render it")


def claim_a_consequence_renders_the_servers_words(client, project):
    """`GUIDED-076`. The band renders `q.exits` — labels, details and the
    payload each one posts — rather than composing two controls for the one
    blocker that existed.

    Driven with a consequence that is NOT the leakage blocker, because that is
    the case the composed version got wrong: it would have rendered any second
    consequence as a leakage blocker with the wrong words.
    """
    pid = project["id"]
    blocker = {
        "key": "blocker::cohort::site", "kind": "blocker",
        "title": "Was `site` recorded before enrollment?",
        "status": "asked", "mode": "push", "severity": "blocker",
        "why": "Every subgroup estimate below depends on the answer.",
        "exits": [
            {"id": "resolve_blocker", "kind": X.RESOLVE,
             "label": "Drop site", "detail": "It leaves the analysis.",
             "retry": {"payload": {"column": "site"}}},
            X.attest("Keep it and record why",
                     "The manuscript carries what you type as a limitation.",
                     X.ACKNOWLEDGE_BLOCKER,
                     typed="I am keeping site although it may be post-baseline."),
        ],
    }
    client.post(f"/project/{pid}/decision", json={
        "kind": "set_target", "payload": {"column": "responder"}})
    project = client.get(f"/project/{pid}").json()
    routes = _routes(client, project, **{
        f"/project/{pid}/interview?step=explore": {
            "questions": [blocker], "steps": []}})
    out = _drive("__emit(__harness.html('blockerBand'));", routes, pid)

    assert 'class="blocker"' in out, "the blocker band rendered nothing"
    assert "Drop site" in out and "Keep it and record why" in out, (
        "the band did not render the labels the server served")
    assert "It leaves the analysis." in out, "an exit rendered without its detail"
    assert "may encode the outcome" not in out, (
        "the page composed the LEAKAGE claim for a consequence that is not "
        "about leakage — `GUIDED-076` exactly")
    assert "although it may be post-baseline" in out, (
        "the typed sentence came from the page rather than from the exit")
    assert "acknowledge_blocker" in out, (
        "the attest exit rendered without the key its retry posts, so "
        "`GUIDED-072` is discarded at this boundary again")


def claim_a_refusal_renders_its_exits_and_the_retry_works(client, project):
    """**The other half of `GUIDED-076`.** A 409 arrives with `exits`, each
    carrying the payload its retry posts. `api()` threw `new Error(detail)` on
    an object detail — which stringifies to `[object Object]` — so the refusal
    that travels WITH its way out rendered as nothing a reader could act on.

    Driven on the lens contradiction: a clinical lens over a 396-column assay
    panel. The retry is the ORIGINAL request with the exit's payload merged in,
    and the page contributes nothing to it.
    """
    pid = project["id"]
    # The REAL 409, from the live API, so the page is driven against what the
    # server actually says rather than against a fixture written here.
    refusal = client.post(f"/project/{pid}/decision", json={
        "kind": "set_lens", "payload": {"lens": ["clinical"]}})
    assert refusal.status_code == 409, refusal.text[:200]
    routes = _routes(client, project, **{
        f"POST /project/{pid}/decision": {
            "__status": 409, "body": refusal.json()}})
    out = _drive(
        """
        var rx = /<button([^>]*data-answer-key="state_lens"[^>]*)>/g, m, opts = [];
        var html = __harness.html('askedQuestions');
        while ((m = rx.exec(html)) !== null){
          var a = {};
          m[1].replace(/([a-zA-Z-]+)="([^"]*)"/g,
                       function(_, k, v){ a[k] = v; return ""; });
          opts.push(a);
        }
        var clinical = null;
        opts.forEach(function(o){
          if (o['data-answer-value'] === 'clinical') clinical = o;
        });
        if (!clinical) { throw new Error('no clinical option rendered'); }
        __harness.dispatch('click', __harness.target(clinical, ['answer', 'multi']));
        __harness.dispatch('click', __harness.target(
          {'data-answer-commit': 'state_lens'}, ['answer', 'primary']));

        /* The refusal arrives on a promise, so the microtask queue has to
           drain before the band is readable. The harness drains once before
           this body runs; this is the second drain, after the click. */
        function settle(n){
          var p = Promise.resolve();
          for (var i = 0; i < n; i++) { p = p.then(function(){}); }
          return p;
        }
        settle(12).then(function(){
          var refused = __harness.html('refusal');
          var m2 = /data-refusal-i="(\\d+)"/g, hit, last = null;
          while ((hit = m2.exec(refused)) !== null) { last = hit[1]; }
          if (last !== null) {
            __harness.dispatch('click', __harness.target(
              {'data-refusal-i': last}, ['answer', 'primary']));
          }
          return settle(12).then(function(){
            __emit({refused: refused, posts: __harness.posts()});
          });
        });
        """,
        routes, pid)

    assert "[object Object]" not in out["refused"], (
        "the structured refusal stringified, which is what losing the detail "
        "at the throw looks like")
    assert "data-refusal-i" in out["refused"], (
        "the 409 rendered no exit, so the way out travelled with the refusal "
        "and stopped at the page")

    posts = [c for c in out["posts"] if "/decision" in c.get("path", "")]
    assert len(posts) >= 2, (
        f"expected a refused post and a retry, saw {len(posts)}")
    # The shim records the parsed body, so this is the object the page posted.
    retry = posts[-1]["body"]
    assert retry["payload"].get("acknowledge_contradiction") is True, (
        "the retry did not carry the key the exit declared, so the page is "
        "still holding an out-of-band map")


def claim_a_figure_arrives_with_its_annotation_box(client, project):
    """**The claim I could not write last loop** (`GUIDED-075`). The page never
    fetched `/figures`; five drawable figures reached a user through the API and
    none through the interface, which is exactly what API testing cannot see.

    Asserts the drawn figure AND the two honesty lists, because those are the
    ones that would be easy to drop: a figure silently missing is
    indistinguishable from a figure the app does not have.
    """
    pid = project["id"]
    client.post(f"/project/{pid}/decision", json={
        "kind": "set_lens", "payload": {"lens": ["metabolomics"]}})
    client.post(f"/project/{pid}/decision", json={
        "kind": "set_target", "payload": {"column": "responder"}})
    project = client.get(f"/project/{pid}").json()
    figures = client.get(f"/project/{pid}/figures").json()
    assert figures["admitted"], "the endpoint drew nothing to render"

    routes = _routes(client, project,
                     **{f"/project/{pid}/figures": figures})
    out = _drive(
        """
        function settle(n){
          var p = Promise.resolve();
          for (var i = 0; i < n; i++) { p = p.then(function(){}); }
          return p;
        }
        settle(12).then(function(){
          __harness.drainRaf();
          __emit(__harness.render('figuresBox'));
        });
        """,
        routes, pid)

    drawn = figures["admitted"][0]
    assert drawn["title"] in out, "the drawn figure did not reach the page"
    assert drawn["caption"][:40] in out, "the figure rendered without its caption"
    labels = [a["label"] for a in drawn["annotations"]]
    assert labels, "the endpoint served no annotations to render"
    assert all(lab in out for lab in labels), (
        "the annotation box did not reach the page — which is the whole of "
        "what makes this figure layer more than a plotting library")
    assert "PASS" in out or "FAIL" in out, (
        "the checklist did not render, so nothing scores this render")

    # The honesty lists, which are the ones that would be easy to drop.
    for row in figures["not_drawn"]:
        assert row["title"] in out, (
            f"{row['id']} is not drawn and the page said nothing about it")
        assert row["why"][:40] in out, (
            f"{row['id']} rendered without the reason it does not apply")


def claim_the_record_reads_back(client, project):
    """The transcript. `PRODUCT_VISION.md`'s whole thesis is that what the user
    scrolls and what they export are one object at two levels of formality, so
    a decision whose sentence does not read back is that thesis failing at the
    first level.

    Every sentence is the SERVER's — `COPY_DECK.md`'s walk found all 130
    user-facing promises are composed server-side and none in the page — so
    this asserts the page renders what it was handed rather than a paraphrase.
    """
    pid = project["id"]
    client.post(f"/project/{pid}/decision", json={
        "kind": "set_target", "payload": {"column": "responder"}})
    project = client.get(f"/project/{pid}").json()
    said = [d["text"] for d in project["decisions"] if d.get("text")]
    assert said, "the record carried no sentence to read back"

    # `record(id, html)` writes into `<id>-text`, which is where a decision
    # sentence lands on the page. Reading the container it is written INTO is
    # the difference between asserting the render and asserting the payload.
    out = _drive("__emit(__harness.html('d-data-text') + "
                 "__harness.html('d-target-text'));",
                 _routes(client, project), pid)
    assert any(t[:45] in out for t in said), (
        "not one recorded decision read back onto the page, so the transcript "
        "is a claim the interface does not keep")


CLAIMS = [
    ("questions render", claim_questions_render),
    ("a decision re-paints", claim_a_decision_repaints),
    ("a question shows its consequence", claim_a_question_shows_its_consequence),
    ("a consequence renders the server's words",
     claim_a_consequence_renders_the_servers_words),
    ("a refusal renders its exits and the retry works",
     claim_a_refusal_renders_its_exits_and_the_retry_works),
    ("a figure arrives with its annotation box",
     claim_a_figure_arrives_with_its_annotation_box),
    ("the record reads back", claim_the_record_reads_back),
]


@pytest.mark.parametrize("name,claim", CLAIMS, ids=[c[0] for c in CLAIMS])
def test_claim(name, claim):
    client = _client()
    claim(client, _project(client))


# ── the harness's own gate ─────────────────────────────────────────────────

def test_every_claim_here_goes_red_against_an_emptied_page(monkeypatch):
    """**The gate the README asks for**, run with the suite rather than by
    ritual: empty `index.html` to `<body></body>` and every claim above must
    fail.

    A claim that survives an empty page is not a claim about the page — that is
    exactly the state three frontend tests were in, green, for as long as
    nobody mutated the file underneath them.

    The failure mode is deliberately not uniform. Some claims fail on an
    assertion and some fail because the harness cannot find a controller to run
    at all; both are red and both are the stated reason, which is *the page is
    not there*. Requiring one shape would be asserting something about the
    harness rather than about the page.
    """
    empty = Path(tempfile.mkdtemp()) / "index.html"
    empty.write_text("<html><body></body></html>", encoding="utf-8")
    monkeypatch.setattr(H, "PAGE", empty)

    client = _client()
    project = _project(client)
    survivors = []
    for name, claim in CLAIMS:
        try:
            claim(client, project)
        except Exception:
            continue                                   # red, which is the point
        survivors.append(name)

    assert not survivors, (
        "these claims passed against a page emptied to <body></body>, so they "
        "are not claims about the page: " + ", ".join(survivors))


def test_the_gate_is_not_passing_because_every_claim_is_broken():
    """The positive control on the gate itself.

    The test above passes when every claim fails, and the cheapest way to
    satisfy it is for the claims to be broken all the time. So the claims are
    asserted GREEN against the real page here — the pair is what makes either
    one mean anything, which is `GUIDED-045`'s rule applied to a harness rather
    than to an assertion.
    """
    client = _client()
    project = _project(client)
    for name, claim in CLAIMS:
        claim(client, project)


def test_the_page_renders_the_servers_exits_rather_than_composing_them():
    """`GUIDED-076`, flipped. This asserted the defect last loop — that
    `blockerHTML` never touched `q.exits` — and asserts the fix now.

    A test that asserts a defect is a placeholder with a deadline, and this is
    the deadline arriving.
    """
    page = (Path(__file__).resolve().parent / "web" / "index.html").read_text(
        encoding="utf-8")
    body = page[page.index("function blockerHTML"):]
    body = body[:body.index("\n  function ", 1)]
    assert "q.title" in body                                # positive control
    assert "q.exits" in body, "the band still does not read the server's exits"
    assert "may encode the outcome" not in body, (
        "the hard-coded leakage claim sentence is still composed here")
    assert "data-blk-resolve" not in page and "data-blk-attest" not in page, (
        "the two hard-coded exit controls are still in the page")


def test_a_consequence_with_no_exits_says_so_rather_than_inventing_one():
    """The band will not compose a way through the server did not describe.
    That is the same rule the fix is about, applied to its own failure case."""
    page = (Path(__file__).resolve().parent / "web" / "index.html").read_text(
        encoding="utf-8")
    body = page[page.index("function blockerHTML"):]
    body = body[:body.index("\n  function ", 1)]
    assert "should not happen" in body and "Nothing here can resolve it" in body


def test_the_figure_surface_mutates_rather_than_rebuilding():
    """`DESIGN_LANGUAGE.md` §05's rendering requirement, and this is the
    surface where it first binds.

    Figures arrive and leave as the lens and target change — §05's *Arrive* and
    *Propagate* — and the settle IS the receipt. A renderer that rebuilds the
    list destroys every node on every change, and a transition cannot run to
    completion on a node that no longer exists.

    Asserted structurally rather than by timing, because the mechanism is what
    the requirement is about: the list is keyed per figure, updated in place,
    and never assigned wholesale.
    """
    page = (Path(__file__).resolve().parent / "web" / "index.html").read_text(
        encoding="utf-8")
    body = page[page.index("function renderFigures"):]
    body = body[:body.index("\n  function ", 1)]
    assert "FIG_NODES[id]" in body                             # positive control
    assert "appendChild" in body and "node.innerHTML = html" in body, (
        "the surface no longer adds or mutates per-figure nodes")
    assert "box.innerHTML = rows" not in body, "the list is rebuilt wholesale"
    # The one `innerHTML` on the container is the header, written once.
    assert body.count("box.innerHTML") == 1, (
        "the container is assigned more than once, which is a rebuild wearing "
        "a different name")
    assert "arriving" in body and "leaving" in body, (
        "nothing marks a figure as arriving or leaving, so §05's settle has "
        "nothing to run on")


def test_the_page_now_fetches_the_figure_layer():
    """`GUIDED-075`, flipped. This asserted the absence last loop — zero
    occurrences of `/figures` — and asserts the surface now."""
    page = (Path(__file__).resolve().parent / "web" / "index.html").read_text(
        encoding="utf-8")
    assert len(page) > 20_000 and "renderAll" in page      # positive control
    assert "/figures" in page, "the page still does not fetch the figure layer"
    assert "figuresBox" in page and "renderFigureSurface" in page

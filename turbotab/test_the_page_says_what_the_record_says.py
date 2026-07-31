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

**The page composes its own exits and never reads the server's.**
`blockerHTML` hard-codes the leakage claim sentence, both button labels and the
attestation text, and never touches `q.exits` — so `GUIDED-072`'s `payload_key`
and `retry`, served with every refusal, reach the page and are discarded. That
is the drift `api._disclosures` argues against, in the direction the finding
named. Asserted below as the gap it is, and filed as `GUIDED-076`.

## The fifth claim is absent and that is the other finding

*A figure arrives with its annotation box* is not here, because **the page never
fetches `/figures`** — zero occurrences in `web/index.html`. Five of six
registered figures reach a user through the API and none reaches one through
the Guided page. Writing a test that passed anyway would be this file's own
subject. Filed as `GUIDED-075`.
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


def claim_a_blocker_renders_a_way_through(client, project):
    """A question of consequence reaches the band with a terminal way out.

    `api.py`'s own invariant: a CONSEQUENCE resolves or is attested, never a
    dead end. This asserts the band, the claim sentence and both terminal
    controls — the half the page actually implements. What it does NOT assert
    is that the exits the SERVER declared reached it; see
    `test_the_page_composes_its_own_exits_instead_of_rendering_the_servers`.
    """
    pid = project["id"]
    blocker = {
        "key": "blocker::leakage::glucose", "kind": "blocker",
        "title": "Was glucose recorded after the outcome was known?",
        "status": "asked", "mode": "push", "severity": "blocker",
        "why": "Every accuracy number below this point depends on the answer.",
        "exits": [
            {"id": "revise", "kind": "resolve", "label": "Drop it",
             "detail": "The column leaves the analysis."},
            X.attest("Keep it and record why", "Recorded as a stated limitation.",
                     X.ACKNOWLEDGE_BLOCKER),
        ],
    }
    # The band renders off the EXPLORE plan and only once a target is set —
    # `renderInterview` returns early otherwise. Seeding the data plan showed
    # an empty band, which is the harness being truthful about a path that did
    # not run rather than a page that did not render.
    client.post(f"/project/{pid}/decision", json={
        "kind": "set_target", "payload": {"column": "responder"}})
    project = client.get(f"/project/{pid}").json()
    routes = _routes(client, project, **{
        f"/project/{pid}/interview?step=explore": {
            "questions": [blocker], "steps": []}})
    out = _drive("__emit(__harness.html('blockerBand'));", routes, pid)
    assert 'class="blocker"' in out, "the blocker band rendered nothing"
    assert "glucose" in out, "the band did not name what it is about"
    assert "data-blk-resolve" in out and "data-blk-attest" in out, (
        "a question of consequence rendered without both terminal controls, "
        "which is the dead end `api.py` says cannot exist")


CLAIMS = [
    ("questions render", claim_questions_render),
    ("a decision re-paints", claim_a_decision_repaints),
    ("a question shows its consequence", claim_a_question_shows_its_consequence),
    ("a blocker renders a way through", claim_a_blocker_renders_a_way_through),
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


def test_the_page_composes_its_own_exits_instead_of_rendering_the_servers():
    """`GUIDED-076`, found by writing this harness.

    The server serves `exits` with every refusal, and `GUIDED-072` gave each
    attest exit a `payload_key` and a ready-to-post `retry` so a client could
    act on it. `blockerHTML` reads neither: it writes its own claim sentence,
    its own two labels and its own attestation text, all specific to the
    leakage blocker. A second consequence with different exits would render as
    a leakage blocker with the wrong words.

    Asserted as the current state rather than fixed here, because fixing it is
    a page change and this loop's C part is the harness. When it is fixed this
    test fails, which is the right way round.
    """
    page = (Path(__file__).resolve().parent / "web" / "index.html").read_text(
        encoding="utf-8")
    body = page[page.index("function blockerHTML"):]
    body = body[:body.index("\n  function ", 1)]
    assert "q.title" in body                                # positive control
    assert "exits" not in body, (
        "the page now reads the server's exits — good; delete this test and "
        "close GUIDED-076")
    assert "may encode the outcome" in body, (
        "the hard-coded leakage sentence moved; re-check what this asserts")


def test_the_figure_layer_has_no_page_surface_to_guard():
    """The fifth claim, absent and recorded rather than faked (`GUIDED-075`).

    Five of six registered figures reach a user through `/project/{id}/figures`
    and the Guided page never fetches it. A harness test asserting a figure
    arrives with its annotation box would have to build the surface it claims
    to guard, which is this file's own subject wearing a different hat.
    """
    page = (Path(__file__).resolve().parent / "web" / "index.html").read_text(
        encoding="utf-8")
    assert len(page) > 20_000 and "renderAll" in page      # positive control
    assert "/figures" not in page, (
        "the page now fetches the figure layer — add the annotation-box claim "
        "to CLAIMS above and close GUIDED-075")

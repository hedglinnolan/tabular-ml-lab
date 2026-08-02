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

import numpy as np
import pandas as pd
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


def json_dumps(value):
    return json.dumps(value)


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


def claim_a_refusal_reaches_a_person(client, project):
    """**`GUIDED-080`, and it is the point of the pack.** `LOOP.md` §04 says
    nutrition went first BECAUSE it is the one pack that forces refusals, and
    `/project/{id}/nutrition/prevalence` appeared zero times in the page: four
    refusals built, verified through the API twice, and reachable by nobody.

    A dietary project asks for a prevalence of inadequacy from a single day's
    intake, is refused in the app, reads why, and is shown the shrinkage plot
    the refusal offers — the figure that IS the size of the error the refusal
    prevented. Then it asks something the research specifies and this app
    cannot draw, and gets a record of what is missing rather than a control
    with nothing behind it.
    """
    with open(DATA / "dietary_recalls.csv", "rb") as fh:
        pid = client.post("/project", files={
            "file": ("d.csv", fh, "text/csv")}).json()["id"]
    for what, payload in [("set_lens", {"lens": ["dietary"]}),
                          ("set_target", {"column": "hba1c"}),
                          ("set_grain", {"answer": "people_repeat",
                                         "group_col": "participant_id"}),
                          ("set_repeat_kind", {"kind": "repeats"})]:
        r = client.post(f"/project/{pid}/decision",
                        json={"kind": what, "payload": payload})
        assert r.status_code == 200, (what, r.text[:200])
    project = client.get(f"/project/{pid}").json()

    query = "?nutrient=energy_kcal&basis=single_day&reference_kind=EAR"
    asked = client.get(f"/project/{pid}/nutrition/prevalence{query}").json()
    assert asked["refused"] is True and asked.get("figure"), (
        "the endpoint stopped refusing or stopped offering a drawable figure")

    # The second ask is the PENDING branch: an offer the research specifies and
    # this app cannot draw, because it needs the DRI table (`GUIDED-067`).
    pending_query = "?nutrient=energy_kcal&basis=usual_intake&reference_kind=RDA"
    pending = client.get(
        f"/project/{pid}/nutrition/prevalence{pending_query}").json()
    assert pending["offer"]["pending"] is True, (
        "the pending offer resolved, so this claim no longer covers it")

    routes = _routes(client, project, **{
        f"/project/{pid}/figures": client.get(f"/project/{pid}/figures").json()})
    routes[f"/project/{pid}/nutrition/prevalence{query}"] = asked
    routes[f"/project/{pid}/nutrition/prevalence{pending_query}"] = pending

    out = _drive(
        """
        function settle(n){
          var p = Promise.resolve();
          for (var i = 0; i < n; i++) { p = p.then(function(){}); }
          return p;
        }
        function pick(name, value){
          var sel = __harness.target({'data-prev': name}, []);
          sel.value = value;
          __harness.dispatch('change', sel);
        }
        function ask(){
          __harness.dispatch('click',
            __harness.target({'data-prev-ask': '1'}, ['answer', 'primary']));
        }
        var seen = {};
        settle(12).then(function(){
          seen.controls = __harness.render('prevalenceBox');
          // The person makes each choice through a control and only then asks.
          // Seeding the state directly would test the renderer against itself.
          pick('nutrient', 'energy_kcal');
          pick('basis', 'single_day');
          pick('kind', 'EAR');
          ask();
          return settle(14);
        }).then(function(){
          __harness.drainRaf();
          // THE PARENT, not just the node written to. This used to read
          // `prevOut` alone, with a comment explaining that the shim could not
          // see a node arriving inside assigned markup — which meant the claim
          // could not tell a surface attached to the page from one built beside
          // it. `GUIDED-077` closed that, so it reads where a user would look.
          seen.out = __harness.html('prevOut');
          seen.box = __harness.html('prevalenceBox');
          seen.figures = __harness.render('figuresBox');
          pick('basis', 'usual_intake');
          pick('kind', 'RDA');
          ask();
          return settle(14);
        }).then(function(){
          seen.pending = __harness.render('prevOut');
          __emit(seen);
        });
        """,
        routes, pid)

    assert "data-prev-ask" in out["controls"], (
        "the dietary lens is recorded and there is no way to ask the question")
    assert asked["reason"][:50] in out["out"], (
        "the refusal reached the page without the reason it refused")
    assert asked["reason"][:50] in out["box"], (
        "the refusal is in a node the page never attached, so nobody scrolling "
        "the transcript would find it")
    assert "SETTLED" in out["out"], (
        "the refusal rendered without its badge, so the one thing a reader "
        "cannot check is its epistemic position")
    assert asked["offer"]["label"] in out["out"], (
        "the refusal offered nothing, which is indistinguishable from a "
        "missing feature")
    assert asked["figure"]["title"] in out["figures"], (
        "the offered figure was named and not drawn")

    assert pending["offer"]["resolved"]["needs"][:60] in out["pending"], (
        "a pending offer rendered without saying what it needs, which makes it "
        "indistinguishable from a control that does nothing")
    assert pending["offer"]["resolved"]["blocked_by"] in out["pending"], (
        "the pending offer did not name the row that blocks it")
    assert "<button" not in out["pending"], (
        "the pending offer rendered a control, and there is nothing behind it")


def _sealed(client, rows_per_person: int, n_people: int, answer: str,
            group_col=None):
    """A project driven to a drawn seal, so the basis is whatever the record
    produced rather than whatever the test wanted."""
    rng = np.random.default_rng(0)
    n = n_people * rows_per_person
    frame = pd.DataFrame({"pid": np.repeat(np.arange(n_people), rows_per_person),
                          "x": rng.normal(0, 1, n),
                          "y": rng.normal(0, 1, n)})
    pid = client.post("/project", files={
        "file": ("s.csv", frame.to_csv(index=False).encode(),
                 "text/csv")}).json()["id"]
    steps = [("set_target", {"column": "y"}),
             ("set_grain", {"answer": answer,
                            **({"group_col": group_col} if group_col else {})})]
    if group_col:
        steps += [("set_repeat_kind", {"kind": "repeats"}),
                  ("set_unit_of_analysis", {"unit": "record"})]
    steps += [("set_eligibility", {"answer": "everyone"}), ("seal", {})]
    for what, payload in steps:
        r = client.post(f"/project/{pid}/decision",
                        json={"kind": what, "payload": payload})
        assert r.status_code == 200, (what, r.text[:200])
    return pid, client.get(f"/project/{pid}").json()


def claim_the_seal_says_which_split_it_drew(client, project):
    """**`GUIDED-078`, and it is the closest of the unread surfaces to a
    governing-rule failure.** `disclosures` appeared zero times in the page, so
    the sentence telling a user whether their holdout was drawn by person or by
    row reached nobody — and a holdout drawn by row on repeated measures scores
    better than the model is, by an amount nothing on screen could show.

    ROADMAP's lockbox constitution §03 requires the bases to stay different
    sentences and `undetermined` to be first-class, **never rendered as a clean
    lock**. The server keeps them different strings. This asserts the page
    keeps them different objects: three projects, three bases, three sentences,
    and the two exploratory ones carry neither the sealed class nor the sealed
    word.
    """
    cases = {}
    for name, kwargs in [
            ("grouped", dict(rows_per_person=4, n_people=40,
                             answer="people_repeat", group_col="pid")),
            ("abandoned", dict(rows_per_person=4, n_people=4,
                               answer="people_repeat", group_col="pid")),
            ("undetermined", dict(rows_per_person=4, n_people=40,
                                  answer="not_sure"))]:
        pid, seen = _sealed(client, **kwargs)
        out = _drive(
            """
            var p = Promise.resolve();
            for (var i = 0; i < 14; i++) { p = p.then(function(){}); }
            p.then(function(){
              __harness.drainRaf();
              __emit(__harness.render('disclosuresBox'));
            });
            """,
            _routes(client, seen, **{
                f"/project/{pid}/figures":
                    client.get(f"/project/{pid}/figures").json()}),
            pid)
        cases[name] = {"basis": seen["lockbox"]["seal_basis"],
                       "said": seen["disclosures"], "html": out}

    assert {c["basis"] for c in cases.values()} == {
        "grouped", "repetition_found_grouping_abandoned", "undetermined"}, (
        "the three bases collapsed to fewer than three, so this claim is no "
        "longer about what it says it is")

    for name, case in cases.items():
        assert case["said"]["seal"] in case["html"], (
            f"{name}: the seal sentence the server composed did not reach the "
            "page")
        for key in ("grain", "eligibility"):
            assert case["said"][key] in case["html"], (
                f"{name}: the {key} disclosure was served and not rendered")

    # §03's rule, asserted as the thing it protects rather than as a string.
    assert "is-sealed" in cases["grouped"]["html"]
    assert "sealed" in cases["grouped"]["html"]
    for name in ("undetermined", "abandoned"):
        assert "is-sealed" not in cases[name]["html"], (
            f"{name}: an exploratory seal carries the sealed treatment, which "
            "is §03's forbidden case — a split drawn BY ROW reading as a clean "
            "lock")
        assert "not a verified clean split" in cases[name]["html"], (
            f"{name}: the seal rendered without saying it is not clean")

    assert (cases["undetermined"]["said"]["seal"]
            != cases["abandoned"]["said"]["seal"]), (
        "two of the four bases render the same sentence, so a user cannot tell "
        "'the shape is unknown' from 'the shape is known and too small'")


def claim_an_attested_answer_does_not_render_as_a_clean_split(client, project):
    """The case that proves the page is not deriving §03's rule for itself.

    The user answered *one row per person* against a table whose shape says
    repeated measures, and attested. The recorded basis is `cross_sectional` —
    honest, because it is what they said — so a page computing "clean means
    grouped or cross-sectional" would print a clean lock. The server marks it
    `exploratory` anyway, because the split rests on a disagreement that is on
    the record, and the page renders THAT rather than recomputing it.
    """
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"pid": np.repeat(np.arange(40), 4),
                          "x": rng.normal(0, 1, 160),
                          "y": rng.normal(0, 1, 160)})
    pid = client.post("/project", files={
        "file": ("s.csv", frame.to_csv(index=False).encode(),
                 "text/csv")}).json()["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": "y"}})
    clash = client.post(f"/project/{pid}/decision", json={
        "kind": "set_grain", "payload": {"answer": "one_row_per_person"}})
    assert clash.status_code == 409, (
        "the contradiction did not fire, so there is nothing to attest to")
    for what, payload in [
            ("set_grain", {"answer": "one_row_per_person",
                           "acknowledge_contradiction": True}),
            ("set_eligibility", {"answer": "everyone"}),
            ("seal", {})]:
        r = client.post(f"/project/{pid}/decision",
                        json={"kind": what, "payload": payload})
        assert r.status_code == 200, (what, r.text[:200])
    seen = client.get(f"/project/{pid}").json()

    assert seen["lockbox"]["seal_basis"] == "cross_sectional", (
        "the basis moved, and this claim is specifically about the basis that "
        "looks clean")
    assert seen["disclosures"]["exploratory"] is True

    out = _drive(
        """
        var p = Promise.resolve();
        for (var i = 0; i < 14; i++) { p = p.then(function(){}); }
        p.then(function(){
          __harness.drainRaf();
          __emit(__harness.render('disclosuresBox'));
        });
        """,
        _routes(client, seen, **{
            f"/project/{pid}/figures":
                client.get(f"/project/{pid}/figures").json()}),
        pid)

    assert "is-sealed" not in out, (
        "a split resting on an attested disagreement rendered as a clean lock")
    assert "not a verified clean split" in out
    assert seen["disclosures"]["attested"] in out, (
        "the attestation was served as its own sentence and rendered nowhere, "
        "so what the user confirmed is only on the server")
    assert "belongs in the methods section" in out, (
        "the note the server appends to the seal sentence was dropped")



def claim_the_features_step_reaches_its_end(client, project):
    """**`GUIDED-079` / `DRIVE-010`, and ROADMAP L9's criterion is the gate:**
    a person uploads a file and reaches the END of this step without leaving
    the Guided door.

    `/features` and `/selection/evidence` appeared zero times in the page, so
    25 of the copy deck's 130 promise rows — nineteen percent — described a
    step with no surface at all.

    The walk is driven as clicks and then REPLAYED against the live API, in
    order. That is the honest version of "reaches the end": a page can render
    controls that compose requests the server refuses, and a claim that only
    watched the DOM would call that success. Every request the page composed
    has to be one the record accepts, and the record has to end settled.
    """
    with open(DATA / "clinic_visits.csv", "rb") as fh:
        pid = client.post("/project", files={
            "file": ("s.csv", fh, "text/csv")}).json()["id"]
    r = client.post(f"/project/{pid}/decision",
                    json={"kind": "set_target", "payload": {"column": "hba1c"}})
    assert r.status_code == 200, r.text[:200]
    seen = client.get(f"/project/{pid}").json()

    routes = _routes(client, seen, **{
        f"/project/{pid}/figures": client.get(f"/project/{pid}/figures").json(),
        f"/project/{pid}/features": client.get(f"/project/{pid}/features").json(),
        f"/project/{pid}/selection/evidence":
            client.get(f"/project/{pid}/selection/evidence").json(),
        f"/project/{pid}/interview?step=features":
            client.get(f"/project/{pid}/interview?step=features").json(),
        f"/project/{pid}/feature/preview?transform=log&columns=glucose":
            client.get(f"/project/{pid}/feature/preview"
                       f"?transform=log&columns=glucose").json(),
        f"POST /project/{pid}/decision": seen,
    })

    out = _drive(
        """
        function settle(n){
          var p = Promise.resolve();
          for (var i = 0; i < n; i++) { p = p.then(function(){}); }
          return p;
        }
        function change(attrs, value){
          var el = __harness.target(attrs, []);
          el.value = value;
          __harness.dispatch('change', el);
        }
        function press(attrs){
          __harness.dispatch('click', __harness.target(attrs, ['answer']));
        }
        var seen = {};
        settle(16).then(function(){
          __harness.drainRaf();
          seen.catalogue = __harness.html('featBuild');
          seen.headings = __harness.html('featTitle') + __harness.html('featWhy') +
            __harness.html('featConsumer') + __harness.html('selTitle') +
            __harness.html('selWhy') + __harness.html('selConsumer');
          // Pick a column, look at what the transform would do, then add it.
          change({'data-feat-col': 'log', 'data-feat-slot': '0'}, 'glucose');
          return settle(4);
        }).then(function(){
          seen.ready = __harness.html('featBuild');
          press({'data-feat-preview': 'log'});
          return settle(10);
        }).then(function(){
          seen.preview = __harness.html('featprev-log');
          seen.build = __harness.html('featBuild');
          press({'data-feat-add': 'log', 'data-feat-kind': 'row_local'});
          return settle(10);
        }).then(function(){
          // The selection question, its ranking, and the answer.
          press({'data-sel-rank': '1'});
          return settle(10);
        }).then(function(){
          seen.ranking = __harness.html('selEvidence');
          press({'data-sel-set': '1'});
          return settle(10);
        }).then(function(){
          press({'data-feat-settle': '1'});
          return settle(10);
        }).then(function(){
          seen.posts = __harness.posts();
          seen.err = __harness.el('upErr').textContent;
          __emit(seen);
        });
        """,
        routes, pid)

    assert not out["err"], f"the walk raised: {out['err']}"

    # Both questions are the ROUTER's, asked in its words. The first version of
    # this step wrote its own headings, and one of them asked a different
    # question from the one the record asks.
    plan = routes[f"/project/{pid}/interview?step=features"]
    asked = {q["key"]: q for q in plan["questions"]}
    assert {"choose_features", "choose_selection"} <= set(asked), (
        "the Router stopped serving the Features questions, so this claim is "
        "no longer about the step it names")
    # Only the two this step owns: the plan also carries questions from earlier
    # steps that are still unanswered, and those have their own surfaces.
    for key in ("choose_features", "choose_selection"):
        q = asked[key]
        assert q["title"] in out["headings"], (
            f"the page asks {key!r} in words the Router did not compose")
        assert q["why"][:40] in out["headings"]
        assert q["consumer"][:40] in out["headings"], (
            f"{key} rendered without saying who consumes the answer (§09)")

    # The catalogue is the ENGINE's, not a list written here.
    served = routes[f"/project/{pid}/features"]
    for row in served["row_local"] + served["deferred"]:
        assert row["label"] in out["catalogue"], (
            f"{row['key']} is in the catalogue the server serves and not on "
            "the page, so the page has its own shorter list")
        assert row["because"][:40] in out["catalogue"], (
            f"{row['key']} rendered without the engine's reason, so the page "
            "asserts the row-local/deferred split instead of showing it")

    # A CHOICE gets a preview (§09), and the preview is the REAL computation.
    shown = routes[f"/project/{pid}/feature/preview?transform=log"
                   f"&columns=glucose"]
    assert shown["sentence"] in out["preview"], (
        "the preview rendered without the sentence the engine composed for it")
    # The BEFORE values, which are the table's own and are rendered exactly.
    # Not the after values: the page formats those for reading, and asserting
    # a formatted number would pin the formatter rather than the computation.
    for row in shown["rows"][:3]:
        assert str(row["before"]) in out["preview"], (
            "the preview rendered no values from the table, so it describes "
            "the transform rather than running it")
    assert shown["new_column"] in out["preview"], (
        "the preview did not name the column it would create")
    assert "Nothing in your table has changed" in out["preview"], (
        "a preview that does not say it is a preview is an applied transform")
    assert shown["sentence"] in out["build"], (
        "the preview sits in a node the catalogue never attached, so it is not "
        "beside the transform it is about")
    assert "data-feat-preview" not in out["catalogue"], (
        "a preview was offered before a column was picked, so pressing it "
        "would ask the server to transform nothing")

    ranked = routes[f"/project/{pid}/selection/evidence"]["ranked"]
    assert ranked[0]["feature"] in out["ranking"], (
        "the ranking did not render the column it ranked first")
    assert ranked[0]["measure"] in out["ranking"], (
        "the ranking rendered scores without saying what they measure")
    assert "ranks and does not choose" in out["ranking"], (
        "a ranking a user reads as a decision is a decision nobody made")

    # THE GATE. Every request the page composed, replayed against the record.
    posted = [p["body"] if isinstance(p["body"], dict) else json.loads(p["body"])
              for p in out["posts"]]
    kinds = [b["kind"] for b in posted]
    assert kinds == ["add_feature", "set_selection", "settle_features"], kinds
    for body in posted:
        again = client.post(f"/project/{pid}/decision", json=body)
        assert again.status_code == 200, (body["kind"], again.text[:250])

    ended = client.get(f"/project/{pid}").json()
    assert ended["features_settled"] is True, (
        "the walk ran to the end of the step and the record does not say the "
        "step is over")
    assert any(e["column"] == "log_glucose" for e in ended["engineered"]), (
        "the added feature is not in the working table")




def claim_the_lattice_shows_which_rows_matched(client, project):
    """**`GUIDED-074`, ported from the L31 prototype.**

    `/recipes` appeared zero times in the page. The engine models the whole
    preprocessing decision space, resolves it per model by a precedence rule,
    and measures it for divergence — and a cell rendered as one sentence says
    *this is what happens* where the structure says *these rows matched, this
    one is the most specific, and here is what the others would have done*.

    Driven on the capture script's dietary case, because it is the one that
    suppresses variant questions: three of them, derived and compared and found
    not to change the answer. `n_choices_suppressed` is a first-class statement
    here, not a number in a payload — a question that silently disappears is
    indistinguishable from one nobody thought of.
    """
    with open(DATA / "dietary_recalls.csv", "rb") as fh:
        pid = client.post("/project", files={
            "file": ("d.csv", fh, "text/csv")}).json()["id"]

    def decide(what, **payload):
        r = client.post(f"/project/{pid}/decision",
                        json={"kind": what, "payload": payload})
        assert r.status_code == 200, (what, r.text[:200])

    decide("set_lens", lens=["dietary"])
    decide("set_target", column="hba1c")
    decide("set_grain", answer="people_repeat", group_col="participant_id")
    decide("set_repeat_kind", kind="repeats")
    decide("set_unit_of_analysis", unit="person")
    decide("set_aggregation", method="mean")
    decide("set_eligibility", answer="everyone")
    decide("seal")
    shelf = client.get(f"/project/{pid}/models").json()
    available = {m["key"] for g in shelf.get("groups", []) for m in g["models"]}
    picks = [k for k in ("ridge", "rf", "knn_reg", "histgb_reg", "nn")
             if k in available]
    decide("select_models", models=picks)
    decide("set_preparation_mode", mode="per_model")

    lattice = client.get(f"/project/{pid}/recipes").json()
    assert lattice["n_choices_suppressed"] == 3, (
        "the dietary case stopped suppressing three variant questions, so this "
        "claim no longer covers the statement it is about")
    assert lattice["candidates"], (
        "the endpoint serves no candidates, so there is no reasoning to render")
    seen = client.get(f"/project/{pid}").json()

    routes = _routes(client, seen, **{
        f"/project/{pid}/figures": client.get(f"/project/{pid}/figures").json(),
        f"/project/{pid}/features": client.get(f"/project/{pid}/features").json(),
        f"/project/{pid}/interview?step=features":
            client.get(f"/project/{pid}/interview?step=features").json(),
        f"/project/{pid}/recipes": lattice,
    })

    out = _drive(
        """
        function settle(n){
          var p = Promise.resolve();
          for (var i = 0; i < n; i++) { p = p.then(function(){}); }
          return p;
        }
        var seen = {};
        settle(18).then(function(){
          __harness.drainRaf();
          seen.grid = __harness.html('latGrid');
          seen.box = __harness.html('latticeBox');
          seen.supp = __harness.html('latSupp');
          // Open the cell whose stack has more than one row in it.
          var keys = [], rx = /data-lat-cell="([^"]+)"/g, m;
          while ((m = rx.exec(seen.grid)) !== null) { keys.push(m[1]); }
          seen.keys = keys;
          __harness.dispatch('click',
            __harness.target({'data-lat-cell': keys[0]}, []));
          return settle(6);
        }).then(function(){
          seen.why = __harness.html('latWhy');
          __emit(seen);
        });
        """,
        routes, pid)

    # THE ORPHAN CHECK, and it is why this claim was rewritten. Before
    # `GUIDED-077`, `latticeBox` was EMPTY — 0 characters — while `latGrid`
    # carried 179, because `if (!$("latGrid"))` could never be true against a
    # `getElementById` that auto-created, so the header was never written and
    # the grid was never attached. The claim read the orphan and passed. A
    # renderer's output is not a page.
    assert len(out["box"]) > len(out["grid"]), (
        "the grid is not inside the lattice container, so it was built and "
        "never attached — which is what this claim used to be measuring")
    assert 'id="latGrid"' in out["box"]

    models = sorted(lattice["models"])
    ops = [o["key"] for o in lattice["operations"]]
    assert len(out["keys"]) == len(models) * len(ops), (
        f"{len(out['keys'])} cells rendered for {len(models)} models and "
        f"{len(ops)} operations — the grid is not the lattice")
    for model in models:
        assert model in out["grid"], f"{model} has no row in the grid"
    for op in ops:
        assert op in out["grid"], f"{op} has no column in the grid"

    # Every cell says what the table resolved AND which selector won it. The
    # selector is the part a sentence drops, and it is the whole lattice.
    for model, rows in lattice["models"].items():
        for row in rows:
            assert row["selector"] in out["grid"], (
                f"{model}/{row['operation']} rendered without the selector "
                "that decided it")

    # THE STACK. Every matched row, ranked, with the winner marked — and the
    # ranking is the server's.
    stack = lattice["candidates"][out["keys"][0]]
    assert stack, "the first cell matched nothing, which resolve() cannot do"
    for row in stack:
        assert row["selector"] in out["why"], (
            f"{row['selector']} matched this cell and the stack did not show it")
        assert row["reason"][:40] in out["why"], (
            "a matched row rendered without the reason it would have given")
    assert sum(1 for r in stack if r["wins"]) == 1
    winner = [r for r in stack if r["wins"]][0]
    assert "wins" in out["why"], (
        "the stack shows every row that matched and does not mark which one "
        "the engine took, which is the one thing the stack is for")
    assert winner["variant"] in out["why"]

    # `n_choices_suppressed` as a statement rather than a count in a payload.
    assert "3 questions were derived, compared and not asked" in out["supp"], (
        "three variant questions were suppressed and the page does not say so")




def claim_preprocess_reaches_its_end(client, project):
    """**`GUIDED-085`, and ROADMAP L9's criterion unchanged:** a person uploads
    a file and reaches the END of Preprocess without leaving the Guided door.

    `/preprocess` was composed, complete and fetched by nothing. Every strategy
    already carried `because` — clause §06's litmus in words — and `defers`, so
    an interface could say WHY a choice changes nothing on screen instead of
    leaving the user to conclude the app did nothing.

    §07's fork is the shape of the step and is asserted as such: the mechanism
    is asked first, and the strategies are not on screen until it is answered,
    because which fills are legitimate depends on the answer.

    Driven as clicks and REPLAYED against the record, for the same reason the
    Features claim is: a page can render controls that compose requests the
    server refuses, and a claim that only watched the DOM would call that
    success.
    """
    with open(DATA / "clinic_visits.csv", "rb") as fh:
        pid = client.post("/project", files={
            "file": ("s.csv", fh, "text/csv")}).json()["id"]
    for what, payload in [("set_target", {"column": "hba1c"}),
                          ("set_purpose", {"answer": "prediction"})]:
        r = client.post(f"/project/{pid}/decision",
                        json={"kind": what, "payload": payload})
        assert r.status_code == 200, (what, r.text[:200])
    seen = client.get(f"/project/{pid}").json()
    served = client.get(f"/project/{pid}/preprocess").json()
    assert served["columns"], (
        "this fixture has no column with blanks, so the claim has nothing to "
        "be about — check the fixture, not the page")
    first = served["columns"][0]["column"]

    routes = _routes(client, seen, **{
        f"/project/{pid}/figures": client.get(f"/project/{pid}/figures").json(),
        f"/project/{pid}/features": client.get(f"/project/{pid}/features").json(),
        f"/project/{pid}/interview?step=features":
            client.get(f"/project/{pid}/interview?step=features").json(),
        f"/project/{pid}/recipes": client.get(f"/project/{pid}/recipes").json(),
        f"/project/{pid}/preprocess": served,
        f"POST /project/{pid}/decision": seen,
    })

    out = _drive(
        """
        function settle(n){
          var p = Promise.resolve();
          for (var i = 0; i < n; i++) { p = p.then(function(){}); }
          return p;
        }
        var seen = {};
        settle(20).then(function(){
          __harness.drainRaf();
          seen.before = __harness.html('prepCols');
          seen.why = __harness.html('prepWhy') + __harness.html('prepConsumer');
          __harness.dispatch('click', __harness.target(
            {'data-prep-mech': COLUMN, 'data-prep-mech-value': 'not_informative'},
            ['pill']));
          return settle(6);
        }).then(function(){
          seen.after = __harness.html('prepCols');
          seen.receipt = __harness.html('prepReceipt');
          var m = /data-prep-strategy="([^"]+)"/.exec(seen.after);
          seen.strategy = m && m[1];
          __harness.dispatch('click', __harness.target(
            {'data-prep-route': COLUMN, 'data-prep-strategy': seen.strategy},
            ['answer', 'primary']));
          return settle(10);
        }).then(function(){
          __harness.dispatch('click',
            __harness.target({'data-prep-settle': '1'}, ['answer']));
          return settle(10);
        }).then(function(){
          seen.posts = __harness.posts();
          seen.err = __harness.el('upErr').textContent;
          __emit(seen);
        });
        """.replace("COLUMN", repr(first).replace("'", '"')),
        routes, pid)

    assert not out["err"], f"the walk raised: {out['err']}"

    # The mechanism question is the server's, with its consumer (§09).
    question = served["mechanism_question"]
    assert question["why"][:50] in out["why"]
    assert question["consumer"][:50] in out["why"], (
        "the step asks what a blank means and does not say who consumes the "
        "answer")

    # §07's ORDER, as the property rather than as a comment: no strategy is on
    # screen before the mechanism is answered, and they arrive when it is.
    assert "data-prep-strategy" not in out["before"], (
        "the fills were offered beside the question that decides which of them "
        "are legitimate, which is how a column that carried information gets a "
        "median written over it")
    assert "data-prep-strategy" in out["after"], (
        "answering the mechanism produced no strategies, so the question leads "
        "nowhere")

    # Every strategy carries the engine's litmus answer and its timing.
    branch = served["columns"][0]["branch"]
    for st in served["strategies"][branch]:
        if st["key"] not in served["columns"][0]["strategies"]:
            continue
        assert st["label"] in out["after"]
        assert st["because"][:40] in out["after"], (
            f"{st['key']} rendered without the reason it defers or does not")

    assert served["receipt"]["headline"] in out["receipt"], (
        "the receipt the server composed did not reach the page")

    # THE GATE. Every request the page composed, replayed against the record.
    posted = [p["body"] if isinstance(p["body"], dict) else json.loads(p["body"])
              for p in out["posts"]]
    assert [b["kind"] for b in posted] == ["route_missingness",
                                           "settle_preprocess"], posted
    for body in posted:
        again = client.post(f"/project/{pid}/decision", json=body)
        assert again.status_code == 200, (body["kind"], again.text[:250])

    ended = client.get(f"/project/{pid}").json()
    assert ended["preprocess_settled"] is True, (
        "the walk ran to the end of the step and the record does not say the "
        "step is over")
    routed = [d for d in ended["decisions"] if d["kind"] == "route_missingness"]
    assert routed, "nothing was routed"

    # ONE SENTENCE, not three. The transcript line, the step's own receipt and
    # the methods prose are the same string composed once by the server — which
    # is the whole reason the timing rides in the sentence rather than beside
    # it. Asserted as identity, because two strings that merely agree today are
    # two strings.
    after = client.get(f"/project/{pid}/preprocess").json()
    declared = [d for d in after["declared"] if d["column"] == first]
    assert declared, f"{first} was routed and the step does not list it"
    assert declared[0]["sentence"] == routed[0]["text"], (
        "the step's receipt and the transcript carry different sentences for "
        "one decision, so one of them is a paraphrase")
    assert declared[0]["fit_on"], (
        "the declaration does not say what the strategy is fitted on, which is "
        "the half of §06 that decides whether it can leak")




def claim_imputing_an_informative_blank_is_a_blocker_with_a_way_through(
        client, project):
    """**§07's fork, where it bites.** A blank that means something and a fill
    that erases it is not a warning — it is a blocker with a typed
    acknowledgment, because the information is gone from the data afterward and
    no model can recover it.

    The exits are the SERVER's (`GUIDED-076`), so this asserts the page renders
    the interruption it was handed and that the resolve exit's retry is a
    request the record accepts — a blocker whose way through does not work is a
    dead end with extra words.
    """
    with open(DATA / "clinic_visits.csv", "rb") as fh:
        pid = client.post("/project", files={
            "file": ("s.csv", fh, "text/csv")}).json()["id"]
    for what, payload in [("set_target", {"column": "hba1c"}),
                          ("set_purpose", {"answer": "prediction"})]:
        client.post(f"/project/{pid}/decision",
                    json={"kind": what, "payload": payload})
    seen = client.get(f"/project/{pid}").json()
    served = client.get(f"/project/{pid}/preprocess").json()

    fills = [c for c in served["columns"]
             if any(k.startswith("impute") for k in c["strategies"])]
    assert fills, "no column offers a fill, so there is no blocker to reach"
    col = fills[0]
    strategy = [k for k in col["strategies"] if k.startswith("impute")][0]

    refused = client.post(f"/project/{pid}/decision", json={
        "kind": "route_missingness",
        "payload": {"column": col["column"], "mechanism": "informative",
                    "strategy": strategy}})
    assert refused.status_code == 409, (
        "filling an informative blank was not blocked", refused.status_code)
    blocker = refused.json()["detail"]

    routes = _routes(client, seen, **{
        f"/project/{pid}/figures": client.get(f"/project/{pid}/figures").json(),
        f"/project/{pid}/features": client.get(f"/project/{pid}/features").json(),
        f"/project/{pid}/interview?step=features":
            client.get(f"/project/{pid}/interview?step=features").json(),
        f"/project/{pid}/recipes": client.get(f"/project/{pid}/recipes").json(),
        f"/project/{pid}/preprocess": served,
        # Shaped as FastAPI shapes it — the refusal rides in `detail`, which is
        # the field `api()` reads and the reason `GUIDED-076` was about the
        # throw rather than the response.
        f"POST /project/{pid}/decision": {"__status": 409,
                                          "body": {"detail": blocker}},
    })

    out = _drive(
        """
        function settle(n){
          var p = Promise.resolve();
          for (var i = 0; i < n; i++) { p = p.then(function(){}); }
          return p;
        }
        settle(20).then(function(){
          __harness.drainRaf();
          __harness.dispatch('click', __harness.target(
            {'data-prep-mech': COLUMN, 'data-prep-mech-value': 'informative'},
            ['pill']));
          return settle(6);
        }).then(function(){
          __harness.dispatch('click', __harness.target(
            {'data-prep-route': COLUMN, 'data-prep-strategy': STRATEGY},
            ['answer', 'primary']));
          return settle(12);
        }).then(function(){
          __emit({band: __harness.html('refusal'),
                  posts: __harness.posts()});
        });
        """.replace("COLUMN", json_dumps(col["column"]))
           .replace("STRATEGY", json_dumps(strategy)),
        routes, pid)

    assert blocker["message"][:60] in out["band"], (
        "the blocker reached the page without the sentence that explains it")
    for exit_ in blocker["exits"]:
        assert exit_["label"] in out["band"], (
            f"the exit {exit_['id']!r} was served and rendered nowhere, so the "
            "interruption is a dead end")
        assert exit_["detail"][:40] in out["band"]

    # The resolve exit is a REQUEST, and it has to be one the record takes.
    # `GUIDED-072`'s unifying test: a client holding only the payload can act.
    # It used to fail here — the resolve exit carried a label and a detail and
    # no `retry`, so the page rendered it DISABLED and the only live way out of
    # the blocker was the attestation. A way through that cannot be taken is a
    # dead end with extra words.
    resolve = [e for e in blocker["exits"] if e["kind"] == "resolve"]
    assert resolve, "the blocker offers no way through that is not an attestation"
    assert resolve[0].get("retry", {}).get("payload"), (
        "the resolve exit says what to do and carries nothing to do it with")
    assert "disabled" not in out["band"], (
        "an exit rendered disabled, so the page describes a way through it "
        "cannot take")
    posted = [p["body"] if isinstance(p["body"], dict) else json.loads(p["body"])
              for p in out["posts"]]
    assert posted and posted[-1]["payload"]["mechanism"] == "informative", (
        "the page posted a mechanism the user did not choose")

    followed = dict(posted[-1])
    followed["payload"] = dict(followed["payload"],
                               **resolve[0]["retry"]["payload"])
    accepted = client.post(f"/project/{pid}/decision", json=followed)
    assert accepted.status_code == 200, (
        "the way through the blocker is refused by the record",
        accepted.text[:250])
    kept = [d for d in accepted.json()["missingness"]
            if d["column"] == col["column"]]
    assert kept and kept[0]["strategy"] != strategy, (
        "taking the exit left the erasing strategy in place")




def claim_an_upload_reaches_a_held_out_number(client, project):
    """**The first number this door COMPUTES rather than reads**, and the two
    rules that are not optional around it.

    `PRODUCT_VISION.md` §04: anything over about a second is an observable job
    with a name in plain language, progress and a cancel — never a bare
    spinner. Training is the reason that rule exists, and `turbotab/jobs.py`
    was built at L7 with a consumer in Classic and none here.

    And the shelf is never shortened: three groups are always rendered,
    including empty ones, because *nothing is recommended for this data* is a
    real state and a renderer that drops the group asserts it was never
    considered. Ranking carries the judgment; absence never does.
    """
    with open(DATA / "leaky_sepsis.csv", "rb") as fh:
        pid = client.post("/project", files={
            "file": ("s.csv", fh, "text/csv")}).json()["id"]
    for what, payload in [("set_target", {"column": "sepsis"}),
                          ("set_grain", {"answer": "one_row_per_person"}),
                          ("set_eligibility", {"answer": "everyone"}),
                          ("seal", {})]:
        r = client.post(f"/project/{pid}/decision",
                        json={"kind": what, "payload": payload})
        assert r.status_code == 200, (what, r.text[:200])
    shelf = client.get(f"/project/{pid}/models").json()
    keys = [m["key"] for g in shelf["groups"] for m in g["models"]][:2]
    assert keys, "the shelf offered nothing, so there is nothing to fit"
    seen = client.get(f"/project/{pid}").json()

    routes = _routes(client, seen, **{
        f"/project/{pid}/figures": client.get(f"/project/{pid}/figures").json(),
        f"/project/{pid}/features": client.get(f"/project/{pid}/features").json(),
        f"/project/{pid}/interview?step=features":
            client.get(f"/project/{pid}/interview?step=features").json(),
        f"/project/{pid}/recipes": client.get(f"/project/{pid}/recipes").json(),
        f"/project/{pid}/preprocess":
            client.get(f"/project/{pid}/preprocess").json(),
        f"/project/{pid}/models": shelf,
        f"/project/{pid}/training": {"run": None, "blocked_by": None},
        # The job as it is MID-RUN, so the claim sees what a user sees while
        # waiting rather than only the finished state.
        f"POST /project/{pid}/train": {
            "id": "job1", "name": "Training 2 model(s) on the held-out split",
            "status": "running", "progress": 0.5, "message": "Fitting",
            "error": None, "seed": 42, "elapsed": 0.2, "terminal": False},
        # The poll's next answer. Terminal, so the page stops asking — a
        # non-terminal reply here would leave the controller polling forever,
        # which is worth knowing is the behavior: it stops when the job stops.
        "/job/job1": {"id": "job1", "name": "Training", "status": "done",
                      "progress": 1.0, "message": "2 model(s) scored",
                      "error": None, "seed": 42, "elapsed": 1.0,
                      "terminal": True, "result": None},
    })

    out = _drive(
        """
        function settle(n){
          var p = Promise.resolve();
          for (var i = 0; i < n; i++) { p = p.then(function(){}); }
          return p;
        }
        var seen = {};
        settle(22).then(function(){
          __harness.drainRaf();
          seen.shelf = __harness.html('shelfBox');
          __harness.dispatch('click',
            __harness.target({'data-pick-model': PICK}, ['pill']));
          return settle(6);
        }).then(function(){
          seen.picked = __harness.html('shelfBox');
          __harness.dispatch('click',
            __harness.target({'data-train-start': '1'}, ['answer', 'primary']));
          return settle(12);
        }).then(function(){
          // Read BEFORE the poll's first tick, which is what a user sees for
          // the whole of a real fit.
          seen.running = __harness.html('trainRun');
          seen.calls = __harness.calls().map(function(x){ return x.path; });
          seen.posts = __harness.posts();
          seen.err = __harness.el('upErr').textContent;
          __emit(seen);
        });
        """.replace("PICK", json.dumps(keys[0])),
        routes, pid)

    assert not out["err"], f"the walk raised: {out['err']}"

    # THE SHELF, WHOLE. Every group the server returned, including empty ones.
    for group in shelf["groups"]:
        assert group["label"] in out["shelf"], (
            f"the group {group['label']!r} was returned and rendered nowhere, "
            "so a model this coach ranked is invisible")
        for model in group["models"]:
            assert model["name"] in out["shelf"], (
                f"{model['key']} is on the shelf and not on the page")
    empty = [g for g in shelf["groups"] if not g["models"]]
    for group in empty:
        assert "considered, not skipped" in out["shelf"], (
            "an empty group rendered as nothing, which reads as never "
            "considered rather than as considered and empty")

    assert 'aria-pressed="true"' in out["picked"], (
        "picking a model changed nothing on screen")

    # §04: a name, progress, and a control that stops it.
    assert "Training" in out["running"], "the job has no name a person can read"
    assert "data-train-cancel" in out["running"], (
        "the job runs with no way to stop it, which is the Classic cancel "
        "button that sets a flag nothing reads")
    assert 'style="width:50%"' in out["running"], (
        "the job reports no progress, so a slow fit and a hung one look alike")

    posted = [p["body"] if isinstance(p["body"], dict) else json.loads(p["body"])
              for p in out["posts"]]
    assert posted and posted[-1]["models"] == [keys[0]], (
        "the page submitted a different set of models than the user picked")

    # THE GATE, through the record: the request the page composed produces a
    # held-out number, and the run says what it was scored on.
    started = client.post(f"/project/{pid}/train", json=posted[-1])
    assert started.status_code == 200, started.text[:250]
    job = started.json()
    for _ in range(200):
        job = client.get(f"/job/{job['id']}").json()
        if job["terminal"]:
            break
    assert job["status"] == "done", (job["status"], job.get("error"))
    run = client.get(f"/project/{pid}/training").json()["run"]
    assert run and run["n_test"] > 0
    scored = [r for r in run["results"] if r["metrics"]]
    assert scored, "no model produced a metric"
    # A model that would not fit keeps its row and says why — the shelf is not
    # shortened by a failure any more than by a ranking.
    #
    # **MUTUALLY EXCLUSIVE, not `or`** (`GUIDED-093`). `metrics or error` is
    # satisfied when BOTH are set, and that is exactly the state the app spent
    # two loops in: Accuracy 0.857 beside "did not fit", from one line that
    # coerced a string class label to `float` after the metrics were assigned.
    # If serialization failed the metric is not trustworthy either.
    for result in run["results"]:
        assert bool(result["metrics"]) != bool(result["error"]), (
            f"{result['key']} carries a score and a reason at once "
            f"({result['metrics']} / {result['error']}), so two readers of one "
            f"result get two answers to one question")


def claim_the_calibration_figure_is_drawn_for_the_first_time(client, project):
    """`GUIDED-065`. The calibration plot has had a renderer, a checklist, an
    annotation box and no data path since L26 — `has_predictions` was `False`
    for every project that could exist, and the figure said so in a sentence
    about the APP rather than about the table.

    There is a training step now. The curve is drawn from the held-out
    predictions and nowhere else: a calibration curve on training predictions
    is a model grading its own homework, and it looks better the more the model
    overfits.
    """
    with open(DATA / "leaky_sepsis.csv", "rb") as fh:
        pid = client.post("/project", files={
            "file": ("s.csv", fh, "text/csv")}).json()["id"]
    for what, payload in [("set_target", {"column": "sepsis"}),
                          ("set_grain", {"answer": "one_row_per_person"}),
                          ("set_eligibility", {"answer": "everyone"}),
                          ("seal", {})]:
        client.post(f"/project/{pid}/decision",
                    json={"kind": what, "payload": payload})

    before = client.get(f"/project/{pid}/figures").json()
    absent = [f for f in before["not_drawn"] + before.get("unavailable", [])
              if f["id"] == "calibration"]
    assert absent, "the calibration figure is not accounted for at all"
    assert "No model has been fitted yet" in absent[0]["why"], (
        "the reason is not about this project's state, so it cannot stop "
        "being true")

    shelf = client.get(f"/project/{pid}/models").json()
    keys = [m["key"] for g in shelf["groups"] for m in g["models"]][:3]
    job = client.post(f"/project/{pid}/train", json={"models": keys}).json()
    for _ in range(200):
        job = client.get(f"/job/{job['id']}").json()
        if job["terminal"]:
            break
    assert job["status"] == "done", (job["status"], job.get("error"))

    after = client.get(f"/project/{pid}/figures").json()
    drawn = [f for f in after["admitted"] + after["held"]
             if f["id"] == "calibration"]
    assert drawn, (
        "the models are fitted and the calibration plot is still not drawn")
    figure = drawn[0]
    assert figure["payload"]["scored_on"] == "held-out rows only", (
        "the curve does not say which rows it was computed on, which is the "
        "one thing that decides whether it means anything")

    # The annotation box renders the ABSENCE of a number rather than a blank,
    # which is the behavior `calibration_render` was built with and which no
    # project could reach until now.
    labels = {a["label"]: a["value"] for a in figure["annotations"]}
    assert "C-statistic" in labels
    for value in labels.values():
        assert value not in ("", None), (
            "an annotation rendered blank; a missing number is stated, not "
            "left empty")

    seen = client.get(f"/project/{pid}").json()
    out = _drive(
        """
        var p = Promise.resolve();
        for (var i = 0; i < 20; i++) { p = p.then(function(){}); }
        p.then(function(){
          __harness.drainRaf();
          __emit(__harness.html('figuresBox'));
        });
        """,
        _routes(client, seen, **{
            f"/project/{pid}/figures": after,
            f"/project/{pid}/features":
                client.get(f"/project/{pid}/features").json(),
            f"/project/{pid}/models": shelf,
            f"/project/{pid}/training":
                client.get(f"/project/{pid}/training").json(),
        }),
        pid)
    assert figure["title"] in out, (
        "the figure is drawn by the server and does not reach the page")



def claim_the_run_says_what_it_actually_fitted(client, project):
    """**`GUIDED-095`, driven.** The trainer used to build its own pipeline and
    read no declaration at all, so the analysis a user specified and the
    analysis that was fitted were two different things and only one of them was
    on screen.

    Driven rather than verified through the API, because the whole class this
    closes has the other shape — a server composing a user-facing string that
    the interface never renders (`GUIDED-080`). A per-model plan that reaches
    only the payload is that defect wearing this loop's fix.

    The case is the hard one on purpose: `indicator` on a column, fitted by a
    model that reads a blank natively AND by one that cannot. The second must
    show the recorded sentence beside the one that is now true of its fit.
    """
    with open(DATA / "metabolomics_untargeted.csv", "rb") as fh:
        pid = client.post("/project", files={
            "file": ("s.csv", fh, "text/csv")}).json()["id"]
    for what, payload in [("set_target", {"column": "responder"}),
                          ("set_purpose", {"answer": "prediction"}),
                          ("set_grain", {"answer": "one_row_per_person"}),
                          ("set_eligibility", {"answer": "everyone"}),
                          ("seal", {"fraction": 0.25}),
                          ("route_missingness",
                           {"column": "bmi", "mechanism": "informative",
                            "strategy": "indicator"})]:
        r = client.post(f"/project/{pid}/decision",
                        json={"kind": what, "payload": payload})
        assert r.status_code == 200, (what, r.text[:200])

    shelf = client.get(f"/project/{pid}/models").json()
    keys = [m["key"] for g in shelf["groups"] for m in g["models"]]
    picks = [k for k in ("histgb_clf", "logreg") if k in keys]
    assert len(picks) == 2, keys
    job = client.post(f"/project/{pid}/train", json={"models": picks}).json()
    for _ in range(400):
        job = client.get(f"/job/{job['id']}").json()
        if job["terminal"]:
            break
    assert job["status"] == "done", (job["status"], job.get("error"))
    served = client.get(f"/project/{pid}/training").json()

    diverged = [d for r in served["run"]["results"]
                for d in r["plan"]["divergences"]]
    assert diverged, (
        "no model diverged, so this claim would pass against a page that "
        "renders divergences nowhere")
    seen = client.get(f"/project/{pid}").json()
    out = _drive(
        """
        var p = Promise.resolve();
        for (var i = 0; i < 26; i++) { p = p.then(function(){}); }
        p.then(function(){
          __harness.drainRaf();
          __emit(__harness.html('trainRun'));
        });
        """,
        _routes(client, seen, **{
            f"/project/{pid}/models": shelf,
            f"/project/{pid}/training": served,
            f"/project/{pid}/figures":
                client.get(f"/project/{pid}/figures").json(),
            f"/project/{pid}/features":
                client.get(f"/project/{pid}/features").json(),
            f"/project/{pid}/preprocess":
                client.get(f"/project/{pid}/preprocess").json(),
            f"/project/{pid}/recipes":
                client.get(f"/project/{pid}/recipes").json(),
        }),
        pid)

    assert "composed from the recorded plan" in out, (
        "the run does not tell a reader where its pipeline came from")
    assert "not the recorded preprocessing plan" not in out, (
        "the page still carries the mitigation for a defect that is closed")

    # THE DIVERGENCE REACHES A PERSON, with both sentences.
    d = diverged[0]
    assert d["recorded_sentence"][:50] in out, (
        "the page shows what was fitted and not what was recorded, so a "
        "reader cannot see that the two differ")
    assert d["fitted_sentence"][:60] in out, (
        "the divergence was composed by the server and rendered nowhere")

    # AND THE STEPS DO. The honored model's own sentence for the same column
    # is the record's, and it is on screen.
    honored = [r for r in served["run"]["results"]
               if not r["plan"]["divergences"]]
    assert honored, "no model honored the declaration"
    kept = [s for s in honored[0]["plan"]["sentences"] if "bmi" in s]
    assert kept and kept[0][:60] in out, (
        "the fitted plan is served per model and the page prints none of it")


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
    ("a refusal reaches a person", claim_a_refusal_reaches_a_person),
    ("the seal says which split it drew",
     claim_the_seal_says_which_split_it_drew),
    ("an attested answer does not render as a clean split",
     claim_an_attested_answer_does_not_render_as_a_clean_split),
    ("the features step reaches its end", claim_the_features_step_reaches_its_end),
    ("the lattice shows which rows matched",
     claim_the_lattice_shows_which_rows_matched),
    ("preprocess reaches its end", claim_preprocess_reaches_its_end),
    ("an upload reaches a held-out number",
     claim_an_upload_reaches_a_held_out_number),
    ("the calibration figure is drawn for the first time",
     claim_the_calibration_figure_is_drawn_for_the_first_time),
    ("imputing an informative blank is a blocker with a way through",
     claim_imputing_an_informative_blank_is_a_blocker_with_a_way_through),
    ("the run says what it actually fitted",
     claim_the_run_says_what_it_actually_fitted),
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


def test_the_shim_reports_a_class_it_was_assigned_rather_than_swallowing_it():
    """`GUIDED-081`, and it is the reason the seal claims mean anything.

    `El` had no `className` property, so `node.className = "x"` set an ordinary
    JS field nothing else read: `classList.contains` said no, `__deep` printed
    no class attribute, and **any assertion about how a node is styled came
    back vacuously true**. Every mutate-in-place renderer §05 requires sets its
    class exactly that way, so the hole grew with each one.

    A shim is allowed to be ignorant of pixels. It is not allowed to accept a
    write and then deny it happened — that reports the page as honest about
    styling precisely where the page has stopped being honest.
    """
    out = H.run(
        """
        var n = document.createElement('div');
        n.id = 'probe';
        n.classList.add('arriving');
        n.className = 'disc-row is-exploratory';
        document.getElementById('askedQuestions').appendChild(n);
        __emit({read: n.className,
                contains: n.classList.contains('is-exploratory'),
                stale: n.classList.contains('arriving'),
                deep: __harness.render('askedQuestions')});
        """,
        routes={}, search="")
    assert out["read"] == "disc-row is-exploratory", (
        "the shim took a className write and read back something else")
    assert out["contains"] is True, (
        "`className` and `classList` are two views of one set, and they "
        "disagree")
    assert out["stale"] is False, (
        "assigning className kept an earlier class, so a node can carry a "
        "state it was told to drop")
    assert 'class="disc-row is-exploratory"' in out["deep"], (
        "the class does not survive serialization, so a claim about styling "
        "cannot be written at all")


def test_the_lattice_mutates_cells_rather_than_rebuilding_the_grid():
    """`GUIDED-074`, and the prototype's finding is the requirement.

    The L31 prototype measured what a repaint costs and found something worse
    than the cost: **a repaint cannot report what it interrupted**, because
    `transitioncancel` does not fire when the element is removed — the
    transition ends with its target and there is nothing left to dispatch on.
    `DESIGN_LANGUAGE.md` §05 is that finding written as a rule, and this is the
    surface the finding came from, so it is the surface where breaking it would
    be least excusable.

    Asserted structurally, for the same reason the figure surface's version is:
    the mechanism is what the requirement is about.
    """
    page = (Path(__file__).resolve().parent / "web" / "index.html").read_text(
        encoding="utf-8")
    body = page[page.index("function renderLattice"):]
    body = body[:body.index("\n  function ", 1)]
    assert "LAT_NODES[k]" in body                              # positive control
    assert "grid.appendChild(cell)" in body, (
        "the grid no longer builds per-cell nodes, so there is nothing to keep")
    assert "node.innerHTML = html" in body and "LAT_STATE[k] === html" in body, (
        "cells are not compared and mutated, so every render touches every one")
    assert "grid.innerHTML = html" not in body, "the grid is rebuilt wholesale"
    # The one wholesale write is the column header row, before any cell exists.
    assert body.count("grid.innerHTML") == 1, (
        "the grid container is assigned more than once, which is a rebuild "
        "wearing a different name")
    assert 'classList.add("changed")' in body, (
        "nothing marks a cell as changed, so §05's settle has nothing to run on")


# ── the standing check: every server surface names its reader ───────────────
#
# `GUIDED-080` named the class from three instances and the adjudicator
# measured it at eight. Four scales of one defect: a function nothing calls
# (`GUIDED-058`), an endpoint nothing fetches (`GUIDED-075`, `-079`, `-080`), a
# payload field nothing renders (`GUIDED-078`), a capability nothing consults
# (`AUDIT-008`). Every instance was found by somebody going looking.
#
# So it stops being a finding and becomes a gate. A route is either fetched by
# the Guided door, or it is listed below WITH THE READER NAMED. There is no
# third option, and adding a route without doing one of the two fails this
# test — which is the only version of "watch for this class" that survives the
# loop that stops watching.
#
# THE PROBE HAD TO BE BUILT TWICE, and the first version is why this comment is
# long. A literal grep for the path reported `evidence/correlations` and
# `evidence/histograms` as unread. They are not: `runPull` builds
# `"/evidence/" + endpoint` from the palette chip's `data-endpoint`, so the
# full path never appears anywhere in the file. A check that cries wolf twice
# on its first run is a check the next person deletes, so the probe recognizes
# the composed form — a literal parent prefix plus a quoted leaf — and the
# unread count fell from nine to seven.
NOT_READ_BY_THE_DOOR = {
    "/project/{project_id}/findings":
        "Read by the Streamlit door. The Guided door gets the same findings "
        "inside the project payload, which is one fetch rather than two views "
        "of one list that can disagree.",
    "/project/{project_id}/grain":
        "Read by the Guided door through `/interview`, which serves the same "
        "question with its `why` and its `consumer` attached. This endpoint is "
        "the question's material without the plan around it.",
    "/project/{project_id}/lens":
        "Same as `/grain`: the Guided door reads the lens question from the "
        "interview plan, which is where the reason to ask it lives.",
    "/project/{project_id}/repeats":
        "Same as `/grain`. Questions 4 to 7 reach the Guided door through the "
        "interview plan, and `applies:false` is the ordinary answer.",
    "/project/{project_id}/gaps":
        "The `[AUTHOR REQUIRED]` gaps reach the Guided door inside `/draft`, "
        "which counts them beside the sentences they interrupt. This endpoint "
        "serves them alone, and is read by the export path.",
}

PROJECT_PREFIX = "/project/{project_id}"

# What counts as a reader. Named explicitly, because "somebody probably uses
# it" is the sentence this whole check exists to stop — a reason that does not
# name one of these or a ledger row is not a reason.
READERS = ("Streamlit door", "interview plan", "`/interview`", "`/draft`",
           "export path", "project payload")


def _routes_of(api_source: str):
    import re

    return re.findall(r'@app\.(?:get|post|delete|put)\("([^"]+)"\)', api_source)


def _is_fetched(path: str, page: str) -> bool:
    """Whether the Guided door reaches this route.

    Two forms, because the controller uses two. Most calls are a literal tail
    concatenated onto `"/project/" + P.id`. The pull affordances build theirs
    from the palette entry's `endpoint` field, so the path exists only at
    runtime — recognized here as a literal parent prefix plus a quoted leaf.
    """
    # A server-level route (`/capabilities`, `/job/{id}`) is probed the same
    # way as a project one, minus the prefix — the controller builds both by
    # concatenation, and only the prefix differs.
    tail = (path[len(PROJECT_PREFIX):] if path.startswith(PROJECT_PREFIX)
            else path)
    if tail == "/":
        return True                    # the page itself; it IS the reader
    if not tail:
        return '"/project/"' in page
    cut = tail.find("{")
    literal = tail if cut == -1 else tail[:cut]
    if literal.rstrip("/") and literal in page:
        return True
    parent, _, leaf = tail.rstrip("/").rpartition("/")
    # Only for NESTED paths: a one-segment tail matched this way would count
    # any quoted occurrence of the word, and `"grain"` is a question kind.
    if not parent:
        return False
    if (parent + "/") not in page:
        return False
    # The leaf arrives either as a quoted argument (`"correlations"`) or
    # concatenated onto the path (`+ "/cancel"`).
    if f'"{leaf}"' in page or f'"/{leaf}"' in page:
        return True
    # OR IT NEVER APPEARS IN THE FILE AT ALL. `runPull` composes
    # `"/evidence/" + endpoint`, and since `GUIDED-084` the endpoint arrives
    # from the SERVER's capability table rather than from a page-local list —
    # so `"correlations"` is not a string in `index.html` and the page fetches
    # it on every press. This is `GUIDED-083`'s correction one step further on:
    # a literal search answers *does this text appear* and the question is
    # *does this run*. Recognized from the server's own declaration, so a leaf
    # nobody declares still reads as unread.
    return leaf in _server_supplied_leaves() and "data-endpoint" in page


def _server_supplied_leaves() -> set:
    """The evidence leaves the capability table hands the page at runtime."""
    from turbotab import api as api_mod

    return {str(cap.get("endpoint")) for cap in api_mod.PULL_CAPABILITIES.values()
            if cap.get("endpoint")}


def test_every_server_surface_names_its_reader():
    root = Path(__file__).resolve().parent
    api_source = (root / "api.py").read_text(encoding="utf-8")
    page = (root / "web" / "index.html").read_text(encoding="utf-8")

    routes = _routes_of(api_source)
    assert len(routes) > 20, "the route scan found almost nothing"   # control
    assert len(page) > 20_000 and "renderAll" in page               # control
    # The probe itself, controlled: a route the page demonstrably fetches must
    # read as fetched, in both forms.
    assert _is_fetched(f"{PROJECT_PREFIX}/figures", page)           # literal
    assert _is_fetched(f"{PROJECT_PREFIX}/evidence/correlations", page)  # composed

    unread = [p for p in routes if not _is_fetched(p, page)]
    undeclared = [p for p in unread if p not in NOT_READ_BY_THE_DOOR]
    assert not undeclared, (
        "these routes are composed by the server and fetched by nothing in the "
        "Guided door, and no reader is named for them:\n  "
        + "\n  ".join(undeclared)
        + "\n\nEither wire the surface or add it to NOT_READ_BY_THE_DOOR with "
          "the reader named. A surface nobody reads is a promise nobody keeps.")
    # The gate has to be able to fail. A route the page cannot possibly fetch
    # must come out undeclared, or the check passes because it finds nothing.
    invented = "/project/{project_id}/a_surface_nobody_wrote"
    assert not _is_fetched(invented, page), (
        "the probe reports an endpoint that does not exist as fetched, so it "
        "would report a real one that way too")

    # The declaration must stay true: a route listed as unread that the page
    # HAS since started fetching is a stale excuse, and stale excuses are how a
    # list like this stops meaning anything.
    stale = [p for p in NOT_READ_BY_THE_DOOR if p not in unread]
    assert not stale, (
        "these are declared as not read by the door and the page fetches them "
        f"now: {stale}")

    for path, reason in NOT_READ_BY_THE_DOOR.items():
        assert len(reason) > 60, f"{path}: the reason is a shrug"
        assert "GUIDED-" in reason or any(r in reason for r in READERS), (
            f"{path}: the reason names neither a reader from {READERS} nor a "
            "ledger row tracking it as unread")


def test_the_capability_table_is_read_rather_than_reimplemented():
    """`/capabilities` exists so the interface cannot claim a capability the
    server does not have. That only holds if the page reads it.

    **This test asserted the defect until L35-A**, which made it a placeholder
    with a deadline; this is the deadline. `GUIDED-084`'s ruling was that the
    gate was never the page's argument — `GUIDED-005` put
    `MAX_FEATURES_FOR_GALLERY` in the engine *precisely so the page and the
    server cannot disagree about it* — so the fix is a fetch and a delete
    rather than a design.

    Driven rather than grepped for the half that is about behavior: the page's
    real controller runs, and `__harness.calls()` reports whether it asked. A
    literal search over `index.html` answers *does this text appear*, and the
    question is *does this run* (`LOOP.md` §06).
    """
    from turbotab import api as api_mod

    page = (Path(__file__).resolve().parent / "web" / "index.html").read_text(
        encoding="utf-8")
    assert api_mod.PULL_CAPABILITIES                                # control

    # THE SECOND IMPLEMENTATION IS GONE, not renamed. Asserted on the
    # SENTENCES rather than on a function name, because a sentence a user reads
    # that no server composed is the half `COPY_DECK.md` cannot review — and
    # that is a property of the strings, not of what the function was called.
    for composed in ("A per-feature gallery is offered up to",
                     "needs at least two numeric features",
                     "wall of plots",
                     "cannot be read is a claim that it can"):
        assert composed not in page, (
            f"the page still composes {composed!r}, so a user reads a sentence "
            "that no server wrote")
    assert "paletteExtras" not in page, (
        "the page still computes its own capability verdicts")

    # AND THE SERVER'S ANSWER IS PER PROJECT. A build-wide `built` cannot say
    # that a correlation matrix is unavailable on THIS table.
    client = _client()
    wide = _project(client, "metabolomics_untargeted.csv")
    narrow = _project(client, "clinic_visits.csv")
    caps_wide = client.get(f"/project/{wide['id']}/capabilities").json()
    caps_narrow = client.get(f"/project/{narrow['id']}/capabilities").json()
    assert caps_narrow["pulls"]["look::r8_collinearity"]["built"] is True
    assert caps_wide["pulls"]["look::r8_collinearity"]["built"] is False, (
        "a 396-column table is offered a live correlation-matrix chip, which "
        "is the affordance /capabilities exists to prevent")
    reason = caps_wide["pulls"]["look::r8_collinearity"]["not_built_reason"]
    assert reason and str(caps_wide["n_numeric"]) in reason, (
        "the reason does not name this table's feature count, so it is a "
        "build-wide sentence wearing a per-project costume")
    assert str(api_mod.get_capabilities.__module__)                  # control

    # THE PAGE ASKS. Driven, because a page that mentions a path and a page
    # that fetches it are different pages. The palette lives past the target
    # question, so the drive answers it first — an undriven palette would make
    # this assertion about a surface the controller never reached.
    pid = wide["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": "responder"}})
    seen = client.get(f"/project/{pid}").json()
    out = _drive(
        """
        var p = Promise.resolve();
        for (var i = 0; i < 24; i++) { p = p.then(function(){}); }
        p.then(function(){
          __harness.drainRaf();
          __emit({calls: __harness.calls().map(function(c){ return c.path; }),
                  bar: __harness.html('palette')});
        });
        """,
        _routes(client, seen, **{
            f"/project/{pid}/capabilities": caps_wide,
            f"/project/{pid}/interview?step=explore":
                client.get(f"/project/{pid}/interview?step=explore").json(),
        }),
        pid)
    assert f"/project/{pid}/capabilities" in out["calls"], (
        "the page never fetched the capability table, so whatever it renders "
        "on those chips it decided by itself")
    assert reason[:40] in out["bar"], (
        "the server's not-built reason was fetched and rendered nowhere, "
        "which is GUIDED-080's class rather than GUIDED-084's fix")


def test_the_shim_says_no_to_an_id_that_does_not_exist():
    """`GUIDED-077`, first half, asserted as the property it restores.

    `getElementById` used to AUTO-CREATE, so it never returned null and
    `if (!node)` was false for every id in the universe. Every branch keyed on
    *does this node exist yet* was unobservable — and the lattice proved what
    that costs: `if (!$("latGrid"))` could never be true, so the container was
    never written and the grid was built and never attached. `latticeBox` held
    0 characters while `latGrid` held 179, and the claim about it passed.
    """
    out = H.run(
        """
        __emit({
          missing: document.getElementById('no_such_id_anywhere') === null,
          declared: document.getElementById('askedQuestions') !== null,
          appended: (function(){
            var n = document.createElement('div');
            n.id = 'made_at_runtime';
            document.getElementById('askedQuestions').appendChild(n);
            return document.getElementById('made_at_runtime') === n;
          })(),
          removed: (function(){
            var host = document.getElementById('askedQuestions');
            host.removeChild(document.getElementById('made_at_runtime'));
            return document.getElementById('made_at_runtime') === null;
          })()
        });
        """,
        routes={}, search="")
    assert out["missing"] is True, (
        "the shim invented an element for an id nothing declares, so every "
        "`if (!node)` branch in the page is unobservable")
    assert out["declared"] is True, "an id in the markup stopped resolving"
    assert out["appended"] is True, (
        "a node created and appended is not findable, which is not how a "
        "browser behaves either")
    assert out["removed"] is True, (
        "a removed node stays findable, so 'did this leave?' cannot be asked")


def test_the_shim_serializes_what_was_appended_not_only_what_was_assigned():
    """`GUIDED-077`, second half. `innerHTML` returned only assigned markup, so
    a surface built by `appendChild` — every mutate-in-place renderer §05
    requires — probed as an empty container, and the workaround was a SECOND
    reader that walked the children. Two readers of one property is two answers
    to one question, and a claim written against the wrong one asserts nothing.

    Assigning replaces the children, as a browser does. That is what makes a
    rebuild observable: after it, the old nodes are gone from the tree and from
    `getElementById`, so a renderer that rebuilds can no longer pass a test
    written for one that mutates.
    """
    out = H.run(
        """
        var host = document.getElementById('askedQuestions');
        host.innerHTML = '<p>assigned</p>';
        var kid = document.createElement('span');
        kid.id = 'kid';
        kid.innerHTML = 'appended';
        host.appendChild(kid);
        var both = host.innerHTML;
        var viaRender = __harness.render('askedQuestions');
        host.innerHTML = 'repainted';
        __emit({both: both,
                same_as_render: both === viaRender,
                after_repaint: host.innerHTML,
                kid_survives: document.getElementById('kid') !== null});
        """,
        routes={}, search="")
    assert "assigned" in out["both"] and "appended" in out["both"], (
        "innerHTML reports one of the two ways content gets into an element")
    assert out["same_as_render"] is True, (
        "`html` and `render` answer different questions again, which is the "
        "hole this closed")
    assert out["after_repaint"] == "repainted", (
        "assigning innerHTML left the old children behind")
    assert out["kid_survives"] is False, (
        "a repainted-away node is still findable, so a rebuild can still pass "
        "for a mutation")

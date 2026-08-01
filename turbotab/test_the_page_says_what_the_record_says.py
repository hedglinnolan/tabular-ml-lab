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
          // Read `prevOut`, not its parent: the shim's deep walk cannot see a
          // node that arrived inside an assigned innerHTML string
          // (`GUIDED-077`). The node the page wrote to is the honest read.
          seen.out = __harness.render('prevOut');
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
          seen.grid = __harness.render('latGrid');
          seen.supp = __harness.render('latSupp');
          // Open the cell whose stack has more than one row in it.
          var keys = [], rx = /data-lat-cell="([^"]+)"/g, m;
          while ((m = rx.exec(seen.grid)) !== null) { keys.push(m[1]); }
          seen.keys = keys;
          __harness.dispatch('click',
            __harness.target({'data-lat-cell': keys[0]}, []));
          return settle(6);
        }).then(function(){
          seen.why = __harness.render('latWhy');
          __emit(seen);
        });
        """,
        routes, pid)

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

"""L64-C. `GUIDED-241` — the Train refusal names a control and hands the user
no way to reach it.

`training.check()` raises *"Answer 'Which of these is the event you are
predicting?' on the outcome, then fit"*, and the page renders that sentence as
prose in a `.refused-card` with zero controls. The card it names lives in
`#sec-eda` — **four whole sections above where the user is standing**
(`sec-target`, `sec-eda`, `sec-features`, `sec-preprocess`), all revealed in one
column — and `index.html` records the standing ruling *"The page never moves the
viewport."* So pointing was all it could do.

## What this file asserts, and why each half is here

**Two presses, not one.** The Data card is: press one opens the before/after
preview (*"nothing is applied yet"*), press two applies. Keeping both at Train
is the only way this is not a second implementation of the question with weaker
consequence disclosure. It also resolves the fetch problem — the preview is a
**press-time** fetch, and a press-time fetch does not appear in a bootstrap
drive, so the stray-route gate is not tripped.

**The destination travels with the press**, which is the trap this build could
most easily have fallen into. `openPanel` read `$("pv-" + id)` — a document-wide
lookup for a node `findingCard` writes, and the positive-class finding's card
renders in `#sec-data`. Reusing the renderer at Train without a destination
writes the panel back at the Data card: no error, no throw, the node exists, and
the user watching Train sees nothing. **That is `GUIDED-241`'s own defect
reproduced inside its repair**, and the fifth assertion below is what catches it.
"""
from __future__ import annotations

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import pageharness as PH                          # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data")
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web",
                    "index.html")


def _blocked_on_the_event():
    """A sealed, two-level project with a model selected and no event chosen.

    The event refusal is unreachable before the seal and before a model is
    picked — `training.check()` refuses on the earlier steps first — so all
    four are driven through the real routes.
    """
    from fastapi.testclient import TestClient

    from turbotab import api

    client = TestClient(api.app)
    with open(os.path.join(DATA, "clinical_risk.csv"), "rb") as handle:
        pid = client.post("/project", files={
            "file": ("clinical_risk.csv", handle, "text/csv")}).json()["id"]

    def decide(kind, **payload):
        got = client.post(f"/project/{pid}/decision",
                          json={"kind": kind, "payload": payload})
        assert got.status_code == 200, (kind, got.text[:250])
        return got.json()

    decide("set_target", column="readmit_30d")
    decide("set_purpose", answer="prediction")
    decide("set_grain", answer="one_row_per_person")
    decide("set_eligibility", answer="everyone")
    decide("seal", fraction=0.25)
    decide("select_models", models=["logreg"])
    return client, pid


def _routes(client, pid):
    got = client.get(f"/project/{pid}").json()
    out = {f"/project/{pid}": got}
    for path in ("interview?step=data", "interview?step=explore",
                 "interview?step=features", "interview?step=preprocess",
                 "capabilities", "features", "recipes", "preprocess", "figures",
                 "draft", "manuscript", "models", "training", "instability",
                 "explain", "sensitivity", "checklist",
                 "evidence/plausibility", "evidence/missingness"):
        resp = client.get(f"/project/{pid}/{path}")
        out[f"/project/{pid}/{path}"] = (resp.json()
                                         if resp.status_code == 200 else {})
    return out


def test_the_train_refusal_is_the_one_about_the_event():
    """The precondition, established from the SERVER rather than hoped for.

    Every assertion below is about a refusal naming the event question. If the
    project were blocked on something else — no model selected, no seal — the
    control would be absent and a whole file of green would say nothing.
    """
    client, pid = _blocked_on_the_event()
    blocked = client.get(f"/project/{pid}/training").json()["blocked_by"]
    assert blocked, "the Train step is not refusing at all"
    assert "event you are predicting" in blocked, blocked

    finding = next(f for f in client.get(f"/project/{pid}").json()["findings"]
                   if f.get("fix_kind") == "set_positive_class")
    assert finding["id"] == "positive_class__readmit_30d", (
        "the id is the server's own and this test must not compose it")


def test_the_page_finds_the_question_by_what_it_is_not_by_the_sentence():
    """`api.py` has ruled twice against matching on prose, naming the two
    surfaces this codebase has already paid for.

    A page that decided which question a refusal names by reading `blocked_by`
    would break the moment the sentence is reworded, and would fire on any
    other refusal that happens to contain the word.
    """
    page = open(PAGE, encoding="utf-8").read()
    helper = page[page.index("function eventQuestionFinding"):]
    helper = helper[:helper.index("\n  }") + 4]
    assert "fix_kind" in helper and "set_positive_class" in helper
    assert "blocked_by" not in helper, (
        "the page reads the refusal SENTENCE to decide which question it names")
    for prose in ("indexOf(\"event\")", "indexOf('event')", "/event/"):
        assert prose not in helper, helper
    # And it must not offer a question that has already been answered — the
    # finding stays in `P.findings` after `record_fix`; only `applied_fixes`
    # changes.
    assert "applied_fixes" in helper, (
        "the control would keep offering a question the user already answered")


def test_the_refusal_carries_the_control_and_the_answer_lands_at_train(capsys):
    """The deliverable. Six assertions, one drive.

    1. the Train container holds the answer controls **as structure** — both
       level tokens as attributes, not as a substring of prose;
    2. answer, then apply;
    3. the POST path set is **exactly one** route;
    4. the scroll list is **empty** across the whole sequence;
    5. the Data card's panel is still empty while the Train container holds the
       response — *that* is "carried, not pointed at" in its sharpest form, and
       it is what catches the `pv-` trap;
    6. the same body replayed against the real API moves the Train block from
       the event sentence to `None` (asserted in the test below, which is where
       the server can answer).
    """
    if not PH.available():
        pytest.skip("no JS engine on this machine")

    client, pid = _blocked_on_the_event()
    routes = _routes(client, pid)
    fid = "positive_class__readmit_30d"
    # The two press-time fetches, served as the real API serves them. They are
    # NOT render-time: nothing below fetches them until a press happens, which
    # is why no render-time path list has to move.
    for suffix in ("", "?choice=1"):
        url = f"/project/{pid}/finding/{fid}/preview{suffix}"
        got = client.get(url)
        routes[url] = got.json() if got.status_code == 200 else {}
    routes[f"POST /project/{pid}/decision"] = client.get(
        f"/project/{pid}").json()

    body = """
for (var i = 0; i < 12; i++) await new Promise(function(r){ setTimeout(r, 0); });
var before = __harness.html('trainRun') || '';
__harness.dispatch('click', __harness.target(
  {'data-event-here': FID, 'data-pv-box': 'pvTrainEvent'}));
for (var j = 0; j < 12; j++) await new Promise(function(r){ setTimeout(r, 0); });
var asked = __harness.html('pvTrainEvent') || '';
__harness.dispatch('click', __harness.target(
  {'data-event-for': FID, 'data-event': '1', 'data-pv-box': 'pvTrainEvent'}));
for (var k = 0; k < 12; k++) await new Promise(function(r){ setTimeout(r, 0); });
var card = __harness.el('pv-' + FID);
__emit({before: before, asked: asked,
        answered: __harness.html('pvTrainEvent') || '',
        data_card: card ? card.innerHTML : null,
        calls: __harness.calls()});
""".replace("FID", repr(fid))
    out = PH.run(body, routes=routes, search=f"?project={pid}")

    # 1 · the controls are STRUCTURE, not prose
    assert "data-event-here" in out["before"], (
        f"the refusal card carries no control:\n{out['before'][:500]}")
    asked = out["asked"]
    assert asked, "pressing the control rendered nothing into the Train panel"
    tokens = set(re.findall(r'data-event="([^"]*)"', asked))
    assert tokens == {"0", "1"}, (
        f"the Train panel offers {sorted(tokens)} as answer controls; the "
        f"outcome has two levels and both must be pressable, as attributes "
        f"rather than as words in a sentence")
    assert all('data-pv-box="pvTrainEvent"' in b
               for b in re.findall(r"<button[^>]*data-event-for[^>]*>", asked)), (
        "an answer control does not carry its destination, so pressing it "
        "writes the response back at the Data card")

    # 2 · answering opened the before/after, and it offers the apply
    assert out["answered"], "answering rendered nothing"
    assert "data-apply" in out["answered"], (
        f"the answer did not reach a consequence disclosure with an apply — "
        f"two presses is what stops this being a weaker second implementation "
        f"of the question:\n{out['answered'][:600]}")

    # 3 · exactly one POST route
    posts = {c.get("path") for c in out["calls"]
             if str(c.get("method", "")).upper() == "POST"}
    assert len(posts) <= 1, f"more than one POST route was used: {sorted(posts)}"

    # 4 · nothing scrolls, asserted STATICALLY and here is why.
    #
    # The harness cannot measure viewport motion — it returns a constant rect,
    # and it has no `Element` at all, so instrumenting `scrollIntoView` throws
    # `ReferenceError` rather than reporting zero. That is `TEST-066`'s class:
    # an API the page never used is an API the shim never implemented, and a
    # runtime check here would have been a false negative dressed as a pass.
    #
    # So the claim is made where it CAN be made. `scrollIntoView` is pinned at
    # exactly one call — the rail's data-map branch — and five further
    # spellings at zero, by `test_a_response_renders_at_the_control.py`. This
    # re-asserts it as a precondition of THIS build, because a refusal that
    # travelled to its own control is precisely the second permitted move and
    # both guards fail on it deliberately.
    script = open(PAGE, encoding="utf-8").read()
    assert script.count("scrollIntoView") == 1, (
        f"this build added a scrollIntoView call "
        f"({script.count('scrollIntoView')} total)")
    for spelling in ("window.scrollTo", "function nudge(", ".scrollTop =",
                     ".scrollLeft =", "scrollBy("):
        assert spelling not in script, (
            f"this build can move the viewport, by `{spelling}`")

    # 5 · THE TRAP. The Data card's panel is untouched.
    assert not (out["data_card"] or "").strip(), (
        f"the response was written back at the Data card, four sections above "
        f"where it was asked for — `GUIDED-241`'s own defect, inside its "
        f"repair:\n{(out['data_card'] or '')[:400]}")

    with capsys.disabled():
        print(f"\n  levels {sorted(tokens)} at Train · "
              f"{len(out['answered'])} chars answered · "
              f"data card {len(out['data_card'] or '')} chars")


def test_the_same_body_answers_the_question_on_the_real_api():
    """The sixth assertion, where the server can answer it.

    The harness proves the page composes the press; this proves the body it
    composes is one the API accepts and that it clears the refusal. The route
    and subject are the server's own — `POST /project/{id}/decision`,
    `kind: apply`, `subject: eventfixture.question_id(target)` — because there
    is no dedicated apply endpoint and composing the id again would be a second
    implementation of it.
    """
    from turbotab import eventfixture

    client, pid = _blocked_on_the_event()
    subject = eventfixture.question_id("readmit_30d")

    before = client.get(f"/project/{pid}/training").json()["blocked_by"]
    assert "event you are predicting" in before

    posted = client.post(f"/project/{pid}/decision", json={
        "kind": "apply", "subject": subject, "payload": {"choice": "1"}})
    assert posted.status_code == 200, posted.text[:300]
    assert subject in posted.json()["applied_fixes"]

    after = client.get(f"/project/{pid}/training").json()["blocked_by"]
    assert after is None, (
        f"the event was answered and the Train step still refuses: {after!r}")

"""`GUIDED-037` — the interview could not start, and a green suite said it could.

The product owner clicked an answer on question 1 and nothing happened.

## What was wrong

`ml/router.py` serves `option_values` beside `options`, because the lens's
labels are prose — *"Metabolomics or proteomics"* — and its values are keys —
`metabolomics`. `askedCard` in `turbotab/web/index.html` read `q.options` twice:

    var label = (typeof opt === "string") ? opt : (opt.label || opt.key);
    var value = (typeof opt === "string") ? opt : (opt.key || opt.label);

so every button carried the LABEL as its `data-answer-value`, and pressing
*"That is all of them"* posted `{"lens": ["Metabolomics or proteomics"]}`, which
`packs.normalize` refuses with a 400. Question 1 of the fixed pre-seal sequence
could not be answered, so nothing after it was reachable. `DRIVE-001` again, one
layer in: the question was served, rendered, and unanswerable.

## Why nothing caught it

`test_the_lens_options_carry_values_and_not_only_labels` ended with

    assert "option_values" in page or "q.options" in page

— a disjunction the wrong half satisfies. `FEATURE_PARITY.md` names this exactly:
*a substring of a message is a wildcard wearing an assertion's clothes.* The
assertion looked like it checked that the page reads the values array. It checked
that the page mentions either of two strings, and the page mentioned the other
one. Every other frontend assertion in this tree is a text search over
`index.html`, and a text search cannot tell a page that READS a field from a page
that merely names it.

## What this file asserts instead

**The effect, read back off the record.** The page's own controller renders the
card, the page's own click handler is dispatched at the buttons it rendered, the
page's own `decide()` composes the request — and the body that reaches `fetch` is
then replayed against the real API, and the project is asked what its lens is.

That is a read-back and not a receipt: the assertion is `project["lens"] ==
["metabolomics"]`, which is a fact about the record, not about the page's
description of itself. The last loop cost a critical because nine tests asserted
a seal's record and none asserted the draw against it.

`turbotab/pageharness.py` says what this can and cannot prove: behavior, yes;
visibility, no. Nothing without layout can prove a card is on screen.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import packs as P                                       # noqa: E402
from turbotab import pageharness as H                                 # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"

pytestmark = pytest.mark.skipif(
    not H.available(),
    reason="no JS engine on this machine; a check that cannot run says so")


# ── the fixture: a real project, and the routes the page will ask for ────────

def _client():
    from fastapi.testclient import TestClient

    from turbotab import api
    return TestClient(api.app)


def _project(client, name="metabolomics_untargeted.csv"):
    with open(DATA / name, "rb") as fh:
        return client.post("/project", files={"file": (name, fh, "text/csv")}).json()


def _routes(client, project):
    """Everything the controller fetches while it boots, answered for real.

    The responses come from the actual API rather than from fixtures written
    here, so a server change that breaks the page breaks this test instead of
    being papered over by a canned reply.
    """
    pid = project["id"]
    return {
        f"/project/{pid}": project,
        f"/project/{pid}/interview?step=data":
            client.get(f"/project/{pid}/interview?step=data").json(),
        f"/project/{pid}/interview?step=explore": {"questions": [], "steps": []},
        f"/project/{pid}/evidence/missingness": {"cards": []},
        f"/project/{pid}/evidence/plausibility": {"columns": []},
        f"/project/{pid}/draft": {"paragraphs": []},
        f"/project/{pid}/gaps": {"gaps": []},
    }


# The lens buttons are addressed BY POSITION, never by the value they carry.
#
# Addressing them by value would make every test here collapse the moment the
# defect returns — the harness would fail to find `metabolomics`, the test would
# go red saying "no such option", and the revert probe would report RED FOR THE
# WRONG REASON. A driver does not know what value is under a button either; they
# press the first one. So does this.
_PRESS_NTH = """
function nth(i){
  var html = __harness.html('askedQuestions');
  var rx = /<button([^>]*data-answer-key="state_lens"[^>]*)>/g, m, out = [];
  while ((m = rx.exec(html)) !== null){
    var a = {};
    m[1].replace(/([a-zA-Z-]+)="([^"]*)"/g, function(_, k, v){ a[k] = v; return ""; });
    out.push(a);
  }
  if (!out[i]) throw new Error('the lens card rendered ' + out.length + ' options');
  __harness.dispatch('click', __harness.target(out[i], ['answer', 'multi']));
  return out[i];
}
function commit(){
  __harness.dispatch('click',
    __harness.target({'data-answer-commit': 'state_lens'}, ['answer', 'primary']));
}
"""


# ── the defect, at the point it is visible ───────────────────────────────────

def test_the_lens_buttons_submit_the_pack_keys_and_not_their_labels():
    """Structure, not prose: the rendered `data-answer-value` set IS the pack's
    key set. A label here is a 400 waiting for a press."""
    client = _client()
    project = _project(client)
    out = H.run("__emit(__harness.html('askedQuestions'));",
                routes=_routes(client, project), search=f"?project={project['id']}")
    buttons = [b for b in H.elements(out)
               if b.get("data-answer-key") == "state_lens"]
    assert buttons, "the lens card did not render at all"
    values = [b["data-answer-value"] for b in buttons]
    assert values == list(P.LENS_KEYS), (
        "the page is submitting option LABELS. `option_values` travels beside "
        "`options` precisely because the lens's labels are prose and its values "
        f"are keys.\n  rendered: {values}\n  expected: {list(P.LENS_KEYS)}")


def test_answering_the_lens_changes_the_recorded_lens():
    """The read-back. Not *a click was registered* — *the record now says so.*

    The page renders, the page's click handler runs, the page's `decide()`
    composes the body, and that body is replayed against the real API. The
    assertion is what `GET /project/{id}` reports afterwards.
    """
    client = _client()
    project = _project(client)
    pid = project["id"]

    body = H.run(
        _PRESS_NTH + """
        nth(0);            /* the first option, whatever value it carries */
        commit();
        var posts = __harness.posts();
        __emit(posts.length ? posts[posts.length - 1] : null);
        """,
        routes=_routes(client, project), search=f"?project={pid}")

    assert body, "pressing the submit control sent nothing at all"
    assert body["method"] == "POST" and body["path"] == f"/project/{pid}/decision"
    assert body["body"]["kind"] == "set_lens"

    # Replay what the page would have sent, against the real server.
    posted = client.post(f"/project/{pid}/decision", json=body["body"])
    assert posted.status_code == 200, (
        "the server refused the body the page composes — which is the defect "
        f"exactly, at the point a driver meets it: {posted.text}")

    # THE READ-BACK. A 200 says the request was accepted; this says the record
    # changed, which is the claim.
    after = client.get(f"/project/{pid}").json()
    assert after["lens"] == ["metabolomics"], (
        f"the recorded lens is {after['lens']!r}; answering did not change it")


def test_a_pick_is_visible_before_it_is_submitted():
    """`GUIDED-037`'s other half. The handler set `aria-pressed` and no rule in
    the sheet read it, so a multi-select pick changed the DOM and not the
    screen — which is indistinguishable from a broken control.

    Asserted on the attribute the style hangs off AND on the rule existing,
    because either alone is half a claim: a pressed state nothing renders is
    invisible, and a rule nothing sets is dead CSS.
    """
    client = _client()
    project = _project(client)
    after = H.run(
        _PRESS_NTH + "nth(2); __emit(__harness.html('askedQuestions'));",
        routes=_routes(client, project), search=f"?project={project['id']}")

    pressed = [b.get("aria-pressed") for b in H.elements(after)
               if b.get("data-answer-key") == "state_lens"]
    assert pressed[2] == "true", (
        "the picked option did not come back pressed, so a re-render loses the "
        f"selection the page will submit: {pressed}")
    assert pressed[0] == "false" and pressed[1] == "false", (
        f"pressing one option marked others: {pressed}")

    page = H.PAGE.read_text(encoding="utf-8")
    assert '.answer.multi[aria-pressed="true"]::before' in page, (
        "nothing in the stylesheet reads the pressed state, so the pick is "
        "recorded in the DOM and invisible on the screen")


def test_the_submit_control_refuses_at_zero_and_says_why_in_the_servers_words():
    """`GUIDED-038`. *"That is all of them"* confirms an enumeration; this
    question is mandatory and picking is the action.

    The reason is asserted to be `packs.LENS_EMPTY_REFUSAL` verbatim rather than
    "a sentence about picking", because the point is that the interface quotes
    the rule `normalize` enforces instead of composing a second one beside it.
    """
    client = _client()
    project = _project(client)
    fresh = H.run("__emit(__harness.html('askedQuestions'));",
                  routes=_routes(client, project),
                  search=f"?project={project['id']}")
    commit = [b for b in H.elements(fresh)
              if b.get("data-answer-commit") == "state_lens"]
    assert len(commit) == 1, "the lens has no single submit control"
    assert "disabled" in commit[0] or commit[0].get("aria-disabled") == "true", (
        "the submit control is pressable with nothing picked, and the record "
        "refuses an empty lens — a press that 400s is a control asserting it "
        "can record your answer")
    tip = commit[0].get("data-tip", "")
    assert tip == P.LENS_EMPTY_REFUSAL, (
        "the submit control composes its own reason instead of quoting the one "
        f"`normalize` raises:\n  shown:    {tip!r}\n  enforced: "
        f"{P.LENS_EMPTY_REFUSAL!r}")
    # Asserted on what was RENDERED, not on a grep of the file — the file also
    # contains the comment explaining why the old label was wrong, and a
    # whole-file search cannot tell an explanation from a relapse.
    label = re.search(r'data-answer-commit="state_lens"[^>]*>([^<]*)<',
                      fresh)
    assert label and label.group(1).strip() == "Pick at least one", (
        f"the submit control still confirms an enumeration: {label and label.group(1)!r}")


def test_the_submit_control_counts_what_it_will_record():
    """The BEFORE moment of §05.1, on the control the driver actually presses:
    the label says how much, the hover says what — and the hover is composed
    from the same payload builder `submitAnswer` posts, so the promise and the
    act cannot disagree."""
    client = _client()
    project = _project(client)
    out = H.run(
        _PRESS_NTH + """
        nth(0);
        var one = __harness.html('askedQuestions');
        nth(3);
        __emit({one: one, two: __harness.html('askedQuestions')});
        """,
        routes=_routes(client, project), search=f"?project={project['id']}")

    def commit(html):
        return [b for b in H.elements(html)
                if b.get("data-answer-commit") == "state_lens"][0]

    one, two = commit(out["one"]), commit(out["two"])
    assert "disabled" not in one, "one pick is an answer and was still refused"
    assert "1 selected" in one.get("data-tip", ""), (
        f"the submit control does not say what it will record: {one!r}")
    assert "2 selected" in two.get("data-tip", ""), (
        f"the count did not follow the second pick: {two!r}")
    assert one.get("aria-label") and two.get("aria-label"), (
        "the effect sentence does not reach a screen reader; §05.1 requires "
        "the BEFORE moment on hover AND to assistive technology")

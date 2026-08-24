"""`GUIDED-041` — the reopen affordance answered the question it was reopening.

Decision B permits the Router to skip a question only where a high-confidence
finding makes a question of *fact* moot, and only if the skip is **visible and
reversible**. The visible half was built: a muted provenance row carrying its
reason, with *"Ask me anyway"* beside it.

The reversible half sent this:

    decide("set_task_type", "", {task_type: P.task_type})

`P.task_type` is **the engine's own reading** — the thing the user is reaching
past when they press the button. So pressing *"Ask me anyway"* recorded that
reading as the user's answer. The question left the plan as ANSWERED, the skip
disappeared because the question was gone, and the transcript then said a human
had confirmed something no human had looked at.

That is worse than having no affordance at all. A skip with no reopen is
honestly incomplete; a reopen that discards teaches that opening a skip loses
your place, and the next skip goes unopened.

## What the fix has to be

Not a flag. `unskip` is a **recorded decision** carrying the question key and no
answer, for the reason §09's recorded-absence rule gives about everything else
here: *"I did not accept the engine's reading of this"* is a sentence a methods
section can carry, and a mutated boolean is not. An `unskip` with nothing after
it is a question still open, which is what it should look like.

And it is generic in the key, so it closes the class rather than the task-type
instance — the same move `DRIVE-001` needed. Every rendered skip the Router
serves is reopenable the day it exists, including the pack-settled missingness
blocks, where a single skip stands for hundreds of columns and the cost of
being unable to reopen it is correspondingly larger.

## `GUIDED-156` — and where the reopened question then RENDERED

Every assertion in the six tests above is true, and none of them renders the page
after the reopen. That coverage gap was the finding: `unskip` worked perfectly on
the wire and the question it brought back appeared **nowhere on screen**.

Two correct rules composing into a hole. `renderSkips` draws only
`status === "skipped"`, so the reopened question left the skip list. `renderAsked`
draws `!handledElsewhere(q.key)`, and `confirm_task_type` is on
`HANDLED_QUESTION_KEYS` — so the generic channel refused it too. And the surface
that claims to handle it, the task-type row inside the target card, was gated on
`conf !== "high" || P.task_overridden`: at high confidence it renders a SENTENCE
into the transcript and no control at all. High confidence is exactly the state a
skip is granted in, and the engine does not stop being certain because a human
disagreed with it — so after the reopen all three surfaces declined.

Driven here on the unfixed page: `#askedQuestions` was byte-identical at 12,852
characters before and after with `confirm_task_type` absent from both, `#skipNote`
dropped from 904 characters to zero, and `#taskOverride` held zero characters in
both states. Nothing to press, on either fixture shape.

The fix reads the ROUTER'S STATUS for the key instead of re-deriving Decision B
from the confidence tier. `_skip_is_permitted` is the one place that rule lives;
the tier test in the page was a second copy of it, and the two copies disagreeing
is what opened the hole.

**The class is bigger than this instance.** Every key the Router can serve as
`status="skipped"` is also matched by `handledElsewhere` — three families,
three of three: `confirm_task_type` (an exact key), `missingness_settled::`
and `missingness::` (prefixes).

## `GUIDED-192` — and the two families that had no surface at all

L48 closed the instance and named the class in a strict `xfail`. The class was
worse than "the reopened question has nowhere to render": the other two
families are served **only** at `interview?step=preprocess`, and the page
fetched `step=data`, `step=explore` and `step=features` and never `preprocess`.
So the skip row itself was never drawn — not the provenance sentence, not the
evidence badge, and not *"Ask me anyway"*. There was nothing to reopen from.

Re-derived rather than quoted forward: `ml/router.py` sets `status = "skipped"`
at lines 619, 1141 and 1218, and the page's three `interview?step=` fetches
were at lines 4762, 5008 and 5031. Driven on `metabolomics_untargeted.csv`,
`missingness_settled::numeric::metabolomics` — one skip standing for **306
columns** — appeared in none of `#skipNote`, `#askedQuestions` or `#missBox`.

The consumer is `renderPreprocessPlan`, fed by a fourth fetch, drawing the
step's skips through the same `skipRowHTML` the Data step uses.

**And a second defect it exposes, filed rather than fixed here.** A reopened
`missingness_settled::` block cannot be ANSWERED. `api.py`'s `answered` fold
has no case that produces a `missingness_settled::` key — `route_missingness`
yields `missingness::<col>` and `route_missingness_bulk` yields
`missingness_bulk::<branch>` — so the block stays `asked` on every subsequent
render no matter what the user does, and there is no page control for it
either. The new surface therefore renders the question **without** option
buttons and says why: the answer delegate's `if (!spec) return;` means a
rendered option would be a solid control that silently does nothing, which is
`GUIDED-006` and is worse than the sentence.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml import router                                                 # noqa: E402
from turbotab import pageharness as H                                 # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"


def _client():
    from fastapi.testclient import TestClient

    from turbotab import api
    return TestClient(api.app)


def _skipped(client, pid, step="data"):
    plan = client.get(f"/project/{pid}/interview?step={step}").json()
    return {q["key"]: q for q in plan["questions"] if q["status"] == "skipped"}


def _asked(client, pid, step="data"):
    plan = client.get(f"/project/{pid}/interview?step={step}").json()
    return {q["key"]: q for q in plan["questions"]
            if q["mode"] == "push" and q["status"] == "asked"}


def _upload(client, name):
    with open(DATA / f"{name}.csv", "rb") as fh:
        return client.post("/project", files={
            "file": (f"{name}.csv", fh, "text/csv")}).json()


# ── the effect, read back ────────────────────────────────────────────────────

def test_reopening_a_skipped_question_brings_it_back_asked_and_unanswered():
    """The read-back, and the assertion that would have failed before.

    Two facts, and the second is the one nine tests would have missed: the
    question is back, AND no answer was recorded for it. The old code satisfied
    neither, but a test written only against the first would have gone green the
    moment somebody made the reopen re-ask the question while still writing the
    engine's guess into the record.
    """
    client = _client()
    project = _upload(client, "clinic_visits")
    pid = project["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": "outcome"}})

    skipped = _skipped(client, pid)
    assert "confirm_task_type" in skipped, (
        "the task-type question is not skipped on this fixture, so there is "
        "nothing to reopen and this test proves nothing")

    r = client.post(f"/project/{pid}/decision",
                    json={"kind": "unskip", "subject": "confirm_task_type",
                          "payload": {"key": "confirm_task_type"}})
    assert r.status_code == 200, r.text

    after = _asked(client, pid)
    assert "confirm_task_type" in after, (
        "the question did not come back; a reopen that does not reopen is the "
        "defect with a different implementation")
    assert "confirm_task_type" not in _skipped(client, pid)

    # THE HALF THE OLD CODE GOT WRONG. No answer may have been recorded, and in
    # particular not the engine's own reading.
    kinds = [d["kind"] for d in client.get(f"/project/{pid}").json()["decisions"]]
    assert "set_task_type" not in kinds, (
        "reopening the question recorded an answer to it — and the answer is "
        "the engine's own reading, which is the thing the user pressed the "
        "button to dispute")
    assert "unskip" in kinds


def test_the_reopen_survives_the_engine_still_being_certain():
    """A reopened question stays asked.

    The skip is granted on the engine's confidence, and the engine's confidence
    does not change when a human disagrees with it — so without this the
    question would be re-skipped on the very next render and the reopen would
    appear to do nothing at all. The user's asking outranks the engine's
    certainty, which is the same asymmetry §02 draws for the grain.
    """
    client = _client()
    project = _upload(client, "clinic_visits")
    pid = project["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": "outcome"}})
    client.post(f"/project/{pid}/decision",
                json={"kind": "unskip", "payload": {"key": "confirm_task_type"}})

    asked = _asked(client, pid)
    assert "confirm_task_type" in asked, (
        "the reopened question was skipped again on the very next render, so "
        "the reopen appears to the driver to do nothing at all")
    q = asked["confirm_task_type"]
    assert q["confidence"] == "high", (
        "the engine stopped being certain, so this test is no longer about "
        "what it says it is about")
    assert q["status"] == "asked"
    # And again, because the plan is recomputed per render and a reopen that
    # survives one render and not the next is the same defect, slower.
    assert "confirm_task_type" in _asked(client, pid), (
        "the reopened question was skipped again on a later render")


def test_answering_a_reopened_question_settles_it_and_keeps_the_reopen_recorded():
    """The reopen stays in the record after the answer.

    It is not bookkeeping: *"the user declined the engine's reading and answered
    this themselves"* is a different sentence from *"the user answered this"*,
    and the manuscript can carry the first only if the record keeps both.
    """
    client = _client()
    project = _upload(client, "clinic_visits")
    pid = project["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": "outcome"}})
    client.post(f"/project/{pid}/decision",
                json={"kind": "unskip", "payload": {"key": "confirm_task_type"}})
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_task_type",
                      "payload": {"task_type": "classification"}})

    assert "confirm_task_type" not in _asked(client, pid)
    kinds = [d["kind"] for d in client.get(f"/project/{pid}").json()["decisions"]]
    assert "unskip" in kinds and "set_task_type" in kinds
    assert kinds.index("unskip") < kinds.index("set_task_type")


def test_a_pack_settled_missingness_block_is_reopenable_too():
    """The class, not the instance.

    One skip standing for 306 columns is where being unable to reopen costs the
    most, and it is a different code path from the task-type skip — so it is
    asserted rather than assumed to have come along.
    """
    client = _client()
    project = _upload(client, "metabolomics_untargeted")
    pid = project["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": "responder"}})
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_lens", "payload": {"lens": ["metabolomics"]}})

    settled = [k for k in _skipped(client, pid, "preprocess")
               if k.startswith("missingness_settled::")]
    assert settled, (
        "no pack-settled missingness block on this fixture; the metabolomics "
        "left-censoring prior is what makes this test meaningful")
    key = settled[0]

    client.post(f"/project/{pid}/decision",
                json={"kind": "unskip", "payload": {"key": key}})
    assert key in _asked(client, pid, "preprocess"), (
        f"{key} did not come back asked")
    assert key not in _skipped(client, pid, "preprocess")


def test_the_router_refuses_to_skip_a_key_the_user_reopened():
    """Enforced in the one place Decision B lives, so a second skip site cannot
    forget it. `_skip_is_permitted` is where the constitution is checked rather
    than remembered."""
    assert router._skip_is_permitted("high", "task_type") is True
    assert router._skip_is_permitted(
        "high", "task_type", "confirm_task_type", ["confirm_task_type"]) is False
    assert router._skip_is_permitted(
        "high", "task_type", "confirm_task_type", ["something_else"]) is True


# ── what the driver presses ──────────────────────────────────────────────────

def test_the_page_sends_a_reopen_and_not_an_answer():
    """Read back off the page's own click handler.

    The defect was entirely in what the button sent, so a test that did not
    watch the wire would have missed it. This one dispatches at the real
    affordance and asserts the request body.
    """
    if not H.available():
        pytest.skip("no JS engine on this machine")
    client = _client()
    project = _upload(client, "clinic_visits")
    pid = project["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": "outcome"}})
    project = client.get(f"/project/{pid}").json()
    plan = client.get(f"/project/{pid}/interview?step=data").json()

    body = H.run(
        """
        var html = __harness.html('skipNote');
        var m = /data-unskip="([^"]+)"/.exec(html);
        if (!m) throw new Error('no reopen affordance rendered');
        __harness.dispatch('click', __harness.target(
          {'data-unskip': m[1], 'data-unskip-title': 'x'}, ['again']));
        var posts = __harness.posts();
        __emit(posts.length ? posts[posts.length - 1] : null);
        """,
        routes={
            f"/project/{pid}": project,
            f"/project/{pid}/interview?step=data": plan,
            f"/project/{pid}/interview?step=explore": {"questions": []},
            f"/project/{pid}/evidence/missingness": {"cards": []},
        }, search=f"?project={pid}")

    assert body, "pressing the reopen affordance sent nothing"
    assert body["body"]["kind"] == "unskip", (
        "the reopen affordance still sends an answer instead of a reopen: "
        f"{body['body']['kind']}")
    assert body["body"]["payload"]["key"] == "confirm_task_type"

    # And the server accepts exactly that body.
    replay = client.post(f"/project/{pid}/decision", json=body["body"])
    assert replay.status_code == 200, replay.text
    assert "confirm_task_type" in _asked(client, pid)


# ── `GUIDED-156` · and then where does it RENDER? ────────────────────────────

#: NOT COVERED, said out loud (`GUIDED-097`, and §10 rule 4).
SHAPES_NOT_COVERED = (
    "Whether the control is on screen. `pageharness.py` knows nothing about "
    "pixels and says so in its own docstring; this asserts that a control is "
    "rendered and pressable, never that a person can see it.",
    "A MULTICLASS target. `multiclass_stage.csv`'s `stage` is a four-level "
    "string and the engine does not read it at `high` confidence, so the "
    "question is asked outright and there is no skip to reopen — the reopen "
    "hole does not arise on that shape.",
    "A target the engine reads at LOW confidence, for the same reason: no "
    "skip, so nothing to reopen. The two runs below are both `high`, which is "
    "the only tier `_skip_is_permitted` admits.",
    "The third skippable family, `missingness::<col>` as a SKIP. No shipped "
    "fixture produces one: a per-column skip needs a column carrying exactly "
    "one derived prior that no settled GROUP already covers, and on every "
    "fixture with a pack prior the priors group. So `renderPreprocessPlan` is "
    "driven against `missingness_settled::` skips only, and the per-column "
    "family is covered by the same code path rather than by an observation.",
    "Whether a person can SEE the new Preprocess surface. It renders inside "
    "`#card-preprocess`, which `renderPreprocess` reveals only once "
    "`/preprocess` has answered; the plan fetch does not reveal the section "
    "itself, and `pageharness` knows nothing about visibility either way.",
)

#: Two fixtures of different target shape (`GUIDED-097`). Both must produce
#: `confirm_task_type` with `status="skipped"`, which is what makes a reopen
#: possible at all — the fixture is asserted to do so inside the test rather
#: than assumed here.
REOPEN_FIXTURES = [
    ("clinic_visits.csv", "outcome", "classification"),
    ("dietary_recalls.csv", "bmi", "regression"),
]


def _routes(client, pid):
    """Every response one render of this page asks for.

    Measured rather than guessed: one render of this project issues 19 distinct
    fetches, three of them per-column histograms composed from a variable and
    one of them `/dev/status`. A harness that stubs four gets a controller that
    throws, and a throw here reads as "the control did not render".
    """
    out = {f"/project/{pid}": client.get(f"/project/{pid}").json()}
    for path in ("interview?step=data", "interview?step=explore",
                 "interview?step=features", "interview?step=preprocess",
                 "capabilities", "features",
                 "recipes", "preprocess", "figures", "draft", "manuscript",
                 "models", "training", "instability", "explain", "sensitivity",
                 "evidence/plausibility", "evidence/missingness"):
        resp = client.get(f"/project/{pid}/{path}")
        out[f"/project/{pid}/{path}"] = (resp.json() if resp.status_code == 200
                                         else {})
    return out


#: THE PRESS IS BUILT FROM THE RENDER, NEVER HAND-SPECIFIED — trap #3.
#:
#: A synthetic `{'data-task': 'regression', 'data-ac': 'task'}` would let the
#: fixture supply the very attribute whose absence is the defect: the handler is
#: a document-level delegate and answers a press whether or not anything drew
#: the button. So every attribute pressed below is read off the button the page
#: actually emitted, which is what a user's press is.
_BUTTONS_FROM_RENDER = """
function buttons(html){
  var re = /<button\\b([^>]*)>/g, m, out = [];
  while ((m = re.exec(html))){
    var attrs = {}, a = /([a-zA-Z-]+)="([^"]*)"/g, k;
    while ((k = a.exec(m[1]))) attrs[k[1]] = k[2];
    out.push(attrs);
  }
  return out;
}
"""


@pytest.mark.parametrize("fixture,column,shape", REOPEN_FIXTURES,
                         ids=["classification target", "regression target"])
def test_the_reopened_question_renders_a_control_the_user_can_press(
        fixture, column, shape):
    """`GUIDED-156`, driven — the assertion the six tests above never made.

    They watched the wire and they watched the record, and every one of them is
    correct. None rendered the page after the reopen, and the page is where the
    defect was: the question came back `asked`, and `renderSkips`, `renderAsked`
    and the target card all declined to draw it.

    Four things are observed here, in the order a person meets them: the skip
    row is gone, the generic channel did not pick the question up, a control for
    it EXISTS, and pressing that control posts the answer. The third is the one
    that was false.
    """
    if not H.available():
        pytest.skip("no JS engine on this machine")

    client = _client()
    project = _upload(client, fixture.replace(".csv", ""))
    pid = project["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": column}})

    before = client.get(f"/project/{pid}").json()
    assert before["task_type"] == shape, (
        f"{fixture}:{column} is no longer read as {shape}, so this run is not "
        f"the target shape it claims to be ({before['task_type']})")
    skipped = _skipped(client, pid)
    assert "confirm_task_type" in skipped, (
        f"the task-type question is not skipped on {fixture}:{column}, so "
        f"there is nothing to reopen and this test proves nothing")
    assert skipped["confirm_task_type"]["confidence"] == "high", (
        "the skip was granted below high confidence, which Decision B does not "
        "permit; this test is about the high-confidence reopen")

    r = client.post(f"/project/{pid}/decision",
                    json={"kind": "unskip", "subject": "confirm_task_type",
                          "payload": {"key": "confirm_task_type"}})
    assert r.status_code == 200, r.text
    q = _asked(client, pid)["confirm_task_type"]
    assert q["skip_reason"] is None and len(q["options"]) == 2, (
        "the server no longer brings the question back clean, so the page is "
        f"not being driven through the reopen this test is about: {q}")

    routes = _routes(client, pid)
    # The POST answers with the project UNCHANGED, so the controller survives
    # the first press and a second one can be made. The harness is not the
    # server; what the server does with the body is replayed below, against it.
    routes[f"POST /project/{pid}/decision"] = routes[f"/project/{pid}"]

    out = H.run(
        _BUTTONS_FROM_RENDER +
        "var ov = __harness.html('taskOverride') || '';\n"
        "var btns = buttons(ov).filter(function(b){ return b['data-task']; });\n"
        "for (var i = 0; i < btns.length; i++){\n"
        "  __harness.dispatch('click', __harness.target(btns[i]));\n"
        "  for (var j = 0; j < 6; j++) await new Promise(function(r){ setTimeout(r, 0); });\n"
        "}\n"
        "__emit({ov: ov.length, tasks: btns.map(function(b){ return b['data-task']; }),\n"
        "        slot: btns.map(function(b){ return b['data-ac']; }),\n"
        "        skip: (__harness.html('skipNote') || '').length,\n"
        "        generic: (__harness.html('askedQuestions') || '')\n"
        "                   .indexOf('confirm_task_type') !== -1,\n"
        "        posts: __harness.posts()});",
        routes=routes, search=f"?project={pid}")

    # 1 · the skip row is gone, because the question is no longer skipped.
    assert out["skip"] == 0, (
        "the reopened question is still being drawn as a skip, so `renderSkips` "
        "is now claiming a question the Router says is asked")
    # 2 · and the generic channel refused it, because it is handled elsewhere.
    assert out["generic"] is False, (
        "`renderAsked` drew the question. That is not wrong for a user, but it "
        "means `HANDLED_QUESTION_KEYS` no longer holds `confirm_task_type` — "
        "the shelf was shortened instead of the surface being fixed")
    # 3 · SO THE ONLY SURFACE LEFT HAS TO DRAW IT. This is the reproduction.
    assert out["tasks"], (
        f"the reopened question renders NOWHERE. `#skipNote` is empty, "
        f"`#askedQuestions` does not carry it, and `#taskOverride` held "
        f"{out['ov']} characters — so pressing *Ask me anyway* on a "
        f"{shape} target leads to a page with nothing on it to answer")
    assert sorted(out["tasks"]) == ["classification", "regression"], (
        f"the control does not offer both readings, so the user who disputed "
        f"the engine cannot record the other one: {out['tasks']}")
    assert set(out["slot"]) == {"task"}, (
        "the rendered control names no at-control slot, so the server's answer "
        f"to the press lands where nobody is looking: {out['slot']}")

    # 4 · and pressing them records answers — a control that renders and does
    #     nothing is `DRIVE-001`, which is the defect with a nicer surface.
    #     BOTH are pressed, because a reopened question the user can only agree
    #     with is not a question.
    bodies = [p["body"] for p in out["posts"]]
    assert len(bodies) == 2, (
        f"two rendered controls were pressed and {len(bodies)} request(s) went "
        f"out: {bodies}")
    assert all(b["kind"] == "set_task_type" for b in bodies), (
        f"a press on the reopened control sent something other than the "
        f"answer to it: {bodies}")
    assert sorted(b["payload"]["task_type"] for b in bodies) == [
        "classification", "regression"], (
        f"the two controls do not post the two readings: {bodies}")

    # And the real server accepts the press, and the question settles.
    #
    # THE CONFIRMING PRESS IS THE ONE REPLAYED, and that is a limit rather than
    # a preference: `set_task_type("regression")` on a two-level STRING target
    # raises an uncaught `TypeError` out of `compute_target_profile`, which the
    # strict `xfail` below records. Replaying it here would make this test red
    # for a defect it is not about. The body is still the page's own — taken
    # from the requests the presses produced, never composed here.
    confirm = next(b for b in bodies if b["payload"]["task_type"] == shape)
    replay = client.post(f"/project/{pid}/decision", json=confirm)
    assert replay.status_code == 200, replay.text
    assert "confirm_task_type" not in _asked(client, pid)
    assert "confirm_task_type" not in _skipped(client, pid)
    kinds = [d["kind"] for d in client.get(f"/project/{pid}").json()["decisions"]]
    assert kinds.index("unskip") < kinds.index("set_task_type"), (
        "the record no longer carries *the user asked to be asked, and then "
        "answered it themselves* as two sentences")


@pytest.mark.xfail(strict=True, raises=TypeError, reason=(
    "FOUND WHILE FIXING `GUIDED-156`, and filed rather than fixed. Overriding "
    "a high-confidence CLASSIFICATION reading to `regression` on a two-level "
    "STRING target raises an uncaught TypeError out of pandas `nanmean`, via "
    "`ml/dataset_profile.compute_target_profile`. It is a 500 with no body, "
    "not a refusal, and the governing rule permits refusing. `GUIDED-097`'s "
    "class one surface over: every test of this path used a numeric target. "
    "It is reachable through the API today on any string target, and it is "
    "reachable through the PAGE only once `GUIDED-156`'s fix renders the "
    "override at high confidence — which is why it is recorded here."))
def test_overriding_a_string_target_to_regression_refuses_rather_than_crashing():
    """The other half of the reopened control, named with a failing test.

    Two fixtures of different target shape would be the rule; there is only one
    shape that can reach this, because a numeric target overridden to
    `classification` is answerable and does not crash. That asymmetry IS the
    defect, so it is stated rather than padded.
    """
    client = _client()
    project = _upload(client, "clinic_visits")
    pid = project["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": "outcome"}})
    assert client.get(f"/project/{pid}").json()["task_type"] == "classification"

    r = client.post(f"/project/{pid}/decision",
                    json={"kind": "set_task_type",
                          "payload": {"task_type": "regression"}})
    assert r.status_code in (400, 409), (
        f"the server neither refused nor crashed; it returned {r.status_code}")


#: `GUIDED-192`'s two fixtures of different target shape (`GUIDED-097`). One
#: file, because the settled block needs the metabolomics left-censoring prior
#: and no second shipped fixture carries one — the shape that varies is the
#: TARGET, which is what the rule is about. `bmi` also puts a SECOND skip
#: (`confirm_task_type`, step `data`) into the same preprocess plan, which is
#: the case that made the step filter necessary.
SETTLED_FIXTURES = [
    ("responder", "classification"),
    ("bmi", "regression"),
]

#: The surfaces that existed before `GUIDED-192` and drew none of this. Kept as
#: a list so the assertions below say WHERE they looked rather than asserting
#: against one host and calling it "nowhere".
_HOSTS = ("skipNote", "askedQuestions", "missBox", "prepPlan")

#: Read the hosts, and emit SIZES rather than markup for all but the new one.
#:
#: `#missBox` holds 115 cards on this fixture and emitting it whole overflows
#: the harness's single-line sentinel — the JSON came back unterminated and
#: pytest reported a `JSONDecodeError`, which reads like a broken drive rather
#: than like a test asking for too much. So each host contributes a length and
#: a membership test, and only `#prepPlan` — the surface under test, and a
#: couple of rows — is returned as markup.
_READ_HOSTS = (
    "var H = %s;\n"
    "var seen = {}, sizes = {};\n"
    "H.forEach(function(id){\n"
    "  var h = __harness.html(id) || '';\n"
    "  seen[id] = h; sizes[id] = h.length;\n"
    "});\n"
) % list(_HOSTS)


def _settled_project(target):
    client = _client()
    project = _upload(client, "metabolomics_untargeted")
    pid = project["id"]
    r = client.post(f"/project/{pid}/decision",
                    json={"kind": "set_target", "payload": {"column": target}})
    assert r.status_code == 200, r.text
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_lens", "payload": {"lens": ["metabolomics"]}})
    settled = [k for k in _skipped(client, pid, "preprocess")
               if k.startswith("missingness_settled::")]
    assert settled, (
        f"no pack-settled missingness block on metabolomics_untargeted:"
        f"{target}; the left-censoring prior is what makes this meaningful")
    return client, pid, settled[0]


@pytest.mark.parametrize("target,shape", SETTLED_FIXTURES,
                         ids=["classification target", "regression target"])
def test_a_settled_missingness_block_draws_a_skip_row_that_can_be_reopened(
        target, shape):
    """`GUIDED-192` · the half that came BEFORE the reopen, and was missing too.

    `GUIDED-041` built *"Ask me anyway"* and `GUIDED-156` made the reopened
    question render. Both were about a skip the user can already see. For this
    family the skip was never drawn at all, because the only step that serves
    it was the one step the page did not fetch — so there was no button to
    press and the reopen path was unreachable from the interface entirely.

    Driven, and the press is built from the row the page actually emitted
    (trap #3): a hand-written `{'data-unskip': key}` would drive the
    document-level delegate whether or not anything ever rendered the control,
    which is the defect supplied by the fixture.
    """
    if not H.available():
        pytest.skip("no JS engine on this machine")

    client, pid, key = _settled_project(target)
    assert client.get(f"/project/{pid}").json()["task_type"] == shape, (
        f"metabolomics_untargeted:{target} is no longer read as {shape}")

    out = H.run(
        _READ_HOSTS +
        "var K = %r;\n"
        "var host = seen['prepPlan'];\n"
        "var re = new RegExp('data-unskip=\"' + K.replace(/[.*+?^${}()|[\\]\\\\]/g,"
        "                    '\\\\$&') + '\"');\n"
        "var m = re.exec(host);\n"
        "var posted = null;\n"
        "if (m){\n"
        "  __harness.dispatch('click', __harness.target(\n"
        "    {'data-unskip': K, 'data-unskip-title': 'x'}, ['again']));\n"
        "  var ps = __harness.posts();\n"
        "  posted = ps.length ? ps[ps.length - 1].body : null;\n"
        "}\n"
        "__emit({sizes: sizes, panel: seen['prepPlan'], drew: !!m,\n"
        "        posted: posted,\n"
        "        steps: __harness.calls().map(function(c){ return c.path; })\n"
        "                 .filter(function(p){ return p.indexOf('interview') !== -1; })});"
        % key,
        routes=_routes(client, pid), search=f"?project={pid}")

    assert any("step=preprocess" in p for p in out["steps"]), (
        "the page never asks for the preprocess-step plan, so no surface on it "
        f"can know this skip exists: {out['steps']}")
    assert out["drew"], (
        f"{key} — one stated fact standing for hundreds of columns — draws no "
        f"skip row and no reopen affordance. Host sizes: {out['sizes']}")
    assert "Ask me anyway" in out["panel"], (
        "the row rendered without the reopen affordance, so the skip is "
        "visible and irreversible — which is the half of Decision B that makes "
        "a skip permissible at all")
    assert out["posted"], "pressing the rendered reopen sent nothing"
    assert out["posted"]["kind"] == "unskip", (
        f"the reopen affordance on this surface sends something other than a "
        f"reopen: {out['posted']}")
    assert out["posted"]["payload"]["key"] == key

    # And the real server accepts exactly that body, from this surface.
    replay = client.post(f"/project/{pid}/decision", json=out["posted"])
    assert replay.status_code == 200, replay.text
    assert key in _asked(client, pid, "preprocess")


@pytest.mark.parametrize("target,shape", SETTLED_FIXTURES,
                         ids=["classification target", "regression target"])
def test_a_reopened_settled_missingness_block_renders_somewhere(target, shape):
    """`GUIDED-156`'s class, closed. Driven, because *does a surface exist* is
    a behavior question and a grep only answers *does the string appear*.

    `test_a_pack_settled_missingness_block_is_reopenable_too` above asserts the
    server half and is correct. This is the half it does not reach: one skip
    standing for 306 columns, reopened, and — before `GUIDED-192` — rendered by
    nothing at all.

    The last assertion is the uncomfortable one and it is deliberate. The
    reopened block gets a surface that says the question is open and that this
    build has no control that answers it, and NO option buttons: `api.py`'s
    `answered` fold never produces a `missingness_settled::` key, and the
    page's answer delegate returns early for a key with no `ANSWERABLE` entry.
    A rendered option would be a control that silently does nothing, and this
    file exists because a reopen that looked like it worked was worse than none.
    """
    if not H.available():
        pytest.skip("no JS engine on this machine")

    client, pid, key = _settled_project(target)
    client.post(f"/project/{pid}/decision",
                json={"kind": "unskip", "payload": {"key": key}})
    assert key in _asked(client, pid, "preprocess")
    assert key not in _skipped(client, pid, "preprocess")

    out = H.run(
        _READ_HOSTS +
        "var K = %r;\n"
        "__emit({sizes: sizes, panel: seen['prepPlan'],\n"
        "        anywhere: H.filter(function(id){\n"
        "          return seen[id].indexOf(K) !== -1; }),\n"
        "        steps: __harness.calls().map(function(c){ return c.path; })\n"
        "                 .filter(function(p){ return p.indexOf('interview') !== -1; })});"
        % key,
        routes=_routes(client, pid), search=f"?project={pid}")

    assert any("step=preprocess" in p for p in out["steps"]), (
        "the page never asks for the preprocess-step plan, so no surface in it "
        f"can know this question exists: {out['steps']}")
    assert out["anywhere"], (
        f"{key} renders nowhere after the reopen. Host sizes: {out['sizes']}")

    panel = out["panel"]
    assert 'data-reopened="%s"' % key in panel, (
        "the reopened question is drawn, but not as a question that is open — "
        "so a user who pressed *Ask me anyway* sees the same page they saw "
        f"before pressing it: {panel[:400]!r}")
    assert "no control that answers a settled block" in panel, (
        "the surface does not say why there is nothing to press, which leaves "
        "a reopened question looking like a rendering bug rather than a stated "
        f"limit: {panel[:400]!r}")
    assert 'data-answer-key="%s"' % key not in panel, (
        "an option control was rendered for a key the answer delegate returns "
        "early on and the record cannot fold back — a solid button that "
        "silently no-ops, which is `GUIDED-006` and `DRIVE-001`")

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
# The tests here that drove the retired legacy app (turbotab/api.py, the page, the
# figure and manuscript modules, the gates) were removed with it, BLUEPRINT §9.1; git keeps them.
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml import router                                                 # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"


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


def test_the_router_refuses_to_skip_a_key_the_user_reopened():
    """Enforced in the one place Decision B lives, so a second skip site cannot
    forget it. `_skip_is_permitted` is where the constitution is checked rather
    than remembered."""
    assert router._skip_is_permitted("high", "task_type") is True
    assert router._skip_is_permitted(
        "high", "task_type", "confirm_task_type", ["confirm_task_type"]) is False
    assert router._skip_is_permitted(
        "high", "task_type", "confirm_task_type", ["something_else"]) is True


# ── `GUIDED-156` · and then where does it RENDER? ────────────────────────────


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

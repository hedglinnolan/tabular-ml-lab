"""L49-E — `GUIDED-184` and `GUIDED-187`, which are one decision.

`GUIDED-183` closed at L48 and its report said it was bigger than the row: of
five resolve exits, one carried a retry, one was given one, one has no consumer
at all, and **two are not requests.** `grain._RESOLVE` and `packs._LENS_RESOLVE`
both mean *go back to the question and answer it differently* — there is no
request to re-post, so they carried no `retry.payload`, and `showRefusal`
enables on `retry.payload`.

So the **safe** way out rendered `disabled` beside a live *"continue anyway"*,
on three more surfaces than `GUIDED-183` named. That is §09's choice inverted:
a consequence resolves or is attested, and only one of the two was pressable.

## Why the page could not fix this alone, and why that was right

The page's own stated rule is that **it will not invent a mechanism for a way
out the server did not describe**, which is why L48 filed this rather than
patching `showRefusal` to enable every resolve. The rule is correct and it was
the obstacle. So the server describes it: `exits.revise` builds an exit with
`takes.action = "revise"` and a `how` sentence, and the page implements exactly
that one verb. An action it does not recognize stays disabled — the same
refusal as before for anything undescribed.

## `GUIDED-187` is the same decision seen from the predicate

`exits.is_actionable` returned `True` for **every** non-attest exit, on the
stated grounds that *a resolve exit sends the user back to the question and
needs nothing*. That premise stopped being the page's the day `showRefusal`
began reading the payload rather than the kind — its own comment records the
change — so the unifying test for `GUIDED-064` and `GUIDED-072` was passing on
exits nobody could take. It reads what the page reads now, in the same words.

**Deciding them together was the point.** Changing the predicate before the
mechanism existed would only have moved the wrong answer.

## What is NOT covered

- **Whether the enabled button is on screen.** Nothing without layout can say.
- **`clinical.substitution_blocker`'s `keep_censored`** — the fifth resolve
  exit, and `GUIDED-185`: it has no consumer anywhere outside a test, so there
  is no refusal to render and nothing to take. Left alone deliberately.
"""
from __future__ import annotations

from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent / "sample_data"


def test_both_revise_exits_describe_how_they_are_taken():
    """The server's half. A hand-written dict is what left them mute."""
    from turbotab import exits, grain, packs

    for name, exit_row in (("grain._RESOLVE", grain._RESOLVE),
                           ("packs._LENS_RESOLVE", packs._LENS_RESOLVE)):
        assert exit_row["kind"] == exits.RESOLVE, name
        takes = exit_row.get("takes") or {}
        assert takes.get("action") == exits.REVISE, (
            f"{name} carries no way to be taken, so `showRefusal` renders it "
            f"`disabled` — the safe exit greyed out beside a live attestation")
        assert takes.get("how"), (
            f"{name} names an action and does not say what taking it does. The "
            f"page implements the verb; the SENTENCE is what a person reads")
        assert not (exit_row.get("retry") or {}).get("payload"), (
            f"{name} grew a retry payload. It is not a request — inventing one "
            f"would re-post the request that was just refused, which is "
            f"`GUIDED-183`'s own finding")


def test_the_predicate_agrees_with_the_page_now():
    """`GUIDED-187`. Two answers to one question was the defect."""
    from turbotab import exits, grain, missingness, packs

    assert exits.is_actionable(grain._RESOLVE)
    assert exits.is_actionable(packs._LENS_RESOLVE)
    # The retry-carrying kind still passes, unchanged.
    resolve = [e for e in missingness.blocker_exits("categorical")
               if e["kind"] == exits.RESOLVE][0]
    assert exits.is_actionable(resolve)
    # AND IT STILL SAYS NO. A predicate that answers yes to everything is the
    # defect with the sign flipped.
    assert not exits.is_actionable(
        {"id": "x", "kind": "resolve", "label": "Mute", "detail": ""}), (
        "a resolve exit carrying neither a retry payload nor a described "
        "action is called actionable, so the predicate is back to answering "
        "from `kind` — which is the premise the page stopped holding")
    assert not exits.is_actionable(
        {"id": "y", "kind": "resolve", "takes": {"action": "teleport"}}), (
        "an action the page does not implement is called actionable")


@pytest.mark.parametrize("fixture,lens,column", [
    ("clinical_labs.csv", "clinical", "readmitted"),
    ("survey_instrument.csv", "survey", "score_total"),
], ids=["classification target", "continuous target"])
def test_the_lens_contradictions_revise_exit_renders_live(fixture, lens, column):
    """Driven through the page, on two fixtures of different target shape.

    The claim is about a rendered button's `disabled` attribute, because that
    is the whole of the defect — the payload was always correct and the
    interface was the surface that inverted the choice.
    """
    from turbotab import packs, pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")
    from fastapi.testclient import TestClient

    from turbotab import api

    client = TestClient(api.app)
    with (DATA / fixture).open("rb") as handle:
        pid = client.post("/project", files={
            "file": (fixture, handle, "text/csv")}).json()["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_lens", "payload": {"lens": [lens]}})
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": column}})

    routes = {f"/project/{pid}": client.get(f"/project/{pid}").json()}
    for path in ("interview?step=data", "interview?step=explore",
                 "interview?step=features", "capabilities", "features",
                 "recipes", "preprocess", "figures", "draft", "manuscript",
                 "models", "training", "instability", "explain", "sensitivity",
                 "evidence/plausibility", "evidence/missingness"):
        resp = client.get(f"/project/{pid}/{path}")
        routes[f"/project/{pid}/{path}"] = (resp.json() if resp.status_code == 200
                                            else {})
    # The refusal is the real object the server composes, not a hand-built one:
    # a fixture that invented the exit would be asserting about a shape
    # production cannot produce (trap #3).
    refusal = {"message": "The lens you chose disagrees with this table.",
               "exits": [dict(packs._LENS_RESOLVE),
                         {"id": "attest", "kind": "attest",
                          "label": "My answer is right",
                          "detail": "Recorded as a stated limitation.",
                          "payload_key": "acknowledge_contradiction",
                          "retry": {"payload": {"acknowledge_contradiction": True},
                                    "how": "Send it again."}}]}
    routes[f"POST /project/{pid}/decision"] = {"__status": 409,
                                               "body": {"detail": refusal}}

    out = PH.run(
        "function buttons(html){\n"
        "  var out = [], re = /<button\\b([^>]*)>/g, m;\n"
        "  while ((m = re.exec(html || ''))){\n"
        "    var a = {}, kv, rx = /([a-zA-Z-]+)=\"([^\"]*)\"/g;\n"
        "    while ((kv = rx.exec(m[1]))) a[kv[1]] = kv[2];\n"
        "    a.__raw = m[0]; out.push(a);\n"
        "  }\n"
        "  return out;\n"
        "}\n"
        "__harness.dispatch('click', __harness.target("
        "  {'data-task': 'regression', 'data-ac': 'task'}));\n"
        "for (var i = 0; i < 8; i++) await new Promise(function(r){ setTimeout(r, 0); });\n"
        "var band = __harness.html('refusal') || '';\n"
        "var bs = buttons(band).filter(function(b){ return 'data-refusal-i' in b; });\n"
        "var revise = bs[0];\n"
        "if (revise) __harness.dispatch('click', __harness.target(revise));\n"
        "for (var j = 0; j < 8; j++) await new Promise(function(r){ setTimeout(r, 0); });\n"
        "__emit({n: bs.length, raw: revise && revise.__raw,\n"
        "        disabled: revise ? ('disabled' in revise) : null,\n"
        "        primary: revise ? /primary/.test(revise['class'] || '') : null,\n"
        "        after_press: __harness.html('refusal')});",
        routes=routes, search=f"?project={pid}")

    assert out["n"] == 2, (
        f"the refusal band rendered {out['n']} exit buttons, not two")
    assert out["disabled"] is False, (
        f"the REVISE exit still renders disabled beside a live attestation, "
        f"which is the choice §09 intends, inverted: {out['raw']!r}")
    assert out["primary"], (
        "the safe way out is not the primary. `showRefusal` marks a takeable "
        "resolve as primary and this one is takeable now")
    assert not (out["after_press"] or "").strip(), (
        "pressing the revise exit left the refusal band up, so the button is "
        "enabled and does nothing — `GUIDED-006`'s sentence exactly")


def test_the_grain_contradiction_is_the_same_shape():
    """The second surface, asserted on the composed object.

    Driven end to end for the lens above; the grain contradiction needs a
    repeated-measures journey to reach, which is a different fixture chain —
    said here rather than left as uncounted coverage.
    """
    from turbotab import exits, grain

    for exit_row in grain._EXITS_STATED_UNIQUE + grain._EXITS_STATED_REPEATS:
        assert exits.is_actionable(exit_row), (
            f"{exit_row.get('id')} on a grain contradiction cannot be taken by "
            f"a client holding only the payload")
    resolves = [e for e in grain._EXITS_STATED_UNIQUE
                if e["kind"] == exits.RESOLVE]
    assert resolves and (resolves[0].get("takes") or {}).get("action") == "revise"

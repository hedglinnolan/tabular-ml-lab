"""L49-A1 — `TEST-048`. The instrument stopped denying, and it was denying twice.

`pageharness.py`'s own docstring holds the rule this file enforces: **dumb is
allowed, lying is not.** `document.querySelector` returned `null`
unconditionally and `querySelectorAll` returned `[]` — which is not ignorance,
it is an ANSWER, and it was *there is no such element* about elements that are
in `index.html`.

**What that cost, measured rather than asserted.** `setMap` addressed its eight
dots with `document.querySelector('.map-step[data-map=…]')`, so it was a total
no-op under every drive this project has ever run. Six steps could wear `now`
at once for the whole life of the analysis map and no test could see it —
`GUIDED-159`, found by a person looking at a screen, two loops after the
instrument that should have caught it shipped.

## The second defect, which the first one hid

`matches` evaluated as it parsed and `break`-ed on the first token that did not
match — leaving the rest of the token unconsumed, so its own
*"selector token not understood"* guard then fired on a selector it understands
perfectly. It only misfires on a MULTI-TOKEN selector against a NON-MATCHING
element, which is every element but one in any real search — so it was
unreachable while `querySelector` never called it, and **repairing
`querySelector` alone would have turned a silent wrong answer into a crash.**
The blind spot had a blind spot.

## What is modeled now, and what still is not

Three populations, which together are everything the controller addresses:

1. **Real children**, appended through `createElement` + `appendChild`.
2. **Elements declared in assigned markup** — `innerHTML = '<button data-x=…>'`
   creates real nodes in a browser; this shim does not parse HTML, so it does
   the same smallest-true-thing `__declareMarkupIds` already did for ids and
   notes the elements with their attributes.
3. **The static body**, from `body_elements()` — 220 elements, read out of the
   same markup `seed_classes()` reads. Nothing modeled it before, and it is
   where the analysis map lives.

**NOT modeled, and `matches` throws rather than guessing:** descendant
selectors (`a b`), `:nth-child`, sibling combinators. A declared node is flat —
it carries its own attributes and classes and does not know its position. A
throw is the right answer where `null` was the wrong one.
"""
from __future__ import annotations

import pytest

from turbotab import pageharness as PH

pytestmark = pytest.mark.skipif(not PH.available(),
                                reason="no JS engine on this machine")


def test_the_static_body_is_modeled_at_all():
    """The positive control, before anything is asserted about a selector.

    A selector engine over an empty population answers `null` to everything and
    looks exactly like the defect it replaced.
    """
    elements = PH.body_elements()
    assert len(elements) > 150, (
        f"only {len(elements)} elements were read out of the static body, so "
        f"the document's own population is nearly empty and every `null` below "
        f"would be vacuous")
    tags = {e["tag"] for e in elements}
    assert {"button", "div", "section"} <= tags, sorted(tags)
    assert any(e["attrs"].get("data-map") == "train" for e in elements), (
        "the analysis map's Train dot is not in the modeled body, and it is "
        "the element this whole repair exists for")


def test_the_selector_finds_the_dots_that_made_this_necessary():
    """`GUIDED-159`'s eight dots, which were unobservable for the map's whole life."""
    out = PH.run(
        "var all = document.querySelectorAll('.map-step[data-map]');\n"
        "var one = document.querySelector('.map-step[data-map=\"train\"]');\n"
        "__emit({n: all.length, tag: one && one.tagName,\n"
        "        state: one && one.getAttribute('data-map-state'),\n"
        "        same_object_as_by_id: one === document.getElementById('map-train')});")
    assert out["n"] == 8, (
        f"the rail declares eight steps and the selector found {out['n']}")
    assert out["tag"] == "BUTTON", (
        f"the dot came back as a {out['tag']}, so the node is a stand-in rather "
        f"than the element the document declares")
    assert out["state"] == "waiting", (
        "the dot does not carry the state the markup gives it, so a claim "
        "about what the map says would be reading an invented node")
    assert out["same_object_as_by_id"], (
        "`querySelector` and `getElementById` return two different objects for "
        "one element, so a write through either is invisible to the other — "
        "which is the two-readers-of-one-property defect `GUIDED-077` was about")


def test_it_still_says_no_when_the_answer_is_no():
    """The negative control. A repair that answers YES to everything is worse.

    This is the assertion that makes every `null` in this repository mean
    something again, so it is asserted rather than assumed.
    """
    out = PH.run(
        "__emit({absent: document.querySelector('[data-there-is-no-such-thing]'),\n"
        "        absent_all: document.querySelectorAll('[data-there-is-no-such-thing]').length,\n"
        "        wrong_value: document.querySelector('[data-map=\"not-a-step\"]'),\n"
        "        wrong_class: document.querySelector('.no-such-class[data-map]')});")
    assert out["absent"] is None
    assert out["absent_all"] == 0
    assert out["wrong_value"] is None, (
        "an attribute selector with a value the document does not carry matched "
        "something, so the engine is ignoring the value half")
    assert out["wrong_class"] is None


def test_a_multi_token_selector_does_not_throw_on_a_non_match():
    """The second defect, pinned. `matches` broke before consuming its token.

    Driven against an element that fails the FIRST token — the case that
    unconsumed the rest and tripped the parser's own guard.
    """
    out = PH.run(
        "var el = __harness.target({'data-map': 'train'});\n"
        "__emit({matched: matches(el, '.map-step[data-map=\"train\"]'),\n"
        "        one_token: matches(el, '[data-map]'),\n"
        "        n: document.querySelectorAll('[data-map]').length});")
    assert out["matched"] is False, (
        "an element carrying `data-map` and NOT `.map-step` matched a selector "
        "requiring both")
    assert out["one_token"] is True
    assert out["n"] >= 8, (
        "the search ran over the whole population without throwing, which is "
        "what the break-before-consume bug prevented")


def test_a_selector_it_cannot_answer_throws_rather_than_saying_no():
    """The shape of the honesty. `null` is a claim; a throw is an abstention.

    A declared node is flat, so a descendant selector is not answerable — and
    the one thing this instrument may not do is answer it anyway.
    """
    out = PH.run(
        "var threw = false;\n"
        "try { document.querySelector('.map-step .md'); }\n"
        "catch (e) { threw = /not understood/.test(String(e)); }\n"
        "__emit({threw: threw});")
    assert out["threw"], (
        "a descendant selector came back as an answer instead of a refusal. "
        "This shim models flat nodes with no position, so `null` there would "
        "be the same lie one shape over")


def test_a_node_declared_by_assigned_markup_is_findable_and_writable():
    """Population 2, which is where the per-row panels live.

    `data-rg-body`, `data-offer-pv` and `data-plaus-reason` are all addressed
    this way by the real controller, and all three were unreachable.
    """
    out = PH.run(
        "var host = document.getElementById('missBox');\n"
        "host.innerHTML = '<div data-rg-body=\"recode\"></div>"
        "<input data-plaus-reason=\"sbp\">';\n"
        "var body = document.querySelector('[data-rg-body=\"recode\"]');\n"
        "var box = document.querySelector('[data-plaus-reason=\"sbp\"]');\n"
        "if (body) body.innerHTML = 'a panel';\n"
        "var after_repaint;\n"
        "__emit({found_body: !!body, found_input: !!box,\n"
        "        tag: box && box.tagName, wrote: body && body.innerHTML,\n"
        "        gone_after_repaint: (function(){\n"
        "          host.innerHTML = '';\n"
        "          return document.querySelector('[data-rg-body=\"recode\"]');\n"
        "        })()});")
    assert out["found_body"] and out["found_input"], (
        "a node declared by assigned markup is still unfindable, so the "
        "per-row panels the controller writes into remain invisible")
    assert out["tag"] == "INPUT", out["tag"]
    assert out["wrote"] == "a panel"
    assert out["gone_after_repaint"] is None, (
        "reassigning the host's markup left the old declared node findable, so "
        "a repaint would read as a mutation — the distinction §05 turns on")

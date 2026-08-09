"""`TEST-066` — the harness's DOM moves a node instead of copying it.

## Why a harness defect is filed `high` and fixed as a part

The standing risk here is the INVERSE of the usual one. Normally a test tool is
too permissive and lets a defect through. This one was too **poor**, so it
pushed production code toward the subset it implemented — a tool dictating the
product. `turbotab/web/index.html`'s deck clears and refills because
`replaceChildren` was absent and `appendChild` duplicated, and both facts are
written in its comments as constraints of the instrument. The deck's rewrite is
defensible on its own terms. The next one might not have been.

## What was wrong, measured rather than inferred

Four things, and the fourth was found by fixing the first three:

1. `typeof element.insertBefore` was `"undefined"`.
2. `typeof element.replaceChildren` was `"undefined"`.
3. `"hidden" in element` was `false`.
4. `appendChild` on an attached node appended a **duplicate** — three cards
   re-appended in reverse produced six. Move semantics are the whole mechanism
   behind identity-preserving reorder, so the instrument could see that a card
   SURVIVED a re-render and could not see that it MOVED.

And then, once `appendChild` moved: **`__unregister` was clearing `_parent` down
the entire subtree**, which no DOM does — a card taken out of the deck still
contains its rows. Nothing could observe that while `appendChild` copied,
because nothing read `_parent` at all. That is `TEST-069`, and it is why fixing
the copying bug did NOT stop the deck's rows doubling on a re-render: measured
at 4 rows per card becoming 8, identically before and after the first fix, and
12 rows in the deck holding at 12 once both were fixed.

## The claim this file exists to make

Every assertion here is about the INSTRUMENT, driven through the same shim every
page test runs on. A harness bug is a claim about every test that uses it, so
the shim is checked directly rather than through a page whose own defects would
be mixed in.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import pageharness as PH                              # noqa: E402


def _run(body):
    if not PH.available():
        pytest.skip("no JS engine on this machine")
    return PH.run(body)


def test_appending_an_attached_node_moves_it_rather_than_copying_it():
    """`TEST-066`'s headline, as the count that gave it away.

    Three nodes re-appended in reverse gave six. They give three, in reverse.
    """
    out = _run("""
      var host = document.createElement('div');
      var made = [];
      for (var i = 0; i < 3; i++){
        var n = document.createElement('span');
        n.className = 'c'; n.dataset.k = String(i);
        host.appendChild(n); made.push(n);
      }
      var before = host.querySelectorAll('.c').map(function(e){ return e.dataset.k; });
      made.slice().reverse().forEach(function(n){ host.appendChild(n); });
      var after = host.querySelectorAll('.c').map(function(e){ return e.dataset.k; });
      __emit({before: before, after: after,
              same: made.every(function(n, i){ return host.children[2 - i] === n; })});
    """)
    assert out["before"] == ["0", "1", "2"]
    assert out["after"] == ["2", "1", "0"], (
        f"a bare re-append did not reorder: {out['before']} -> {out['after']}. "
        f"If the count grew, the shim is copying again and every identity "
        f"claim written against it is unverifiable.")
    assert len(out["after"]) == 3, (
        f"re-appending three attached nodes produced {len(out['after'])} of "
        f"them. TEST-066: appendChild must MOVE.")
    assert out["same"], (
        "the reordered children are not the same objects, so the shim rebuilt "
        "rather than moved — which is the distinction the whole of "
        "DESIGN_LANGUAGE §05.2 turns on")


def test_a_node_moved_to_another_parent_leaves_the_first_one():
    """The other half of a move, and the half a naive fix forgets."""
    out = _run("""
      var a = document.createElement('div'), b = document.createElement('div');
      var n = document.createElement('i'); n.id = 'moved';
      a.appendChild(n);
      var beforeA = a.children.length, beforeB = b.children.length;
      b.appendChild(n);
      __emit({beforeA: beforeA, beforeB: beforeB,
              afterA: a.children.length, afterB: b.children.length,
              stillFindable: !!document.getElementById('moved')});
    """)
    assert (out["beforeA"], out["beforeB"]) == (1, 0)
    assert (out["afterA"], out["afterB"]) == (0, 1), (
        f"moving a node left it in both parents: {out}")
    assert out["stillFindable"], (
        "a MOVED node stopped being findable by id. A move is one operation, "
        "not a removal followed by an insertion — routing it through "
        "removeChild would unregister the id, and 'did this leave the tree?' "
        "must not be momentarily true of a node that never left it")


def test_removing_a_node_does_not_orphan_the_nodes_inside_it():
    """`TEST-069`. The defect the first fix uncovered.

    A card taken out of the deck still contains its rows. The shim used to
    clear `_parent` down the whole subtree, so after a detach every row inside
    had no parent to be moved out of — and the next append duplicated it.
    """
    out = _run("""
      var host = document.createElement('div');
      var card = document.createElement('article');
      var body = card.appendChild(document.createElement('div'));
      var row = body.appendChild(document.createElement('p'));
      host.appendChild(card);
      host.removeChild(card);
      /* the row is still inside the body, so re-appending it must MOVE it */
      body.appendChild(row);
      __emit({rows: body.children.length,
              same: body.children[0] === row});
    """)
    assert out["rows"] == 1, (
        f"re-appending a row inside a detached card produced {out['rows']} of "
        f"them. Removing the card orphaned everything under it, so the row had "
        f"no parent to be detached from. TEST-069.")
    assert out["same"]


def test_insert_before_puts_the_node_before_the_reference():
    """`TEST-066`'s second API. A null reference appends, as in a browser."""
    out = _run("""
      var box = document.createElement('div');
      var x = document.createElement('i'); x.dataset.k = 'x';
      var y = document.createElement('i'); y.dataset.k = 'y';
      var z = document.createElement('i'); z.dataset.k = 'z';
      box.appendChild(x);
      box.insertBefore(y, x);
      box.insertBefore(z, null);
      var moved = document.createElement('div');
      moved.appendChild(x);
      __emit({order: box.children.map(function(e){ return e.dataset.k; }),
              afterMove: box.children.map(function(e){ return e.dataset.k; }),
              movedCount: moved.children.length});
    """)
    assert out["order"] == ["y", "z"], (
        f"insertBefore did not order the nodes: {out['order']}. Expected y "
        f"before z once x was moved away, and a null reference to append.")
    assert out["movedCount"] == 1


def test_replace_children_leaves_nothing_of_what_was_there():
    """Replaces, so the old children are gone from `getElementById` with them.

    The ASSIGNED markup goes too: in a browser this method leaves no previous
    content of any kind, and a shim that kept `_html` would report a container
    as still holding what was wiped.
    """
    out = _run("""
      var box = document.createElement('div');
      var old = document.createElement('span'); old.id = 'old';
      box.appendChild(old);
      box.innerHTML = box.innerHTML + '<b id="declared">assigned</b>';
      var hadDeclared = !!document.getElementById('declared');
      var fresh = document.createElement('u'); fresh.id = 'fresh';
      box.replaceChildren(fresh);
      __emit({children: box.children.length,
              hadDeclared: hadDeclared,
              oldGone: !document.getElementById('old'),
              declaredGone: !document.getElementById('declared'),
              freshFindable: !!document.getElementById('fresh'),
              html: box.innerHTML});
    """)
    assert out["hadDeclared"], (
        "the assigned markup never declared its id, so 'it is gone afterwards' "
        "would be true for the wrong reason")
    assert out["children"] == 1
    assert out["oldGone"], "a replaced child stayed findable by id"
    assert out["declaredGone"], "the replaced markup's declared id survived"
    assert out["freshFindable"]
    assert "assigned" not in out["html"], (
        f"replaceChildren left the previously ASSIGNED markup behind: "
        f"{out['html']!r}")


def test_hidden_is_a_reflected_attribute_in_both_directions():
    """As a plain field it would accept a write no other reader could see.

    That is `GUIDED-081`'s defect — `className` — one property over: every
    assertion about a hidden surface would have come back vacuously true.
    """
    out = _run("""
      var el = document.createElement('div');
      var present = ('hidden' in el);
      var before = el.hidden;
      el.hidden = true;
      var on = el.hidden, serialized = el.__deep();
      el.hidden = false;
      __emit({present: present, before: before, on: on, off: el.hidden,
              serialized: serialized, afterOff: el.__deep()});
    """)
    assert out["present"], "`hidden` is not a property of an element at all"
    assert out["before"] is False and out["on"] is True and out["off"] is False
    assert "hidden" in out["serialized"], (
        f"setting `hidden` did not reach the serializer, so a test reading the "
        f"rendered region cannot see it: {out['serialized']!r}")
    assert "hidden" not in out["afterOff"], (
        f"clearing `hidden` left the attribute behind: {out['afterOff']!r}")


def test_the_four_apis_the_deck_was_written_around_are_all_present():
    """The row's own list, asserted as a list.

    `TEST-066` names four things by name. A file that fixes three of them and
    reads as done is the shape this project keeps meeting, so the row's list is
    checked rather than trusted.
    """
    out = _run("""
      var el = document.createElement('div');
      __emit({insertBefore: typeof el.insertBefore,
              replaceChildren: typeof el.replaceChildren,
              hidden: ('hidden' in el),
              appendChild: typeof el.appendChild});
    """)
    assert out == {"insertBefore": "function", "replaceChildren": "function",
                   "hidden": True, "appendChild": "function"}, (
        f"TEST-066 names four capabilities and the shim has {out}")


def test_parent_node_is_still_absent_and_that_is_recorded_not_forgotten():
    """`TEST-070`, stated as a test so it cannot be quietly assumed present.

    `parentNode` is NOT added by this loop. `turbotab/web/index.html` guards two
    removal branches with `gone.parentNode`, and those branches are dead under
    this shim — so adding the property would start EXECUTING code that has never
    run in a test, changing what several figure and disclosure tests observe.
    That is a separate change with its own measurement, and a harness that
    silently grew it inside a loop about move semantics would have made every
    one of those tests mean something new without saying so.
    """
    out = _run("""
      var host = document.createElement('div');
      var kid = host.appendChild(document.createElement('i'));
      __emit({parentNode: kid.parentNode === undefined ? 'undefined'
                          : String(kid.parentNode)});
    """)
    assert out["parentNode"] == "undefined", (
        "`parentNode` is now defined on the shim. That is not wrong — it is "
        "UNMEASURED: index.html's `gone.parentNode` removal branches become "
        "live, and the figure and disclosure tests that read those regions "
        "must be re-run and re-read before this test is deleted. TEST-070.")

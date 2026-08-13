"""`TEST-081` — `ml/router.py` said this file existed for three loops.

The comment above `SEQUENCE` reads:

> `test_the_marker_is_the_constitutional_position` asserts the two agree so the
> document cannot drift from the interface silently.

**Nothing of that name was anywhere in the tree.** The only occurrence of the
string was the comment itself. That is `TEST-077`'s class for the third time —
a claim in a comment standing in for a mechanism — and it is not academic: the
drift it names is exactly what `DRIVE-020`'s numbering half was, a card
rendering `01` where the Router serves `02`, and two human drives read it before
anyone noticed.

## What it checks, in both directions

`ml/router.py`'s `SEQUENCE` maps a question key to the position the page
renders. `OPENING_SEQUENCE.md` §01 is the constitution's own table of the same
thing. The failure this prevents has two shapes and the test has to have both:

* **a key with no documented position** — the code asks something the
  constitution does not describe;
* **a documented position with no key** — the constitution describes a question
  the Router cannot serve.

## What it deliberately does not check

**The ORDER of the sequence, and the fires-when column.** `SEQUENCE` is a
map from key to marker and carries neither; asserting an ordering against it
would be asserting a property the object does not have. `DRIVE-020`'s ordering
half is a separate open row and a router change.

**The fractional positions are the point, not an accident.** `state_orientation`
is `1.5` and `state_purpose` is `2.5` because both were inserted between
positions that were already cited in the constitution, the fixtures and three
documents. A test that normalized them to integers would be re-deriving the
table rather than checking it.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "turbotab" / "OPENING_SEQUENCE.md"

#: The one mapping in the document that is not a question: the seal has a row in
#: §01's table and no Router key, because it is a decision rather than a
#: question. Declared here with its reason rather than filtered silently.
NOT_A_QUESTION = {"SEAL"}

#: Keys the Router serves that §01 deliberately does not number — a pack's added
#: question and a Features-step question are not steps of the pre-seal
#: agreement. `SEQUENCE` already omits them; this is the same statement from the
#: other side, so a key added to `SEQUENCE` without a documented position fails.
_SEQ_START = "SEQUENCE: Dict[str, str] = {"


def _sequence() -> Dict[str, str]:
    """`ml/router.py`'s table, read out of the source.

    Read rather than imported so the test names the file the comment names, and
    so a `SEQUENCE` that stopped being a literal fails here loudly instead of
    being silently re-derived by an import.
    """
    text = (ROOT / "ml" / "router.py").read_text(encoding="utf-8")
    start = text.index(_SEQ_START)
    body = text[start:text.index("\n}", start)]
    return dict(re.findall(r'"([a-z_]+)":\s*"([^"]+)"', body))


def _documented() -> Dict[str, str]:
    """§01's table: position → the question's bolded text.

    Rows whose position is an em dash are not questions — the diagnosis pass and
    the seal — and are skipped by the same rule the document uses to write them.
    """
    text = DOC.read_text(encoding="utf-8")
    start = text.index("## 01 · The sequence")
    table = text[start:text.index("\n\n", text.index("| — | **SEAL**", start))]
    out: Dict[str, str] = {}
    for line in table.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2 or cells[0] in ("#", "---", "—"):
            continue
        position, question = cells[0], cells[1]
        if not re.fullmatch(r"\d+(?:\.\d+)?", position):
            continue
        out[position] = question
    return out


def test_the_document_and_the_router_agree_on_every_position(capsys):
    """The check the comment promised, both directions at once."""
    sequence = _sequence()
    documented = _documented()

    # THE POSITIVE CONTROL. Both readers parse prose, and a reader that found
    # nothing would make every assertion below true of an empty set.
    assert len(sequence) >= 10, (
        f"only {len(sequence)} keys read out of SEQUENCE; the reader is broken "
        f"and this test would pass over anything")
    assert len(documented) >= 10, (
        f"only {len(documented)} positions read out of {DOC.name} §01; the "
        f"reader is broken")

    # COMPARED AS POSITIONS, NOT AS PRESENTATION. `SEQUENCE` zero-pads because
    # the marker is rendered in a fixed-width slot — `01`, `02` — and §01's
    # table writes bare integers. A string comparison would report every key as
    # undocumented, which is what the first run of this test did: it was
    # measuring the padding.
    def _pos(value: str) -> float:
        return float(value)

    served = {_pos(v) for v in sequence.values()}
    written = {_pos(p) for p in documented}

    undocumented = sorted(k for k, v in sequence.items() if _pos(v) not in written)
    assert not undocumented, (
        f"the Router serves these at positions {DOC.name} §01 does not "
        f"describe: "
        + ", ".join(f"{k}={sequence[k]!r}" for k in undocumented)
        + ". Either the document gained a step it did not record, or a key was "
          "given a marker nobody agreed to.")

    unserved = sorted(written - served)
    assert not unserved, (
        f"{DOC.name} §01 describes questions at positions "
        f"{[('%g' % p) for p in unserved]} that no "
        f"Router key claims. A position in the constitution with nothing to "
        f"serve it is a step the app cannot ask.")

    with capsys.disabled():
        print(f"\n  {len(sequence)} keys · {len(documented)} documented "
              f"positions · both directions clean")
        for key, position in sorted(sequence.items(), key=lambda kv: float(kv[1])):
            print(f"      {position:>4}  {key}")


def test_two_keys_may_share_a_position_and_that_is_the_target_card(capsys):
    """`choose_target` and `confirm_task_type` are both `02`, and that is
    deliberate — the task-type row lives inside the target card.

    Named, because the check above would otherwise be satisfied by a table where
    every key collided on one marker, and because a future reader meeting two
    keys at `02` should find the reason here rather than infer a bug.
    """
    sequence = _sequence()
    shared: Dict[str, list] = {}
    for key, position in sequence.items():
        shared.setdefault(position, []).append(key)
    collisions = {p: sorted(k) for p, k in shared.items() if len(k) > 1}
    assert collisions == {"02": ["choose_target", "confirm_task_type"]}, (
        f"positions are shared by keys this file does not account for: "
        f"{collisions}")
    with capsys.disabled():
        print(f"\n  one shared position: 02 → choose_target + confirm_task_type")


def test_the_comment_names_this_file(capsys):
    """The reason the file has this name.

    `ml/router.py` names the guard in prose. If the name drifts, the comment
    becomes a claim about a mechanism again — which is the whole finding.
    """
    text = (ROOT / "ml" / "router.py").read_text(encoding="utf-8")
    assert "test_the_marker_is_the_constitutional_position" in text, (
        "ml/router.py no longer names this guard; either restore the reference "
        "or rename this file to whatever it does name")
    assert Path(__file__).stem == "test_the_marker_is_the_constitutional_position"
    with capsys.disabled():
        print("\n  the comment and the file agree on the name")

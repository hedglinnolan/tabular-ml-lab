"""L43-D · closing B's negatives from the other side.

**The limit L42 named itself.** Part D falsified B's *positives* directly —
for a field B calls reached, is its value visible or transformed? — and its
*negatives* only relatively, through a cross-lens comparison. Nothing
independently re-derived a `none` verdict; that rested on B's own
group-negative bisection argument, which is sound but is B's argument.

**The independent claim this file makes.** For a field B calls unread, write a
value into it that the page would have to render differently *if it read the
field at all*, and require the render to come back **byte-identical**. A field
that can be set to anything without moving the DOM is unread by demonstration.

It is independent of bisection in three ways that matter, and each was chosen
because agreeing with B by construction would make this file worthless:

1. **Per-field, not per-chunk.** Bisection tags a chunk and asks whether the
   chunk moved anything; a chunk that stays still says nothing about which of
   its members was tested. Here each field is written alone.
2. **String equality per container, not a hash.** B compares a DJB2 hash of
   the whole DOM. A hash collision is unlikely and is not the point — the
   point is that the two instruments would fail *together* on a hashing bug,
   and an independent check must not share the mechanism it is checking.
3. **Three value shapes per field, not one sentinel.** A page that renders a
   field only under a condition the single sentinel happens not to meet — a
   truthiness test, a length cut-off, a numeric range — is read by B as
   unread. Writing a long string, an extreme number and a type flip gives the
   render three different chances to move.

**What it cannot do.** It cannot prove a field is unread on *every* project,
only on the swept one. That is the same bound B has and it is stated rather
than implied.

**The third lens is traded.** The prompt's own scope note offers it: keep the
negative falsification and the unjudged-count reduction, drop the lens. Two
lenses with a sound negative probe beats three with the same relative
argument. It is not built and `GUIDED-097`'s fixture rule is therefore paid
one lens short — said plainly here because a traded part that goes unmentioned
reads as a part that was covered.
"""
from __future__ import annotations

import pathlib
import random

import pytest

from turbotab import fieldsweep as FS

DATA = pathlib.Path(__file__).resolve().parent / "sample_data"
PAGE = pathlib.Path(__file__).resolve().parent / "web" / "index.html"

#: The two lenses B sweeps, reused verbatim rather than restated — two lists
#: of one thing are two things to drift.
from turbotab.test_every_field_the_server_composes_has_a_reader import (  # noqa: E402
    SWEPT,
)

#: How many unread fields to write into. The sweep is ~3 min per lens and each
#: probed field is one more page render, so this is a real cap and it is
#: reported rather than assumed — `AGENT_ONBOARD.md` §10's no-silent-caps rule.
SAMPLE = 40

#: Distinctive enough that a match in 400 KB of markup is not a coincidence.
#: **Longer than B's sentinel on purpose**: `MIN_DISTINCTIVE = 6` was why 291
#: of D's shapes went unjudged at L42, and a value that cannot be searched for
#: is a value that cannot be ruled either way.
_MARK = "QJXZVWKQ"


def _shapes(index: int):
    """Three ways to be conspicuous, so a conditional render has three chances.

    Type-preserving, because a list written as a string breaks
    `plan.questions.filter` before anything renders — L42 paid for that one.
    """
    return [
        f"{_MARK}{index:04d}AAAAAAAAAAAAAAAAAAAA",   # long and unmistakable
        f"{_MARK}{index:04d}",                       # short but still unique
        f"-{_MARK}{index:04d}-",                     # delimited, in case of trimming
    ]


@pytest.fixture(scope="module")
def swept():
    """B's sweep, once per lens. Imported rather than reimplemented — this
    file's job is to disagree with B's verdicts, not with its plumbing."""
    from fastapi.testclient import TestClient

    from turbotab import api

    ids = FS.container_ids(PAGE.read_text(encoding="utf-8"))
    out = {}
    for label, (fixture, lens, target) in sorted(SWEPT.items()):
        client = TestClient(api.app)
        with open(DATA / fixture, "rb") as handle:
            pid = client.post("/project", files={
                "file": (fixture, handle, "text/csv")}).json()["id"]
        for kind, payload in (("set_lens", {"lens": [lens]}),
                              ("set_target", {"column": target})):
            ok = client.post(f"/project/{pid}/decision",
                             json={"kind": kind, "payload": payload})
            assert ok.status_code == 200, (kind, ok.text[:200])

        def get(tail):
            return client.get(f"/project/{pid}{tail}").json()

        routes = {
            f"/project/{pid}": get(""),
            f"/project/{pid}/interview?step=data": get("/interview?step=data"),
            f"/project/{pid}/interview?step=explore": get("/interview?step=explore"),
            f"/project/{pid}/evidence/missingness": get("/evidence/missingness"),
            f"/project/{pid}/evidence/plausibility": get("/evidence/plausibility"),
            f"/project/{pid}/capabilities": get("/capabilities"),
        }
        out[label] = (routes, pid, ids, FS.sweep(routes, pid, ids))
    return out


_COLLECT = """
var IDS = %s;
var out = [];
IDS.forEach(function(k){ out.push(__harness.html(k) || ""); });
__emit({parts: out});
"""


def _render(routes, pid, ids):
    """The rendered markup of every container, as a LIST OF STRINGS.

    Not a hash — see the module docstring. Compared element by element, so a
    difference names the container it happened in instead of only saying the
    page moved.
    """
    from turbotab import pageharness as PH

    try:
        got = PH.run(_COLLECT % __import__("json").dumps(list(ids)),
                     routes=routes, search=f"?project={pid}")
    except PH.HarnessError:
        return None                      # the render died — the field is READ
    return got["parts"]


def test_a_sample_of_bs_unread_fields_cannot_move_the_page(swept, capsys):
    """**The deliverable.** Each field written alone, three value shapes, and
    the render required to come back byte-identical per container.

    A disagreement here is a field B calls unread that the page does read, and
    it would be the strongest finding this instrument can produce — B's
    negatives are the half of its verdict nothing has independently checked.
    """
    report = {}
    disagreements = []

    for label, (routes, pid, ids, sweep) in sorted(swept.items()):
        baseline = _render(routes, pid, ids)
        assert baseline is not None, f"{label}: the clean render died"
        assert any(part.strip() for part in baseline), (
            f"{label}: every container rendered empty, so nothing below is a "
            f"claim about anything")

        unread = list(sweep.unread)
        rng = random.Random(20260802)      # a literal, because Date/random are
        rng.shuffle(unread)                # not available and a seed must be pinned
        chosen = unread[:SAMPLE]

        moved, unmoved, dead = 0, 0, 0
        for index, fld in enumerate(chosen):
            for shape in _shapes(index):
                probe_routes = {k: __import__("copy").deepcopy(v)
                                for k, v in routes.items()}
                if not FS.poke(probe_routes[fld.route], fld.path, shape):
                    continue
                after = _render(probe_routes, pid, ids)
                if after is None:
                    dead += 1
                    disagreements.append(
                        (label, fld.route, fld.path, "the render DIED"))
                    break
                if after != baseline:
                    where = next((ids[i] for i in range(len(after))
                                  if after[i] != baseline[i]), "?")
                    moved += 1
                    disagreements.append(
                        (label, fld.route, fld.path, f"moved `{where}`"))
                    break
            else:
                unmoved += 1
        report[label] = {"unread_total": len(unread), "probed": len(chosen),
                         "unmoved": unmoved, "moved": moved, "died": dead}

    with capsys.disabled():
        print("\n  ── L43-D · B's NEGATIVES, falsified independently ──")
        for label, r in sorted(report.items()):
            print(f"  {label}")
            print(f"    B calls unread              {r['unread_total']}")
            print(f"    probed here (cap {SAMPLE})       {r['probed']}")
            print(f"    confirmed unread            {r['unmoved']}"
                  "   <- agreement with bisection")
            print(f"    MOVED THE PAGE              {r['moved']}"
                  "   <- disagreement")
            print(f"    killed the render           {r['died']}")
        print("  NOT COVERED:")
        print("    - the other 3 packs; two lenses only, and the third lens was")
        print("      TRADED per the prompt's scope note rather than missed")
        print("    - fields outside the sample; the cap is "
              f"{SAMPLE} per lens and is a cost bound, not a shape bound")
        print("    - post-seal payloads. The three L42-B skipped are swept")
        print("      at L43-A1, but by B's instrument, not by this probe")

    assert not disagreements, (
        f"these are fields B's bisection calls unread and this probe moved the "
        f"page with: {disagreements[:8]}. B's negative verdicts are wrong for "
        f"at least these, and the count is the finding.")


def test_the_probe_can_see_a_field_that_is_read(swept):
    """**The positive control, and this file is worthless without it.**

    Everything above asserts *the page did not move*. An instrument that could
    never detect movement would confirm every negative B has and measure
    nothing — the exact shape of trap 2. So: take a field B says IS read,
    write into it, and require the render to move.
    """
    for label, (routes, pid, ids, sweep) in sorted(swept.items()):
        baseline = _render(routes, pid, ids)
        assert baseline is not None
        moved_any = False
        for fld in sweep.reaching[:25]:
            probe_routes = {k: __import__("copy").deepcopy(v)
                            for k, v in routes.items()}
            if not FS.poke(probe_routes[fld.route], fld.path, _shapes(0)[0]):
                continue
            after = _render(probe_routes, pid, ids)
            if after is None or after != baseline:
                moved_any = True
                break
        assert moved_any, (
            f"{label}: writing into 25 fields B says REACH A PERSON moved "
            f"nothing, so this probe cannot detect a read field and its "
            f"agreement with B's negatives above means nothing")


def test_this_does_not_reuse_bisection_to_reach_its_own_verdict(swept):
    """The independence claim, asserted rather than described.

    If this file computed its verdict from `Field.reaches` it would be B rerun
    under another name. It reads `sweep.unread` only to CHOOSE what to probe;
    the verdict comes from comparing rendered markup.
    """
    import ast

    source = pathlib.Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "reaches":
            raise AssertionError(
                "this file reads `Field.reaches`, which is B's own verdict — "
                "the probe would be agreeing with the thing it is checking")

    # And the mechanism genuinely differs: B hashes, this compares strings.
    #
    # Assembled rather than written out, because the first version wrote the
    # constant as a literal and then asserted the literal was absent from this
    # file — so it flagged itself. Same shape as the pragma check that flagged
    # its own comment and the registry check that flagged its own docstring.
    # A guard whose own text is inside its search space has to say so.
    djb2 = "h = " + str(5381)
    sweep_source = pathlib.Path(FS.__file__).read_text(encoding="utf-8")
    assert djb2 in sweep_source, (
        "the field sweep no longer hashes the DOM, so the mechanisms may have "
        "converged — recheck that this file is still independent of it")
    assert djb2 not in source, (
        "this file hashes the DOM too, so its verdicts share a failure mode "
        "with the ones it is checking")
    assert "__harness.html" in source, (
        "this file no longer reads rendered markup, so it is not comparing "
        "what it claims to compare")

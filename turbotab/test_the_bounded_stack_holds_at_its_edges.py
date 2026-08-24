"""L45-D — a bounded list lies at its edges, so drive it there. `GUIDED-149`.

Every edge here is reachable by a real project: a table nothing was found in, a
table with one finding, a table sitting exactly on the bound, a table one over
it, `clinical_labs.csv`'s twenty-one, a stack that is all `critical`, a stack
with none, and a project where no lens was answered so no pack ever ran.

## The two properties, asserted rather than eyeballed

Both are one-liners and both are the kind of thing that silently stops being
true, which is why they are asserted at **every** size rather than at the one
the loop happened to build against:

1. **No finding that gates a decision is ever inside the collapsed group.**
   `ROADMAP.md` Decision B: *a blocker that only offers is not gating*, and
   *blockers rank first*.
2. **Rendered + collapsed == served.** The number in the affordance is the
   number behind it. A count that is off by one at an edge is the app asserting
   something false about its own contents, in a sentence the user reads.

## Why it is driven twice

`attention.stack` is checked directly, because that is where the rule lives and
because Python can construct sizes no fixture produces — a stack of zero, a
stack of exactly two, twenty-one criticals. **And the page is driven**, because
this door's oldest habit is a server that computes correctly beside an interface
that renders something else (`GUIDED-142`, `GUIDED-075`, `GUIDED-058`, and six
measured surfaces). An arithmetic that is exact on the wire and wrong on screen
is the version of this defect that matters, since the affordance is a sentence a
person reads and acts on.

## The findings are real, and that is deliberate

`LOOP.md` trap #3 — *a guard that manufactures the thing whose absence is the
defect*. Every card at every size here is a finding a real driven project served;
sizes are composed by **resampling** that pool with fresh ids, never by inventing
a payload or by overriding a severity to make a case appear. A stack of
twenty-one criticals is twenty-one copies of criticals the metabolomics lens
actually produced.

## `GUIDED-097` — two lenses, and what neither covers

Clinical and metabolomics, plus the no-lens project. `SHAPES_NOT_COVERED` names
what none of them reaches.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

import pytest

from turbotab import attention as A
from turbotab import engine

DATA = Path(__file__).resolve().parent / "sample_data"

#: Two lenses of deliberately different stack shape, plus a project where the
#: lens question was never answered. `(fixture, lens, target)`.
LENSES = {
    "clinical": ("clinical_labs.csv", "clinical", "readmitted"),
    "metabolomics": ("metabolomics_untargeted.csv", "metabolomics", "responder"),
    # A LENS WITH NO PACK, which in this app is *no lens answered* — the pack
    # stream is empty and the Explore stack is profile findings alone. It still
    # carries a target, because `renderEda` returns early without one and the
    # Explore section has not been reached at all: a probe driving a page that
    # never rendered would report the surface as dead rather than as bounded.
    "no lens answered": ("longitudinal_visits.csv", None, "outcome"),
}

#: NOT COVERED, said out loud. A probe that reports only what it drove has not
#: reported its coverage.
SHAPES_NOT_COVERED = (
    "A stack containing a `blocker`-severity FINDING. `ml/router.py:77` ranks "
    "that severity beside `critical` and `NEVER_COLLAPSED` holds both, but "
    "nothing in this repository emits it onto a finding — blockers are "
    "Questions, built from signals (`GUIDED-151`). So the `blocker` half of the "
    "never-collapse rule is exercised synthetically below and by no real "
    "project.",
    "A stack whose findings carry no `rank` at all. `_rank` falls back to "
    "arrival order and that fallback is driven, but every live producer goes "
    "through `engine.rank_findings`, which always sets one.",
    "A real table producing exactly two Explore findings. The small end of the "
    "sixteen fixtures is 1 · 3 · 3 · 3 · 3, so two is synthetic here and is "
    "absent from `prototypes/explore-stack.html` for the same reason.",
)

#: The sizes driven. Zero and one are the degenerate ends; `BOUND` and
#: `BOUND + 1` are where a bound reads as arbitrary if it is going to; 13 and 21
#: are `clinical_labs.csv`'s Explore stack and its whole finding set.
SIZES = (0, 1, 2, A.BOUND - 1, A.BOUND, A.BOUND + 1, 13, 21)


# ── the pool ────────────────────────────────────────────────────────────────

def _driven(fixture: str, lens, target) -> Dict[str, Any]:
    from fastapi.testclient import TestClient

    from turbotab import api

    client = TestClient(api.app)
    with (DATA / fixture).open("rb") as handle:
        pid = client.post("/project", files={
            "file": (fixture, handle, "text/csv")}).json()["id"]
    if lens:
        client.post(f"/project/{pid}/decision",
                    json={"kind": "set_lens", "payload": {"lens": [lens]}})
    if target:
        client.post(f"/project/{pid}/decision",
                    json={"kind": "set_target", "payload": {"column": target}})
    return {"pid": pid, "client": client,
            "project": client.get(f"/project/{pid}").json()}


@pytest.fixture(scope="module")
def pool() -> Dict[str, Any]:
    """Real findings from real projects, split by whether they gate a decision."""
    out: Dict[str, Any] = {"projects": {}, "gating": [], "ordinary": []}
    for label, (fixture, lens, target) in LENSES.items():
        run = _driven(fixture, lens, target)
        out["projects"][label] = run
        for finding in A.explore_findings(run["project"]["findings"]):
            # The literal severities again, and for the reason given at the
            # collapsed check: a pool split by `A.gates_a_decision` is a fixture
            # built out of the rule under test, so emptying `NEVER_COLLAPSED`
            # would empty the pool and every case would fail on the fixture
            # instead of on the property. That is a revert going red for the
            # wrong reason, which the probe reports as verifying nothing.
            bucket = ("gating" if finding.get("severity") in ("critical", "blocker")
                      else "ordinary")
            out[bucket].append(finding)
    assert out["gating"], "no fixture produced a finding that gates a decision"
    assert out["ordinary"], "no fixture produced an ordinary finding"
    return out


def _resample(source: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    """`n` findings drawn from `source`, each a real one with a fresh id."""
    return [dict(source[i % len(source)], id=f"{source[i % len(source)]['id']}#{i}")
            for i in range(n)]


def _ranked(findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rank them the way `engine.rank_findings` does, using its own tables.

    Reusing the engine's ordering rather than restating it: a probe with its own
    sort would be checking the partition against a rule the app does not use.
    """
    ordered = sorted(findings, key=lambda d: (
        engine.SEVERITY_RANK.get(d.get("severity"), 99),
        engine.CONFIDENCE_RANK.get(d.get("confidence"), 1),
        str(d.get("id")),
    ))
    return [dict(f, rank=i) for i, f in enumerate(ordered)]


def _mix(pool: Dict[str, Any], n: int, n_gating: int) -> List[Dict[str, Any]]:
    n_gating = min(n_gating, n)
    return _ranked(_resample(pool["gating"], n_gating)
                   + _resample(pool["ordinary"], n - n_gating))


#: `(label, n, n_gating)` — the eight the prompt names, plus the mixes that make
#: "all critical" and "no critical" a property rather than one case.
CASES = (
    [(f"{n} findings, none gating", n, 0) for n in SIZES]
    + [(f"{n} findings, all gating", n, n) for n in SIZES if n]
    + [("21 findings, one gating", 21, 1),
       ("21 findings, all but one gating", 21, 20)]
)


# ── the two properties, at every size ───────────────────────────────────────

@pytest.mark.parametrize("label,n,n_gating", CASES, ids=[c[0] for c in CASES])
def test_the_arithmetic_is_exact_and_nothing_gating_is_collapsed(
        pool, label, n, n_gating):
    findings = _mix(pool, n, n_gating)
    for bound in (0, 1, 2, A.BOUND, A.BOUND + 1, 40):
        st = A.stack(findings, bound=bound)

        assert st["served"] == n, f"{label} @ {bound}: served {st['served']}, not {n}"
        assert len(st["pushed"]) + len(st["collapsed"]) == st["served"], (
            f"{label} @ bound {bound}: {len(st['pushed'])} pushed + "
            f"{len(st['collapsed'])} collapsed != {st['served']} served")
        assert st["remainder"]["n"] == len(st["collapsed"]), (
            f"{label} @ bound {bound}: the affordance would say "
            f"{st['remainder']['n']} and {len(st['collapsed'])} are behind it")

        # THE SEVERITY WORDS ARE WRITTEN OUT HERE, and that is deliberate rather
        # than a duplicated constant. Reading `A.NEVER_COLLAPSED` back would
        # measure the rule with the rule: empty that set and `gates_a_decision`
        # answers False for everything, so the check would pass while every
        # critical went behind the affordance. `critical` is `engine`'s word and
        # `blocker` is `ROADMAP.md` Decision B's; a test of a constitutional
        # clause states the clause.
        by_id = {f["id"]: f for f in findings}
        collapsed_gating = [i for i in st["collapsed"]
                            if by_id[i].get("severity") in ("critical", "blocker")]
        assert not collapsed_gating, (
            f"{label} @ bound {bound}: {len(collapsed_gating)} findings that "
            f"gate a decision are inside the collapsed group. A blocker that "
            f"only offers is not gating.")

        # THE TYPED REMAINDER'S ARITHMETIC, both ways. A tally that does not sum
        # to the count is the affordance disagreeing with itself in one line.
        for axis in ("by_severity", "by_source"):
            total = sum(e["n"] for e in st["remainder"][axis])
            assert total == st["remainder"]["n"], (
                f"{label} @ bound {bound}: {axis} sums to {total}, not "
                f"{st['remainder']['n']}")

        # AND EVERY ID IS IN EXACTLY ONE PLACE — the arithmetic above is
        # satisfiable by a duplicate paired with an omission.
        seen = st["pushed"] + st["collapsed"]
        assert len(set(seen)) == len(seen), f"{label} @ {bound}: a finding is in both"
        assert set(seen) == set(by_id), f"{label} @ {bound}: the sets differ"


@pytest.mark.parametrize("label,n,n_gating", CASES, ids=[c[0] for c in CASES])
def test_what_a_person_sees_is_stated_at_every_size(pool, label, n, n_gating):
    """The slot always answers, and it answers with a number or with a claim.

    The recorded-absence rule (`DESIGN_LANGUAGE.md` §09): a reader who sees no
    affordance cannot tell *this is everything* from *this is the top few*.
    """
    st = A.stack(_mix(pool, n, n_gating))
    assert st["affordance"].strip(), f"{label}: the slot says nothing at all"
    if st["complete"]:
        assert not st["affordance_open"] and not st["affordance_detail"], (
            f"{label}: a complete stack is offering an expand")
        if n:
            assert str(n) in st["affordance"], (
                f"{label}: 'all shown' without saying how many: "
                f"{st['affordance']!r}")
        else:
            assert "Nothing" in st["affordance"], (
                f"{label}: an empty stack should say so: {st['affordance']!r}")
    else:
        n_more = st["remainder"]["n"]
        assert st["affordance"].startswith(f"{n_more} more"), (
            f"{label}: the affordance does not lead with its count: "
            f"{st['affordance']!r}")
        assert st["affordance_open"], f"{label}: no way to fold it back"
        assert st["affordance_title"], f"{label}: the control states no effect"
        # The count in the sentence IS the count behind it — read back off the
        # prose a person actually sees, not off the field it was composed from.
        said = int(re.match(r"(\d+) more", st["affordance"]).group(1))
        assert said == len(st["collapsed"]), (
            f"{label}: the sentence says {said} and {len(st['collapsed'])} are "
            f"behind it")


def test_a_bound_of_zero_still_cannot_hide_something_that_gates_a_decision():
    """The sharpest form of rule 1: no bound can bury a blocker, including one
    that pushes nothing else at all.

    Three ordinary findings rather than one, because `MIN_COLLAPSE` means a
    remainder of one is shown — so a two-finding fixture would collapse nothing
    and the claim would pass without the rule being exercised. That is `L45`'s
    own lesson about a fixture nothing is wrong for, arriving on the test that
    lesson was written in.
    """
    gating = {"id": "g", "severity": "blocker", "source": "pack",
              "pack": "clinical", "rank": 7}
    ordinary = [{"id": f"o{i}", "severity": "warning", "source": "profile",
                 "rank": i} for i in range(3)]
    st = A.stack(ordinary + [gating], bound=0)
    assert st["pushed"] == ["g"], (
        f"a bound of zero collapsed something that gates a decision: {st}")
    assert st["collapsed"] == ["o0", "o1", "o2"]
    # AND IT IS FIRST, ahead of a finding the engine ranked above it.
    # `engine.SEVERITY_RANK` has no `blocker` key, so `rank_findings` would sort
    # that severity to 99 and put it LAST while `ml/router.py:77` ranks it 0.
    # The surface re-asserts the one clause the constitution makes absolute
    # rather than trusting a rank table that disagrees with itself
    # (`GUIDED-151`).
    st2 = A.stack(ordinary + [gating], bound=40)
    assert st2["pushed"][0] == "g", (
        f"a blocker is not first: {st2['pushed']}. A blocker third in a list of "
        f"nine is a blocker in name only.")


def test_a_finding_with_no_rank_is_placed_by_arrival_and_never_dropped():
    """The fallback `_rank` takes when a producer sets no rank.

    Three, not two: `MIN_COLLAPSE` shows a remainder of one, so a two-finding
    fixture at bound 1 collapses nothing and says nothing about arrival order.
    """
    raw = [{"id": k, "severity": "warning", "source": "profile"}
           for k in ("a", "b", "c")]
    st = A.stack(raw, bound=1)
    assert st["served"] == 3
    assert st["pushed"] == ["a"] and st["collapsed"] == ["b", "c"]


def test_the_structural_stream_is_not_in_this_stack():
    """`structure` findings render in their own card at the Data step, filtered
    by the repair groups. Pulling them in here would put a question the user
    already answered back in front of them."""
    st = A.stack([{"id": "s", "severity": "critical", "source": "structure"},
                  {"id": "p", "severity": "warning", "source": "profile"}])
    assert st["served"] == 1 and st["pushed"] == ["p"]


def test_a_negative_bound_is_refused_rather_than_clamped():
    with pytest.raises(A.StackError):
        A.stack([], bound=-1)


# ── the same arithmetic, on the page ────────────────────────────────────────

def _routes(run, project):
    pid = run["pid"]
    client = run["client"]
    return {
        f"/project/{pid}": project,
        f"/project/{pid}/interview?step=data":
            client.get(f"/project/{pid}/interview?step=data").json(),
        f"/project/{pid}/interview?step=explore":
            client.get(f"/project/{pid}/interview?step=explore").json(),
        f"/project/{pid}/evidence/missingness": {"cards": []},
        f"/project/{pid}/capabilities":
            client.get(f"/project/{pid}/capabilities").json(),
    }


_CARD = re.compile(r'<article class="[^"]*" id="find-([^"]+)"')


@pytest.mark.parametrize("label", sorted(LENSES))
@pytest.mark.parametrize("n,n_gating", [(0, 0), (1, 0), (2, 0),
                                        (A.BOUND, 0), (A.BOUND + 1, 0),
                                        (21, 0), (21, 21), (13, 1)],
                         ids=lambda v: str(v))
def test_the_page_shows_exactly_what_the_stack_says_it_shows(
        pool, label, n, n_gating):
    """The arithmetic where it is read: on screen.

    Server-composed and never rendered is this door's oldest habit, and the
    inverse — rendered and never counted — would put a false number in a
    sentence the user acts on.
    """
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")

    run = pool["projects"][label]
    findings = _mix(pool, n, n_gating)
    project = dict(run["project"], findings=findings,
                   explore_stack=A.stack(findings))
    st = project["explore_stack"]

    # IDS, NOT MARKUP. `pageharness.py`'s own docstring warns that the DOM runs
    # to hundreds of kilobytes and the harness emits over a pipe — "the first
    # version truncated the JSON and the sweep died on its own output". At L47
    # the finding card grew an at-control slot, twenty-one cards crossed the
    # line, and this test started failing with `Unterminated string` rather than
    # with anything about the stack. The assertions only ever wanted the ids and
    # the affordance, so only those cross.
    out = PH.run(
        "function ids(h){ var m, r = [], x = /<article class=\"[^\"]*\" "
        "id=\"find-([^\"]+)\"/g; while ((m = x.exec(h || '')) !== null) "
        "r.push(m[1]); return r; }\n"
        "var shut = {list: ids(__harness.html('profList')),"
        "            more: (__harness.html('profMore') || '').slice(0, 400),"
        "            rest: ids(__harness.html('profRest')),"
        "            calls: __harness.calls().length};\n"
        "__harness.dispatch('click', __harness.target("
        "{'data-stack-more':'1','aria-expanded':'false'}));\n"
        "__emit({shut: shut,"
        " open: {list: ids(__harness.html('profList')),"
        "        rest: ids(__harness.html('profRest')),"
        "        more: (__harness.html('profMore') || '').slice(0, 400),"
        "        calls: __harness.calls().length}});",
        routes=_routes(run, project), search=f"?project={run['pid']}")

    pushed = out["shut"]["list"]
    assert pushed == st["pushed"], (
        f"{label} @ {n}/{n_gating}: the page pushed {pushed}, the server said "
        f"{st['pushed']}")
    assert not out["shut"]["rest"], (
        f"{label} @ {n}/{n_gating}: the collapsed group is in the DOM before "
        f"anyone opened it — hidden content probes as READ while no person can "
        f"see it")

    if st["complete"]:
        assert st["affordance"] in (out["shut"]["more"] or ""), (
            f"{label} @ {n}/{n_gating}: a complete stack does not say so: "
            f"{out['shut']['more']!r}")
        assert "data-stack-more" not in (out["shut"]["more"] or ""), (
            f"{label} @ {n}/{n_gating}: nothing is collapsed and there is an "
            f"expand anyway")
        return

    # THE COUNT IN THE AFFORDANCE IS THE COUNT BEHIND IT, read off the rendered
    # sentence and off the rendered cards — never off the payload twice.
    said = re.search(r">(\d+) more", out["shut"]["more"] or "")
    assert said, (f"{label} @ {n}/{n_gating}: no count in the affordance: "
                  f"{out['shut']['more']!r}")
    opened = out["open"]["rest"]
    assert int(said.group(1)) == len(opened) == len(st["collapsed"]), (
        f"{label} @ {n}/{n_gating}: the affordance says {said.group(1)}, the "
        f"expand rendered {len(opened)}, the server collapsed "
        f"{len(st['collapsed'])}")
    assert opened == st["collapsed"], (
        f"{label} @ {n}/{n_gating}: the expand rendered the wrong findings")

    # AND THE EXPAND IS INSTANT. `DESIGN_LANGUAGE.md` §05.2 ruled it disclosure
    # rather than consequence, so no fetch and no fifth motion slot: everything
    # the group holds is already in `P`.
    assert out["open"]["calls"] == out["shut"]["calls"], (
        f"{label} @ {n}/{n_gating}: opening the group fetched "
        f"{out['open']['calls'] - out['shut']['calls']} time(s)")

    # NOTHING GATING IS BEHIND THE AFFORDANCE, checked on the render rather than
    # on the payload — the payload was already checked above, and this is the
    # copy a person sees.
    by_id = {f["id"]: f for f in findings}
    assert not [i for i in opened if A.gates_a_decision(by_id[i])], (
        f"{label} @ {n}/{n_gating}: the page collapsed something that gates a "
        f"decision")


def test_the_page_shows_everything_when_the_server_serves_no_partition(pool):
    """An older payload, or a route that answered without `explore_stack`.

    The page must not invent a bound — that is the second copy of the rule
    arriving through the back door — and must not show nothing, which would
    shorten the shelf on a server that never asked it to.
    """
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")

    run = pool["projects"]["clinical"]
    findings = _mix(pool, 13, 1)
    project = dict(run["project"], findings=findings)
    project.pop("explore_stack", None)

    out = PH.run("__emit({list: __harness.html('profList'),"
                 "        more: __harness.html('profMore'),"
                 "        rest: __harness.html('profRest')});",
                 routes=_routes(run, project), search=f"?project={run['pid']}")
    shown = _CARD.findall(out["list"] or "")
    assert len(shown) == 13, (
        f"no partition served and the page showed {len(shown)} of 13")
    assert shown == [f["id"] for f in A.explore_findings(findings)], (
        "no partition served and the page invented an order")
    assert not (out["more"] or "").strip(), (
        f"no partition served and the page composed an affordance anyway: "
        f"{out['more']!r}")


def test_the_probe_reports_its_own_coverage(pool, capsys):
    """`LOOP.md` §10: a probe that reports only what it fixed has not reported
    its coverage. Sizes driven, the bound, criticals ever collapsed, and any
    size where the arithmetic failed."""
    sizes: List[int] = []
    collapsed_gating = 0
    mismatched: List[str] = []
    for label, n, n_gating in CASES:
        findings = _mix(pool, n, n_gating)
        by_id = {f["id"]: f for f in findings}
        for bound in (0, 1, 2, A.BOUND, A.BOUND + 1, 40):
            st = A.stack(findings, bound=bound)
            sizes.append(n)
            collapsed_gating += sum(1 for i in st["collapsed"]
                                    if A.gates_a_decision(by_id[i]))
            if len(st["pushed"]) + len(st["collapsed"]) != st["served"]:
                mismatched.append(f"{label} @ bound {bound}")

    with capsys.disabled():
        print(f"\n  ── L45-D · the bounded stack at its edges ──")
        print(f"  shipping bound                 {A.BOUND}")
        print(f"  distinct sizes driven          {sorted(set(sizes))}")
        print(f"  size × severity-mix cases      {len(CASES)}")
        print(f"  partitions computed            {len(sizes)}")
        print(f"  lenses                         {len(LENSES)}  "
              f"({', '.join(sorted(LENSES))})")
        print(f"  real findings in the pool      "
              f"{len(pool['gating'])} gating, {len(pool['ordinary'])} ordinary")
        print(f"  criticals ever collapsed       {collapsed_gating}   <- must be 0")
        print(f"  rendered + collapsed != served {len(mismatched)}")
        for m in mismatched:
            print(f"      {m}")
        print(f"  shapes NOT covered             {len(SHAPES_NOT_COVERED)}")
        for shape in SHAPES_NOT_COVERED:
            print(f"      · {shape}")

    assert collapsed_gating == 0
    assert not mismatched

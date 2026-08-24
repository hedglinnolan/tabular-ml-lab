"""§11 · the thirteen places the metabolomics pack declines to be confident.

`research/METABOLOMICS_PACK.md` §11, *"where confident automation would embarrass
us"* — ranked, and the pack's own framing is that they are **what make it
credible rather than confident**.

## They are badges and refusals, not detectors

Nothing in `packs.METABOLOMICS_HEDGES` reads a table. Every one of them is true
of untargeted metabolomics before any data arrives, which is why they hang off
the **lens answer** rather than off a detector: the user said what kind of
measurements these are, and these are the positions that follow.

That is also why the consumer question has a different shape here. A detector
proves it reaches a person by firing on a fixture; a position proves it by being
**rendered**, and this repository has paid for the difference at six surfaces.
So the claim these tests make is the driven one: the API serves it under the
metabolomics lens, and the page's real controller builds it.

## The badge is per item, and it is not uniform

**Seven DISPUTED, two CONVENTION, four SETTLED**, read out of the file rather
than applied as a policy. Two of the thirteen are not disputes at all:

* item 7, OPLS-DA — §08 marks it *[SETTLED among chemometricians; widely
  misunderstood by practitioners]*. Saying the field disagrees about whether a
  rotation reduces overfitting would be inventing a controversy.
* item 11, Hotelling's T² versus group confidence ellipses — §06.1 calls it a
  *critical distinction the pack must not get wrong*, not an open question.

`AGENT_ONBOARD.md` §00 is why this matters more than it looks: a second,
uncalibrated layer of caution *"makes a SETTLED fact and a DISPUTED one read the
same, which is the exact failure the badge exists to prevent."*
`test_the_statuses_are_the_files_and_are_not_uniform` is the guard.

## `GUIDED-097`

Two target shapes, and the shapes not covered are named in
`SHAPES_NOT_COVERED` — including the one that would actually change the content,
which is a **sub-domain** rather than a target.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from turbotab import packs as P

DATA = Path(__file__).resolve().parent / "sample_data"
RESEARCH = (Path(__file__).resolve().parents[1] / "docs" / "turbotab"
            / "research" / "METABOLOMICS_PACK.md")

#: `GUIDED-097`. Two target shapes on the metabolomics lens. The hedges are
#: lens-scoped and are expected to be INVARIANT to the target, which is a claim
#: worth checking rather than assuming — a block that appeared only on a
#: classification project would be a position the app holds conditionally on
#: something irrelevant to it.
TARGET_SHAPES = {
    "binary numeric": ("metabolomics_untargeted.csv", "responder"),
    "continuous": ("metabolomics_untargeted.csv", "bmi"),
}

#: NOT COVERED, said out loud, because a sweep that reports only what it covered
#: has not reported its coverage.
#:
#: ONE METABOLOMICS TABLE. `sample_data/` holds exactly one untargeted
#: metabolomics fixture, so the two arms above vary the target and not the
#: table. That is the weaker half of `GUIDED-097` and it is stated rather than
#: dressed up.
#:
#: THE SUB-DOMAIN, WHICH IS THE SHAPE THAT WOULD ACTUALLY CHANGE THE CONTENT.
#: §00 forks before anything else — a **targeted** panel takes *no RSD filtering
#: of validated analytes* at all, so item 1 is wrong for it; NMR has no blanks,
#: so item 2 has nothing to say; DIA proteomics has much lower missingness and a
#: different imputation calculus, which is item 3. Every hedge here is written
#: for **untargeted LC-MS** and the block does not yet ask which sub-domain this
#: is. That is the real gap and it is a capability, not a fixture.
#:
#: A PROJECT WITH TWO LENSES. `hedges()` concatenates across selected packs and
#: sorts by rank, and metabolomics is the only pack with a §11 built — so the
#: multi-pack ordering runs but has nothing to interleave with.
SHAPES_NOT_COVERED = [
    "a second metabolomics TABLE — sample_data holds one, so the two arms vary "
    "the target and not the table",
    "a targeted panel, NMR, or DIA proteomics — §00's sub-domain fork changes "
    "items 1, 2 and 3, and the block does not yet ask which sub-domain this is",
    "a project selecting two packs that both carry hedges — metabolomics is "
    "the only §11 built, so the cross-pack rank ordering is exercised with "
    "nothing to interleave",
]

#: The status the FILE gives each item. Written out rather than derived from the
#: code, because a test that read the status off the object it is checking would
#: assert nothing at all — it would be `LOOP.md` trap #2 in one line.
EXPECTED_STATUS = {
    1: ("qc_rsd_threshold", P.DISPUTED),
    2: ("blank_ratio_fold_change", P.DISPUTED),
    3: ("imputation_method", P.DISPUTED),
    4: ("pareto_vs_autoscaling", P.CONVENTION_STATUS),
    5: ("compositionality", P.DISPUTED),
    6: ("batch_correction", P.DISPUTED),
    7: ("oplsda_is_a_rotation", P.SETTLED),
    8: ("q2_threshold", P.DISPUTED),
    9: ("sample_size", P.DISPUTED),
    10: ("eighty_percent_rule", P.CONVENTION_STATUS),
    11: ("hotelling_versus_group_ellipses", P.SETTLED),
    12: ("software_defaults", P.SETTLED),
    13: ("microbiome_analogy", P.SETTLED),
}


# ═══════════ THE THIRTEEN, AND WHAT EACH ONE OWES ═══════════

def test_all_thirteen_of_section_11_are_built_and_ranked():
    """§11 is a ranked list of thirteen and the rank is content: item 1 is where
    the field's disagreement is most likely to reach a reader.

    Both directions, so a fourteenth invented here fails as loudly as a missing
    one — the register is the research's list, not a superset of it.
    """
    ranks = [h.rank for h in P.METABOLOMICS_HEDGES]
    assert ranks == list(range(1, 14)), (
        f"§11 ranks 1..13 and this register has {ranks}")
    assert {h.key for h in P.METABOLOMICS_HEDGES} == {
        key for key, _ in EXPECTED_STATUS.values()}


def test_the_statuses_are_the_files_and_are_not_uniform():
    """**The load-bearing test in this file.**

    Badging all thirteen DISPUTED would be the second, uncalibrated layer of
    caution `AGENT_ONBOARD.md` §00 names as a defect, and it would land on two
    items the research settles outright: OPLS-DA being a rotation that does not
    reduce overfitting, and the T² ellipse not being a group confidence
    ellipse. Rendered identically to *"nobody agrees on the QC RSD threshold"*,
    both stop being useful.
    """
    by_rank = {h.rank: h for h in P.METABOLOMICS_HEDGES}
    for rank, (key, status) in EXPECTED_STATUS.items():
        hedge = by_rank[rank]
        assert hedge.key == key, f"rank {rank} is {hedge.key!r}, not {key!r}"
        assert hedge.evidence.status == status, (
            f"item {rank} ({key}) is badged {hedge.evidence.status} and the "
            f"research badges it {status}")

    statuses = {h.evidence.status for h in P.METABOLOMICS_HEDGES}
    assert statuses == {P.SETTLED, P.CONVENTION_STATUS, P.DISPUTED}, (
        "the register uses one or two of the three badges, which is the "
        "uniform-confidence state the badge exists to end — in whichever "
        "direction it happens to be uniform")


def test_every_disputed_item_states_both_sides_and_offers_a_sensitivity():
    """`DOMAIN_SCIENCE.md` §01's rendering obligation, all three clauses:
    **never defaulted silently, both sides stated, a sensitivity analysis
    offered.**

    The third clause is the one nothing carried before. `Evidence.both_sides`
    enforced the second at construction, and a DISPUTED item could still leave
    the user with two positions and no way to find out which one mattered for
    their study.
    """
    disputed = [h for h in P.METABOLOMICS_HEDGES
                if h.evidence.status == P.DISPUTED]
    assert len(disputed) == 7
    for hedge in disputed:
        assert hedge.evidence.both_sides, hedge.key
        assert hedge.sensitivity, hedge.key
        assert hedge.evidence.may_preselect is False, (
            f"{hedge.key} is DISPUTED and may_preselect is True; DISPUTED is "
            f"never defaulted silently")


def test_a_stated_default_is_a_recommendation_and_never_a_preselection():
    """§11 item 1: *assert a default with a stated rationale, never a rule.*

    Four DISPUTED items state one, and the distinction the payload has to keep
    is between recommending in prose and pre-selecting. `stated_default` is the
    first; `may_preselect` reports the second, and it is False on every one of
    them.
    """
    served = P.hedges([P.METABOLOMICS])
    with_default = [i for i in served["items"] if i["stated_default"]]
    assert len(with_default) >= 4
    for item in with_default:
        if item["evidence_status"] == P.DISPUTED:
            assert item["may_preselect"] is False, item["key"]
    # AND THE ONE §11 ITEM 1 IS ABOUT SAYS ITS RATIONALE, not just its number.
    qc = next(i for i in served["items"] if i["key"] == "qc_rsd_threshold")
    assert "most commonly published untargeted cutoff" in qc["stated_default"]
    assert "not a rule" in qc["stated_default"]
    assert "no widely accepted metric" in qc["both_sides"]


def test_the_imputation_conflict_is_presented_as_the_finding():
    """`DOMAIN_SCIENCE.md` §04's standing rule, on the instance it names:
    *"the metabolomics benchmark says QRILC for MNAR and random forest for MAR;
    a major proteomics benchmark says random forest is robust even under MNAR.
    Both are cited. Present both. The disagreement is the finding."*

    Both halves in one `both_sides`, because splitting them across two claims
    would let a consumer render one.
    """
    hedge = next(h for h in P.METABOLOMICS_HEDGES
                 if h.key == "imputation_method")
    both = hedge.evidence.both_sides
    assert "QRILC" in both and "MNAR" in both
    assert "consistently robust across all MNAR situations" in both
    assert "does not pretend the contradiction is resolved" in both
    # AND THE SENSITIVITY IS THE ONE THE PACK STARRED, wired to the module that
    # actually runs it rather than described.
    assert "two imputation schemes" in hedge.sensitivity
    assert "turbotab/sensitivity.py" in hedge.what_the_app_does


def test_pareto_is_never_presented_as_correct():
    """§11 item 4. CONVENTION rather than DISPUTED because that is the badge
    §04 gives it — *dominant but arbitrary* — and the counter-evidence travels
    in the statement rather than being softened away.

    The recipe table is the behavioral half: Pareto is registered as the pack's
    scaling default with its reason, and autoscaling is PUSHED against it, which
    is what CONVENTION permits and what *never as a fact* requires.
    """
    from turbotab import recipes as _rec

    hedge = next(h for h in P.METABOLOMICS_HEDGES
                 if h.key == "pareto_vs_autoscaling")
    assert "van den Berg" in hedge.statement
    assert "preferred autoscaling" in hedge.statement
    assert "convention rather than as a fact" in hedge.statement

    state = _rec.snapshot()
    try:
        P.load([P.METABOLOMICS])
        operation = _rec.operation("scale")
        pushed = {pair for pair in operation.pushed_alternatives}
        assert ("pareto", "standard") in pushed, (
            "Pareto is the default and autoscaling is not offered beside it, "
            "which is presenting Pareto as the correct choice")
    finally:
        _rec.restore(state)


def test_the_two_settled_items_are_not_dressed_as_disputes():
    """Items 7 and 11. The research settles both, and a DISPUTED badge on either
    would be the app inventing a controversy — the mirror image of `GUIDED-170`,
    which was a SETTLED badge on a claim the app had no business making.
    """
    opls = next(h for h in P.METABOLOMICS_HEDGES
                if h.key == "oplsda_is_a_rotation")
    assert opls.evidence.status == P.SETTLED
    assert opls.evidence.both_sides is None
    assert "does not reduce overfitting" in opls.statement
    assert "predictive subspace is the same" in opls.statement

    ellipse = next(h for h in P.METABOLOMICS_HEDGES
                   if h.key == "hotelling_versus_group_ellipses")
    assert ellipse.evidence.status == P.SETTLED
    assert "outlier boundary" in ellipse.what_the_app_does
    assert "mean and spread" in ellipse.what_the_app_does


def test_nothing_gates_on_q2():
    """§11 item 8, and the behavioral half is a subtraction: the string that
    would implement the gate is nowhere in the engine."""
    hedge = next(h for h in P.METABOLOMICS_HEDGES if h.key == "q2_threshold")
    assert hedge.evidence.status == P.DISPUTED
    assert "rule of thumb rather than a test" in hedge.statement
    assert "no pass/fail gate" in hedge.what_the_app_does.lower()


# ═══════════ ITEM 12 · THE HARD STOP ═══════════

def test_every_source_resolves_to_a_real_section_of_the_pack():
    """The gate's own check, asserted here so the register cannot be read as
    self-certifying. A citation that resolves to nothing is the defect the badge
    was built to remove, one level in.
    """
    text = RESEARCH.read_text(encoding="utf-8")
    headings = {m.group(1).strip()
                for m in re.finditer(r"^#{1,6}\s+(.*?)\s*$", text, re.M)}
    assert headings, "the research file has no headings; this test reads nothing"
    for hedge in P.METABOLOMICS_HEDGES:
        filename, _, section = hedge.evidence.source.partition("#")
        assert filename == "research/METABOLOMICS_PACK.md", hedge.key
        assert section in headings, f"{hedge.key} cites {section!r}, which is not a heading"


def test_the_three_software_defaults_are_refused_and_never_returned():
    """§11 item 12 is a hard stop: *any claim about a specific software default
    — MetaboAnalyst's IQR filter, `pmp`'s blank fold change, structToolbox's
    D-ratio — may not ship as a hard-coded constant.*

    `software_default` has no branch that returns a number. That is structural
    rather than careful: a function that could return one is a place a later
    loop puts a constant.
    """
    keys = [key for key, _ in P.SOFTWARE_DEFAULTS_REFUSED]
    assert len(keys) == 3
    for key in keys:
        with pytest.raises(P.SoftwareDefaultRefusal) as caught:
            P.software_default(key)
        refusal = caught.value
        assert refusal.evidence.status in P.EVIDENCE_STATUSES
        assert refusal.evidence.source.startswith("research/METABOLOMICS_PACK.md#")
        # A REFUSAL THAT OFFERS NOTHING is indistinguishable from a missing
        # feature, and the user still has a real question.
        assert refusal.offer.get("label")
        assert "version" in refusal.offer.get("note", "")

    with pytest.raises(P.PackError):
        P.software_default("not_a_named_software_default")


def test_no_software_default_number_reaches_the_payload():
    """The absence assertion, with its positive control.

    The three values live in §02 of the research and in no served string: the
    IQR filter's aggressive setting for wide tables, the D-ratio acceptance
    criterion, and the blank fold change `pmp` ships. Searched over the whole
    served block, refusals included, because the refusal is exactly where a
    later loop would be tempted to say *the value is X, and we will not use it.*
    """
    served = json.dumps(P.hedges([P.METABOLOMICS]))

    # POSITIVE CONTROL. An empty payload would satisfy every absence below, and
    # this is the deletion that must fail instead.
    assert "near-constant IQR filter" in served
    assert "structToolbox" in served
    assert len(served) > 5000

    for number in ("40%", "50%", "3x", "5x"):
        # `3x`/`5x` appear only inside item 2's list of published blank-ratio
        # thresholds, which the research states as literature values rather than
        # as any tool's default. What must not appear is a value attributed to a
        # named tool, so the check is on the ATTRIBUTION.
        if number in served:
            for tool in ("MetaboAnalyst", "structToolbox", "`pmp`"):
                window = re.findall(
                    re.escape(tool) + r".{0,160}", served, re.S)
                for chunk in window:
                    assert number not in chunk, (
                        f"{number} is attributed to {tool} in the served "
                        f"payload; §11 item 12 says that number belongs to a "
                        f"version this app has not read")


def test_the_served_list_states_its_bound_and_is_not_cut():
    """`GUIDED-209`. A hedge register that dropped its tail would be the surface
    most likely to drop the awkward one, so the two counts travel and are
    equal."""
    served = P.hedges([P.METABOLOMICS])
    assert served["n"] == 13
    assert served["showing"] == len(served["items"]) == served["n"]
    assert served["complete"] is True
    assert served["n_refused"] == len(served["refuses"]) == 3
    assert served["by_status"] == {"SETTLED": 4, "CONVENTION": 2, "DISPUTED": 7}


def test_a_pack_with_no_section_11_produces_nothing_rather_than_an_empty_block():
    """*Nothing to say* and *a section that says nothing* are different
    sentences. Four packs have no §11 built and must produce the first."""
    for lens in ([P.CLINICAL], [P.DIETARY], [P.SURVEY], [P.GENOMICS], [P.OTHER]):
        assert P.hedges(lens) is None, lens
    assert P.hedges([P.CLINICAL, P.METABOLOMICS])["n"] == 13


# ═══════════ AND IT REACHES A PERSON ═══════════

@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES))
def test_the_hedge_block_is_served_under_the_metabolomics_lens(shape):
    """Driven through the real API, and across two target shapes.

    Not `packs.hedges(...)`, which would prove the function and prove nothing
    about the app.
    """
    from fastapi.testclient import TestClient

    from turbotab import api

    fixture, target = TARGET_SHAPES[shape]
    client = TestClient(api.app)
    with open(DATA / fixture, "rb") as handle:
        pid = client.post("/project", files={
            "file": (fixture, handle, "text/csv")}).json()["id"]

    # BEFORE THE LENS IS ANSWERED there is no pack and therefore no position.
    assert client.get(f"/project/{pid}").json()["pack_hedges"] is None

    for kind, payload in (("set_lens", {"lens": [P.METABOLOMICS]}),
                          ("set_target", {"column": target})):
        ok = client.post(f"/project/{pid}/decision",
                         json={"kind": kind, "payload": payload})
        assert ok.status_code == 200, (kind, ok.text[:400])

    block = client.get(f"/project/{pid}").json()["pack_hedges"]
    assert block is not None and block["n"] == 13
    assert [i["rank"] for i in block["items"]] == list(range(1, 14))
    for item in block["items"]:
        assert item["evidence_status"] in P.EVIDENCE_STATUSES
        assert item["source"].startswith("research/METABOLOMICS_PACK.md#")
        assert item["what_the_app_does"]
        if item["evidence_status"] == P.DISPUTED:
            assert item["both_sides"] and item["sensitivity"]


@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES))
def test_the_hedge_block_reaches_a_person(shape):
    """**Trap #6, which this door has paid for at six surfaces.** Composed
    correctly and rendered nowhere is the failure mode; the check is the page's
    own controller, run for real.

    Every DISPUTED item's `both_sides` is asserted ON THE PAGE, because that is
    the string the badge's tooltip promises and nothing rendered before this:
    the tooltip said *"both positions are stated"* while no surface stated them.
    """
    from fastapi.testclient import TestClient

    from turbotab import api
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")

    fixture, target = TARGET_SHAPES[shape]
    client = TestClient(api.app)
    with open(DATA / fixture, "rb") as handle:
        pid = client.post("/project", files={
            "file": (fixture, handle, "text/csv")}).json()["id"]
    for kind, payload in (("set_lens", {"lens": [P.METABOLOMICS]}),
                          ("set_target", {"column": target})):
        assert client.post(f"/project/{pid}/decision",
                           json={"kind": kind, "payload": payload}
                           ).status_code == 200

    project = client.get(f"/project/{pid}").json()
    routes = {
        f"/project/{pid}": project,
        f"/project/{pid}/interview?step=data":
            client.get(f"/project/{pid}/interview?step=data").json(),
        f"/project/{pid}/interview?step=explore":
            client.get(f"/project/{pid}/interview?step=explore").json(),
        f"/project/{pid}/evidence/missingness": {"cards": []},
        f"/project/{pid}/capabilities":
            client.get(f"/project/{pid}/capabilities").json(),
    }
    out = PH.run(
        "__emit({hedges: (__harness.html('packHedges') || '').slice(0, 200000),"
        " parent: (__harness.html('card-eda') || '').slice(0, 400000)});",
        routes=routes, search=f"?project={pid}")
    html = out["hedges"]
    assert html, "the hedge block rendered nothing at all"

    block = project["pack_hedges"]
    for item in block["items"]:
        head = item["statement"][:40]
        assert head in html, (
            f"item {item['rank']} ({item['key']}) is on the wire and not on "
            f"the page: {head!r}")
        assert item["what_the_app_does"][:40] in html, item["key"]
        if item["both_sides"]:
            assert item["both_sides"][:60] in html, (
                f"{item['key']} is DISPUTED and the page shows the badge "
                f"without either position — which is the badge's tooltip "
                f"asserting something the surface does not do")
        if item["sensitivity"]:
            assert item["sensitivity"][:50] in html, item["key"]

    # ALL THREE BADGE STATUSES ON THE PAGE, so a reader can see the register is
    # not uniformly cautious.
    statuses = set(re.findall(r'class="badge (\w+)"', html))
    assert {"settled", "convention", "disputed"} <= statuses, sorted(statuses)

    # THE BOUND, and it is the server's number rather than the page's count.
    assert 'data-hedge-showing="13"' in html and 'data-hedge-of="13"' in html
    assert "Showing 13 of 13" in html

    # ITEM 12'S REFUSALS, rendered with what to do instead.
    for refusal in block["refuses"]:
        assert refusal["reason"][:40] in html, refusal["key"]
        assert refusal["offer"]["label"] in html, refusal["key"]

    # AND THE CONTAINER SITS INSIDE THE EXPLORE STEP.
    #
    # **This one is a file claim and is answered from the file, deliberately.**
    # `out["parent"]` is empty and that is the harness being honest rather than
    # a defect: it reports what was ASSIGNED to a node, and `#card-eda` is
    # static markup nothing assigns. Driving cannot answer *is this container
    # inside that section* without layout, so the honest instrument is the
    # markup — `LOOP.md` trap #5's own carve-out, reserved for claims that are
    # genuinely about the file. What the drive above proves is the harder half:
    # the controller wrote into it.
    page = (Path(__file__).resolve().parent / "web" / "index.html").read_text(
        encoding="utf-8")
    explore = page[page.index('id="sec-eda"'):]
    explore = explore[:explore.index('id="sec-', 10)]
    assert 'id="profList"' in explore, (
        "the Explore section no longer holds the findings list, so this test "
        "is measuring the wrong region")
    assert 'id="packHedges"' in explore

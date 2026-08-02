"""L41-C — sentinel codes, and the reverse-coding audit nothing scored.

`research/CLINICAL_SURVEY_PACK.md` §B1.1 and §B1.2. Deliberately a different
shape from L41's clinical batch: that one was column typing and parse detection,
this is **block structure and a recorded decision reaching a consumer** — which
is what shows a seam if the detector contract has one.

## §B1.1 — the check that was structurally unreachable

The research calls sentinel detection *"the highest-yield check in this pack"*,
and the block detector could not have supported it. `likert_block` tested
`values <= scale`, exact containment, so a 1–5 item carrying a single `9` for
*refused* **failed to be part of its own block** — and a block where enough
items carried one was not found at all. The check the research ranks first was
unreachable from the detector that finds the blocks.

Extended rather than duplicated, on `FEATURE_PARITY.md`'s
`theory_anchors`/`theory_demos` lesson: two registries describing one thing
drift, and two block detectors would be that with the drift able to change what
an instrument *is*.

## §B1.2 — `GUIDED-136`, and it is trap #1

The app **already asks**. `api.py` dispatches `set_reverse_coding`, `packs.py`
carries the `reverse_coding` prior as guard #1's one deliberate exception, and
the question renders. Nothing scored the answer — `AGENT_ONBOARD.md` §07's first
trap, a recorded decision with no consumer, on the one question this pack is
allowed to add.

**Re-rendered after every declared change** is the clause that makes it an audit
rather than a report, and it is why the table is computed per request from the
record as it stands rather than cached.

## `GUIDED-097` — the fixture rule

Two survey fixtures of different shape: the clean instrument, and the same
instrument with its codebook's missing codes written in. The pair is the point —
`survey_instrument.csv` has no sentinels by construction, so every claim about
sentinel handling made against it would be a claim about their absence.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbotab import packs as P
from turbotab import survey as S

DATA = Path(__file__).resolve().parent / "sample_data"

#: `GUIDED-097`. The same 40-item instrument with and without missing codes.
SURVEY_FIXTURES = {
    "a clean return": ("survey_instrument", False),
    "a return with the codebook's missing codes": ("survey_sentinels", True),
}

#: The true reversals, from `make_fixtures.REVERSE_CODED` — the codebook, which
#: is the only place reverse-coding may come from.
REVERSE_CODED = ("item_05", "item_11", "item_17", "item_23",
                 "item_29", "item_34", "item_38", "item_40")

#: NOT COVERED, said out loud.
#:
#: POLYCHORIC CORRELATIONS. §B5.4 is SETTLED that they are the appropriate
#: choice for ordinal items and nothing here computes one. `GUIDED-127`, open
#: and deliberately unbuilt. The audit uses Pearson, says Pearson, and carries
#: the direction of the bias, which is §B5.4's own caption requirement — an
#: approximation would have been worse than the disclosure.
#:
#: THE ITEM–REST DOT PLOT. §B1.2 specifies the table and a dot plot beside it.
#: The table shipped; the plot did not, which is the trade the loop prompt named
#: as the one to make if C ran long. The re-run-after-reversal clause is the
#: hard constraint and it shipped.
#:
#: C/IER SCREENING. §B1.1's longstring, straightlining and even-odd consistency.
#: The research says the thresholds are *arbitrary, no consensus longstring
#: cutoff exists*, so it is `[verify-at-build]` shaped and deliberately unbuilt.
#:
#: INSTRUMENT FINGERPRINTING. `DOMAIN_SCIENCE.md` §01.2 hard-stops it and the
#: loop prompt forbids it: a fingerprint library built from memory is the exact
#: failure this apparatus exists to prevent.
#:
#: ITEM TEXT. The audit's second column is `text` and both fixtures are numeric
#: exports with no item labels, so every row's text is empty. The column is
#: carried because §B1.2 specifies it and because a codebook upload is the
#: obvious next source; nothing here exercises it.
#:
#: A SENTINEL THAT DOES NOT BREAK THE RUN. A `9` in a 0–9 block is a legitimate
#: response and must not be flagged — that is the hard stop's whole basis.
#: Covered by a constructed frame below; no fixture has a 0–9 instrument.
SHAPES_NOT_COVERED = [
    "polychoric correlations — not computable here (GUIDED-127); Pearson is "
    "used, said, and the direction of its bias is stated",
    "the item–rest dot plot — the table shipped and the plot did not, which is "
    "the trade the loop named",
    "C/IER screening — the thresholds are arbitrary by the research's own "
    "account, so it is [verify-at-build] shaped and unbuilt",
    "instrument fingerprinting — hard-stopped by DOMAIN_SCIENCE §01.2",
    "item text — both fixtures are numeric exports with no labels, so the "
    "audit's text column is empty on every row",
    "a 0–9 instrument where 9 is a real response — constructed below, no "
    "fixture has one",
]


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv")


# ═══════════ §B1.1 · THE BLOCK DETECTOR WAS EXTENDED, NOT DUPLICATED ═══════════

def test_there_is_exactly_one_block_detector():
    """`FEATURE_PARITY.md`'s `theory_anchors`/`theory_demos` lesson, applied
    before the second one exists rather than after.

    Two functions that both decide what an instrument is can drift, and the
    drift would change which columns are a scale — so `survey.py` reads
    `packs.likert_block` and defines no block-finding of its own.
    """
    import ast

    tree = ast.parse(open("turbotab/survey.py").read())
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for banned in ("likert_block", "find_block", "detect_block", "_block"):
        assert banned not in defined, (
            f"survey.py defines {banned}; there is one block detector and it "
            f"is packs.likert_block")
    imported = {a.name for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom) for a in n.names}
    assert "likert_block" in imported


@pytest.mark.parametrize("shape", sorted(SURVEY_FIXTURES))
def test_the_block_is_found_on_both_fixtures(shape):
    """**The extension's whole reason.** Before L41 the sentinel fixture's block
    would have been smaller or absent — `values <= scale` dropped every item
    carrying a `9`, so the check the research ranks first could not run on the
    tables it is about."""
    name, has_sentinels = SURVEY_FIXTURES[shape]
    block = P.likert_block(load(name))
    assert block is not None
    assert len(block["columns"]) == 40, (
        f"only {len(block['columns'])} of 40 items are in the block; items "
        f"carrying a sentinel dropped out of their own instrument")
    assert block["scale"] == [1, 2, 3, 4, 5]
    assert bool(block["sentinels"]) is has_sentinels


def test_the_support_is_the_union_across_the_block_not_per_item():
    """§B1.1's own instruction, and the reason is a rarely-endorsed category.

    Read per item, an instrument where nobody picked 5 on one item has a 1–4
    item in it — and a legitimate `5` would then look like the value that breaks
    the run.
    """
    df = load("survey_instrument")
    items = [c for c in df.columns if c.startswith("item_")]
    starved = df.copy()
    # Nobody picks 5 on item_02. Per item its support is 1–4; across the block
    # it is still 1–5.
    starved.loc[starved["item_02"] == 5, "item_02"] = 4
    assert sorted(starved["item_02"].dropna().unique()) == [1, 2, 3, 4]

    block = P.likert_block(starved)
    assert block["observed_support"] == [1, 2, 3, 4, 5], (
        "the support was read per item, so an unused category became a missing "
        "one")
    assert block["sentinels"] == {}, (
        "a real response was reported as a sentinel because one item never "
        "used it")
    assert len(block["columns"]) == len(items)


def test_the_rule_is_the_run_and_the_known_list_only_corroborates():
    """§B1.1 lists 7/8/9/77/88/99/-1/-8/-9/98/999. The **rule** is that the
    value breaks the observed contiguous run, because a codebook may use
    anything — so a `12` in a 1–5 block is flagged although no list names it,
    and this is what keeps the detector from being a lookup table."""
    df = load("survey_instrument").copy()
    df.loc[df.index[:9], "item_03"] = 12
    block = P.likert_block(df)
    assert block["sentinels"].get("item_03") == [12]

    reading = next(r for r in S.read_sentinels(df)[1] if r.column == "item_03")
    assert reading.known == (), (
        "12 was reported as a known code; it is not in the list and the "
        "finding must say so rather than implying a codebook it has not seen")


def test_too_many_out_of_run_values_make_it_a_different_variable():
    """**The share guard, and it is the boundary the rule needs.**

    Every value outside a 1–5 block's support breaks the run by definition, so
    *breaks the run* alone would read any 1–7 item dropped into a 1–5
    instrument as an item with two hundred sentinel codes. Above a quarter of a
    column, it stops being an item with coded absences and becomes a different
    variable, and it drops out of the block as it always did.

    The threshold is **this module's own** — §B1.1 states no share — and the
    reason for a quarter is that a *don't know* rate above it is a question
    about the question rather than a coding artifact.
    """
    df = load("survey_instrument").copy()
    df.loc[df.index[:20], "item_03"] = 9              # 6.7% — an item with a code
    block = P.likert_block(df)
    assert block["sentinels"].get("item_03") == [9]
    assert "item_03" in block["columns"]

    heavy = load("survey_instrument").copy()
    heavy.loc[heavy.index[:120], "item_03"] = 9       # 40% — a different variable
    block = P.likert_block(heavy)
    assert "item_03" not in block["columns"]
    assert "item_03" not in (block["sentinels"] or {})
    assert P._MAX_SENTINEL_SHARE == 0.25


def test_a_nine_in_a_zero_to_nine_scale_is_a_response():
    """**The hard stop's whole basis.** *Some legitimate scales do run 0–9*, so
    a detector keyed on the value rather than on the run would recode a real
    response — which is the irreversible-if-wrong action §01.2 forbids."""
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({
        f"nrs_{i:02d}": rng.integers(0, 10, 200) for i in range(10)})
    block = P.likert_block(frame)
    # 0–9 is not one of the declared response sets, so this is not read as an
    # instrument at all — which is the safe answer. What must NOT happen is the
    # 9s being reported as sentinels.
    if block is not None:
        assert not block["sentinels"], (
            "a 9 on a 0–9 scale was called a sentinel, which is the recode "
            "that silently inverts a legitimate instrument")
    assert S.sentinel_codes_finding(frame) is None


# ═══════════ §B1.1 · THE FINDING ═══════════

def test_the_finding_reports_the_shift_the_research_leaves_as_x():
    """§B1.1 writes the coaching sentence with an `[X]` in it:

    > *"a 9 treated as a response would shift this item's mean by [X] and
    > propagate into every scale score and every model."*

    Computing the X is what makes it actionable rather than a warning.
    """
    finding = S.sentinel_codes_finding(load("survey_sentinels"))
    assert finding is not None
    items = {row["item"]: row for row in finding["params"]["items"]}
    assert set(items) == {"item_05", "item_09", "item_14", "item_22", "item_33"}

    worst = items["item_14"]
    assert worst["sentinel_values"] == [9]
    assert worst["n_sentinel"] == 33
    # The shift is SIGNED, because the direction is information: a high sentinel
    # inflates and a -9 would deflate.
    assert worst["mean_shift"] > 0
    # `abs=0.002` because all three are rounded to three places independently,
    # so the difference of two rounded numbers and the rounded difference can
    # disagree in the last digit. Asserting they are the same NUMBER would be
    # asserting something arithmetic that is not true.
    assert worst["mean_shift"] == pytest.approx(
        worst["mean_as_responses"] - worst["mean_excluding"], abs=0.002)
    assert f"{worst['mean_shift']:+.3f}" in finding["detail"]
    assert "propagates into every scale score" in finding["detail"]
    assert "moves the item's mean" in finding["why_it_matters"]


def test_it_is_a_hard_stop_and_the_payload_says_so():
    """`DOMAIN_SCIENCE.md` §01.2's third hard stop. `GUIDED-064`'s class says
    the machine-readable form must not be lossier than the sentence, and *never
    auto-recode* is most of what this finding is."""
    finding = S.sentinel_codes_finding(load("survey_sentinels"))
    assert finding["severity"] == "critical"
    assert finding["fix_kind"] == "none"
    assert finding["params"]["hard_stop"] == "never_auto_recode"
    assert "0-9" in finding["params"]["hard_stop_because"]
    assert "has not recoded" in finding["detail"]

    # AND THE TABLE IS UNTOUCHED, asserted rather than assumed.
    before = load("survey_sentinels")
    S.sentinel_codes_finding(before)
    pd.testing.assert_frame_equal(before, load("survey_sentinels"))


def test_dont_know_is_carried_as_DISPUTED_and_refusal_is_not():
    """§B1.1 holds two of these at different statuses and the badge has to say
    so. *Sentinels must be recoded* is SETTLED; *'don't know' is the same as
    missing* is a genuine, unresolved survey-methodology question, and dropping
    it can bias the sample toward people with formed opinions."""
    finding = S.sentinel_codes_finding(load("survey_sentinels"))
    claims = {c["key"]: c for c in finding["evidence"]["claims"]}
    assert claims["must_recode"]["evidence_status"] == "SETTLED"
    assert claims["never_auto_recode"]["evidence_status"] == "SETTLED"
    assert claims["dont_know_is_not_missing"]["evidence_status"] == "DISPUTED"
    assert claims["dont_know_is_not_missing"]["both_sides"]


def test_the_clean_fixture_reports_nothing():
    """The negative control, and it is the fixture eleven other claims use."""
    assert S.sentinel_codes_finding(load("survey_instrument")) is None


def test_the_detector_reaches_an_upload():
    """`GUIDED-058`'s class. Driven through the API with the lens answered."""
    from fastapi.testclient import TestClient

    from turbotab import api

    client = TestClient(api.app)
    with open(DATA / "survey_sentinels.csv", "rb") as handle:
        pid = client.post("/project", files={
            "file": ("survey_sentinels.csv", handle, "text/csv")}).json()["id"]
    ok = client.post(f"/project/{pid}/decision", json={
        "kind": "set_lens", "payload": {"lens": [P.SURVEY]}})
    assert ok.status_code == 200, ok.text
    served = {f["id"] for f in client.get(f"/project/{pid}").json()["findings"]}
    assert "pack::survey::sentinel_codes" in served
    assert "pack::survey::ordinal_declared" in served


# ═══════════ §B1.2 · THE AUDIT ═══════════

def test_the_audit_is_the_table_the_research_specifies():
    """item | text | item–rest r (raw) | reversal declared? | r after | status."""
    out = S.audit(load("survey_instrument"))
    assert out["available"] is True
    assert out["n_items"] == 40
    for row in out["rows"]:
        assert set(row) >= {"item", "text", "item_rest_r_raw",
                            "reversal_declared", "item_rest_r_after_reversal",
                            "status", "because"}
        assert row["status"] in S.STATUSES


def test_it_never_proposes_a_reversal():
    """**§B1.2's central constraint, and it is SETTLED.**

    A negative item–rest correlation has four incompatible explanations and
    correlations cannot distinguish them. Asserted as the ABSENCE of a whole
    category — no status in the vocabulary proposes an action — because
    `FEATURE_PARITY.md` prefers asserting the absence of a class where the
    guarantee is a subtraction.
    """
    for status, because in S.STATUSES.items():
        assert "should be reversed" not in because.lower()
        assert not status.startswith("should_")
        assert status not in ("pass", "fail")
        assert "PASS" not in because and "FAIL" not in because

    out = S.audit(load("survey_instrument"))
    assert "four incompatible causes" in out["will_not_infer"]
    for row in out["rows"]:
        assert "reverse this" not in row["because"].lower()

    negative = [r for r in out["rows"] if r["status"] == "negative_undeclared"]
    assert {r["item"] for r in negative} == set(REVERSE_CODED), (
        "the fixture's genuinely reverse-worded items are the negative ones, "
        "which is exactly why the app must not act on that")


def test_it_says_pearson_and_which_way_the_bias_runs():
    """§B5.4 is SETTLED that polychoric is the appropriate choice and nothing
    here computes one (`GUIDED-127`). The disclosure is **not symmetric
    hedging** — it says which way every number in the table is wrong."""
    out = S.audit(load("survey_instrument"))
    assert out["correlation_method"] == "pearson"
    assert "polychoric" in out["correlation_disclosure"]
    assert "attenuated" in out["correlation_disclosure"]
    assert "nearer zero" in out["correlation_disclosure"]
    assert "does not approximate" in out["correlation_disclosure"]


def test_the_convention_is_never_a_verdict():
    """`DOMAIN_SCIENCE.md` §01.2's last hard stop: never stamp PASS/FAIL on a
    threshold. 0.30 is Nunnally & Bernstein's convention and several of the
    thresholds in that family are actively contested."""
    out = S.audit(load("survey_instrument"))
    assert out["convention"] == 0.30
    assert "not a law" in out["convention_is"]
    assert "PASS" in out["convention_is"] and "FAIL" in out["convention_is"]
    assert "Nothing here is stamped" in out["convention_is"]


def test_the_item_rest_correlation_is_corrected():
    """The item is excluded from its own rest score. An uncorrected item–total
    correlation includes the item in its own total, which inflates it by
    construction and by more the shorter the scale."""
    df = load("survey_instrument")
    items = [c for c in df.columns if c.startswith("item_")]
    corrected = S._corrected_item_rest(df[items], "item_01")

    total = df[items].sum(axis=1, min_count=len(items))
    uncorrected = float(df["item_01"].corr(total))
    assert corrected < uncorrected, (
        "the item is inside its own total, so the correlation is inflated by "
        "construction")


def test_sentinels_come_out_before_any_correlation_is_computed():
    """A `9` in a 1–5 item read as a response is a four-point outlier dropped
    into a correlation, and on a reverse-worded item it pushes the correlation
    the wrong way twice. `item_05` is the fixture's deliberate both-at-once
    cell."""
    out = S.audit(load("survey_sentinels"))
    assert out["sentinels_excluded"] == 102
    assert set(out["sentinel_items"]) == {"item_05", "item_09", "item_14",
                                          "item_22", "item_33"}
    row = next(r for r in out["rows"] if r["item"] == "item_05")
    assert row["sentinels_excluded"] == 18

    # And the correlation is the one computed WITHOUT them. Compared against a
    # frame where the sentinels are already blank, which is the same number by
    # construction if and only if the exclusion happened.
    df = load("survey_sentinels")
    blanked = df.copy()
    for column in out["sentinel_items"]:
        blanked.loc[blanked[column].isin([7, 8, 9]), column] = np.nan
    clean = S.audit(blanked)
    for a, b in zip(out["rows"], clean["rows"]):
        assert a["item"] == b["item"]
        assert a["item_rest_r_raw"] == pytest.approx(b["item_rest_r_raw"],
                                                     abs=1e-9)


# ═══════════ §B1.2's HARD CONSTRAINT · IT RE-RUNS ═══════════

@pytest.mark.parametrize("shape", sorted(SURVEY_FIXTURES))
def test_the_table_re_renders_after_a_declared_change(shape):
    """**The clause that makes this an audit rather than a report.**

    Driven through the decision route, because the claim is that the *record*
    reaches the table — `set_reverse_coding` has been dispatched since the
    survey pack shipped and nothing read it (`GUIDED-136`).
    """
    from fastapi.testclient import TestClient

    from turbotab import api

    name, _ = SURVEY_FIXTURES[shape]
    client = TestClient(api.app)
    with open(DATA / f"{name}.csv", "rb") as handle:
        pid = client.post("/project", files={
            "file": (f"{name}.csv", handle, "text/csv")}).json()["id"]

    before = client.get(f"/project/{pid}/evidence/reverse-coding").json()
    assert before["declared_reversed"] == []
    negative = {r["item"] for r in before["rows"]
                if r["status"] == "negative_undeclared"}
    assert negative == set(REVERSE_CODED)

    ok = client.post(f"/project/{pid}/decision", json={
        "kind": "set_reverse_coding",
        "payload": {"columns": list(REVERSE_CODED)}})
    assert ok.status_code == 200, ok.text

    after = client.get(f"/project/{pid}/evidence/reverse-coding").json()
    assert after["declared_reversed"] == sorted(REVERSE_CODED)
    resolved = {r["item"] for r in after["rows"]
                if r["status"] == "resolved_by_reversal"}
    assert resolved == set(REVERSE_CODED), (
        "the audit did not re-render against the new declaration, which makes "
        "it a report rather than an audit")
    assert not any(r["status"] == "negative_undeclared" for r in after["rows"])
    assert after["warnings_after_reversal"] == []


def test_the_last_declaration_wins_because_the_past_is_editable():
    """A user who corrects their codebook gets an audit of the correction.
    Folding forward rather than taking the first is what makes that true."""
    from fastapi.testclient import TestClient

    from turbotab import api

    client = TestClient(api.app)
    with open(DATA / "survey_instrument.csv", "rb") as handle:
        pid = client.post("/project", files={
            "file": ("survey_instrument.csv", handle, "text/csv")}).json()["id"]
    for columns in (["item_01"], list(REVERSE_CODED)):
        client.post(f"/project/{pid}/decision", json={
            "kind": "set_reverse_coding", "payload": {"columns": columns}})
    out = client.get(f"/project/{pid}/evidence/reverse-coding").json()
    assert out["declared_reversed"] == sorted(REVERSE_CODED)
    assert "item_01" not in out["declared_reversed"]


def test_an_item_still_negative_after_reversal_is_warned_about():
    """§B1.2, verbatim: *"After reversal, item `x` still correlates −0.12 with
    the rest of its scale. Either it was already reversed in the source data,
    or it does not belong to this scale."*

    Constructed by declaring a reversal on an item that was **already** in the
    right direction, which is the double-reversal case the sentence names.
    """
    df = load("survey_instrument")
    out = S.audit(df, declared=["item_01"])
    row = next(r for r in out["rows"] if r["item"] == "item_01")
    assert row["item_rest_r_raw"] > 0
    assert row["item_rest_r_after_reversal"] < 0
    assert row["status"] == "negative_after_reversal"
    assert "already reverse-scored" in row["because"]
    assert [w["item"] for w in out["warnings_after_reversal"]] == ["item_01"]


def test_the_warning_is_lifted_out_of_the_table():
    """The re-run's verdict buried in a 40-row table is a report. §B1.2's hard
    constraint is that the check runs *again*, so its result is served
    separately as well as in the rows."""
    out = S.audit(load("survey_instrument"), declared=["item_01"])
    assert out["warnings_after_reversal"]
    assert out["warnings_after_reversal"][0]["item"] == "item_01"
    assert out["warnings_after_reversal"][0]["because"]


def test_the_union_support_is_what_the_audit_reverses_against():
    """**Where the union is load-bearing, driven end to end.**

    A revert probe on `observed_support` came back `GREEN — NOT LOAD-BEARING`
    against the block-detection tests, and it was right: detection matches
    against the declared `_LIKERT_SETS` and already tolerates one absent
    category, so reading the support per item does not change *which* columns
    are a block.

    It changes what the block is reversed against. `audit()` passes
    `observed_support` to `_reversed`, so an item where nobody picked the top
    category would be flipped against its own 1–4 range while every other item
    flips against 1–5 — a silent rescale of one item inside a scale being
    summed. This is that path, with a starved item and a declared reversal.
    """
    starved = load("survey_instrument").copy()
    starved.loc[starved["item_05"] == 5, "item_05"] = 4
    assert sorted(starved["item_05"].dropna().unique()) == [1, 2, 3, 4]

    block = P.likert_block(starved)
    assert block["observed_support"] == [1, 2, 3, 4, 5], (
        "the support collapsed to the starved item's own range")

    out = S.audit(starved, declared=["item_05"])
    row = next(r for r in out["rows"] if r["item"] == "item_05")
    assert row["status"] == "resolved_by_reversal", (
        f"the starved item did not reverse cleanly: {row}")
    # The reversal is arithmetic and this is the arithmetic: against the block's
    # 1–5, a 4 becomes a 2; against the item's own 1–4 it would become a 1.
    against_block = S._reversed(starved["item_05"], out["observed_support"])
    assert against_block[starved["item_05"] == 4].eq(2).all()


def test_reversal_uses_the_blocks_support_and_not_the_items_own_max():
    """An item where nobody picked 5 has an observed max of 4; reversing it
    against its own max would map its 1s to 4 while every other item's map to 5,
    silently rescaling one item inside a scale being summed."""
    starved = load("survey_instrument").copy()
    starved.loc[starved["item_02"] == 5, "item_02"] = 4
    observed = sorted(starved["item_02"].dropna().unique())
    assert observed == [1, 2, 3, 4], "the fixture no longer starves the top"

    against_block = S._reversed(starved["item_02"], [1, 2, 3, 4, 5])
    against_itself = S._reversed(starved["item_02"], observed)
    # A 1 maps to 5 against the block and to 4 against the item's own range.
    # Against the item's own range it is a DIFFERENT variable from every other
    # item in the sum, which is the silent rescale.
    assert against_block.max() == 5 and against_block.min() == 2
    assert against_itself.max() == 4, (
        "the two are the same, so this test is not comparing anything")
    assert not against_block.equals(against_itself)


# ═══════════ AND IT REACHES A PERSON ═══════════

def test_every_pull_affordance_survives_being_clicked():
    """**`GUIDED-139`, found by building this one and driving it.**

    `nudge()` was deleted at `DRIVE-006` — correctly; it scrolled a reader past
    the card they were reading on every reveal — and **seven call sites
    outlived it.** One of them is inside `runPull`, so *every* pull affordance
    in the Guided door threw `ReferenceError: nudge is not defined` and rendered
    `notBuiltPanel(err.message)`.

    That is worse than a crash, and the reason is the panel it landed in:
    `notBuiltPanel` is the app's honest *we do not have this* surface
    (`GUIDED-006`), so four live capabilities rendered as deliberate absences
    with an internal error where the reason belongs. The governing rule's
    assert-something-false branch, in the control that exists to prevent it.

    Asserted over **every** built pull rather than over the one that found it,
    because a test naming `reverse-coding` would have passed with the other four
    still dead — which is precisely the state that shipped.
    """
    from fastapi.testclient import TestClient

    from turbotab import api
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")

    client = TestClient(api.app)
    with open(DATA / "survey_sentinels.csv", "rb") as handle:
        pid = client.post("/project", files={
            "file": ("survey_sentinels.csv", handle, "text/csv")}).json()["id"]
    client.post(f"/project/{pid}/decision", json={
        "kind": "set_target", "payload": {"column": "sought_support"}})

    caps = client.get(f"/project/{pid}/capabilities").json()
    built = {key: cap for key, cap in caps["pulls"].items() if cap["built"]}
    assert len(built) >= 4, sorted(built)

    routes = {
        f"/project/{pid}": client.get(f"/project/{pid}").json(),
        f"/project/{pid}/interview?step=data":
            client.get(f"/project/{pid}/interview?step=data").json(),
        f"/project/{pid}/interview?step=explore":
            client.get(f"/project/{pid}/interview?step=explore").json(),
        f"/project/{pid}/capabilities": caps,
    }
    for cap in built.values():
        endpoint = cap["endpoint"]
        url = (f"/project/{pid}/evidence/{endpoint}"
               + ("?page=0" if endpoint == "histograms" else ""))
        routes[url] = client.get(
            f"/project/{pid}/evidence/{endpoint}").json()

    for key, cap in sorted(built.items()):
        out = PH.run(
            """
            __harness.dispatch('click', __harness.target(
              {'data-look': '%s', 'data-endpoint': '%s'}, ['pill']));
            setTimeout(function(){
              /* Truncated: the histogram pager renders forty inline SVGs and
                 the emitted JSON is read back over a pipe. Every string this
                 asserts on is in the first panel. */
              var h = __harness.html('paletteBox') || "";
              __emit({html: h.slice(0, 3000), length: h.length});
            }, 0);
            """ % (key, cap["endpoint"]),
            routes=routes, search=f"?project={pid}")
        html = out["html"]
        assert html, f"{key} rendered nothing at all"
        # **THE ASSERTION IS ABOUT THE INTERNAL ERROR, not about the panel.**
        # `look::r1_plausibility` on a survey table legitimately renders
        # *Nothing to draw* with a true sentence — no variable here matches the
        # physiologic reference — and that is the panel doing its job. What must
        # never appear inside it is a JavaScript message, because that turns the
        # app's honest *we do not have this* surface into a lie about a
        # capability it does have.
        for leak in ("is not defined", "is not a function", "undefined is not",
                     "Cannot read propert"):
            assert leak not in html, (
                f"{key} threw a {leak!r} into the panel, so a live capability "
                f"renders as a deliberate absence: {html[:220]}")


def test_the_deleted_helper_has_no_callers_left():
    """`GUIDED-139`'s instance, pinned so it cannot come back.

    `nudge()` is the one function in this page whose deletion is a **recorded
    decision** — `DRIVE-006`, and the reasoning sits above where it used to be,
    because scrolling a reader past the card they were reading was a real
    defect. So a caller is not a missing function; it is a call to something the
    project decided against.

    **The general form is `GUIDED-140` and is deliberately not attempted here.**
    Checking *every* called name against the page's declarations needs a
    JavaScript tokenizer: the regex version reports 15 to 37 false positives
    because an apostrophe in a comment opens a string literal that runs until
    the next one, and the code in between disappears. A guard with a
    double-digit false-positive rate is one that gets muted, which is worse than
    no guard. The behavioral check above is what caught this and is the one that
    generalizes.
    """
    import re

    source = open("turbotab/web/index.html", encoding="utf-8").read()
    # Not preceded by a backtick, because the note explaining the deletion
    # quotes the signature as `nudge(el)` and that mention is the record rather
    # than a caller. The one place the name should appear is the paragraph
    # saying why it does not.
    calls = [m.start() for m in re.finditer(r"(?<![\w.$`])nudge\s*\(", source)]
    assert not calls, (
        f"{len(calls)} call(s) to `nudge()` are back. It was deleted at "
        f"DRIVE-006 and the reasoning is in the file where it used to be; a "
        f"caller throws ReferenceError into whatever panel it lands in, and "
        f"for `runPull` that panel is the one that means *we do not have "
        f"this*.")

def test_the_audit_is_offered_only_where_there_is_a_scale_to_audit():
    """`GUIDED-006`: a control that silently does nothing is worse than one that
    says so — and so is one that is simply absent. Dark, with the reason on the
    chip."""
    from fastapi.testclient import TestClient

    from turbotab import api

    client = TestClient(api.app)
    for name, expect_built in (("survey_instrument", True),
                               ("clinical_labs", False)):
        with open(DATA / f"{name}.csv", "rb") as handle:
            pid = client.post("/project", files={
                "file": (f"{name}.csv", handle, "text/csv")}).json()["id"]
        caps = client.get(f"/project/{pid}/capabilities").json()
        chip = caps["pulls"]["look::reverse_coding"]
        assert chip["built"] is expect_built, name
        if not expect_built:
            assert "no block of items" in chip["not_built_reason"].lower()

        served = client.get(f"/project/{pid}/evidence/reverse-coding").json()
        assert served["available"] is expect_built


@pytest.mark.skipif(
    not __import__("turbotab.pageharness", fromlist=["x"]).available(),
    reason="no JS engine on this machine")
def test_the_page_renders_the_audit_table():
    """**Trap #6**, which this door has paid for at six surfaces: the server
    composes a user-facing string and the interface never renders it. Driven
    through the page's own controller."""
    from fastapi.testclient import TestClient

    from turbotab import api
    from turbotab import pageharness as PH

    client = TestClient(api.app)
    with open(DATA / "survey_sentinels.csv", "rb") as handle:
        pid = client.post("/project", files={
            "file": ("survey_sentinels.csv", handle, "text/csv")}).json()["id"]
    # THE TARGET, because `renderPalette` runs only once the project has one —
    # the palette is an Explore-step surface and Explore does not open until
    # the outcome is named.
    client.post(f"/project/{pid}/decision", json={
        "kind": "set_target", "payload": {"column": "sought_support"}})
    client.post(f"/project/{pid}/decision", json={
        "kind": "set_reverse_coding", "payload": {"columns": ["item_01"]}})
    audit = client.get(f"/project/{pid}/evidence/reverse-coding").json()
    assert audit["warnings_after_reversal"], "nothing to render"

    routes = {
        f"/project/{pid}": client.get(f"/project/{pid}").json(),
        f"/project/{pid}/interview?step=data":
            client.get(f"/project/{pid}/interview?step=data").json(),
        f"/project/{pid}/interview?step=explore":
            client.get(f"/project/{pid}/interview?step=explore").json(),
        f"/project/{pid}/evidence/missingness": {"cards": []},
        f"/project/{pid}/evidence/plausibility": {"columns": []},
        f"/project/{pid}/capabilities":
            client.get(f"/project/{pid}/capabilities").json(),
        f"/project/{pid}/evidence/reverse-coding": audit,
    }
    # **TWO CLAIMS, and both are needed.** `__harness.target` builds a synthetic
    # element, so dispatching a click on one proves the HANDLER routes it and
    # proves nothing about whether the control exists — which would be a guard
    # manufacturing the thing whose absence is the defect (`GUIDED-134`). So the
    # real palette is read first, off the render, and the click is dispatched
    # second.
    out = PH.run(
        """
        var bar = __harness.html('palette') || "";
        __harness.dispatch('click', __harness.target(
          {'data-look': 'look::reverse_coding',
           'data-endpoint': 'reverse-coding'}, ['pill']));
        setTimeout(function(){
          __emit({bar: bar, html: __harness.html('paletteBox'),
                  calls: __harness.calls().map(function(c){ return c.path; })});
        }, 0);
        """,
        routes=routes, search=f"?project={pid}")

    assert 'data-look="look::reverse_coding"' in out["bar"], (
        "no control in the Guided door opens the reverse-coding audit, so the "
        f"consumer GUIDED-136 was filed about still does not exist. The "
        f"palette rendered: {out['bar'][:400]}")
    assert 'data-endpoint="reverse-coding"' in out["bar"], (
        "the chip is offered and carries no endpoint, so clicking it fetches "
        "nothing")
    assert "notbuilt" not in out["bar"].split("look::reverse_coding")[0][-120:], (
        "the chip is rendered dark on a survey table that has a scale to audit")
    assert any("evidence/reverse-coding" in url for url in out["calls"]), (
        "the click did not reach the endpoint")
    html = out["html"]
    assert "Reverse-coding audit" in html
    assert "item_01" in html
    # THE HARD CONSTRAINT IS ABOVE THE NUMBERS. A reader who meets the table
    # first has already started inferring from it.
    assert html.index("four incompatible causes") < html.index("<table")
    assert "polychoric" in html
    assert "already reverse-scored" in html
    assert "not a law" in html
    # The sentinel exclusion is stated, because a correlation computed over 102
    # excluded values is a different number from one computed over all of them.
    assert "sentinel value(s)" in html

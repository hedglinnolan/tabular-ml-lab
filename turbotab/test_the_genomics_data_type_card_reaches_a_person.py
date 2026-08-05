"""L50-B — `GENOMICS_PACK.md` §02, the highest-leverage diagnostic in the pack.

§02 says why it is: **it determines what is legal downstream, and getting it
wrong is the commonest real failure.** TPM handed to a count model runs cleanly
and reports p-values that are wrong — nothing errors, and the numbers look
ordinary. So the classification is not a label. It is the thing every later
sentence about this matrix rests on.

## The orientation that decides every statistic, stated once

§01's convention is **genes in rows, samples in columns**. This app's tables are
**samples in rows**, which `turbotab/orientation.py` exists to establish. So
§02's *"per column"* is **per sample** here and a row sum is a **library size**.
Read the other way round the CPM test asks whether every gene sums to a million,
which is true of nothing, and every reading in this file would be transposed.

## What this file asserts, and in which half

**THE DETECTOR HALF** — nine signatures against eight shipped fixtures and one
constructed frame, each asserting the classification, the confidence and the
capability rows it produces. That is behavior and it is driven.

**THE REACH HALF** — every one of them goes up through the real API under the
genomics lens, and the card is then rendered by the page's own controller under
node. `GUIDED-058` and `GUIDED-142` are both in this file's history: a detector
reachable only from its own test, and five packs' findings computed correctly
and rendered nowhere. *Does an upload reach it* and *does a person see it* are
different questions and this file asks both.

## `GUIDED-097` — eight fixtures, and the ninth signature named as uncovered

The rule is two fixtures of different shape minimum. There are eight here and
they are eight different signatures, which is the strongest form of it this
pack can produce. `SHAPES_NOT_COVERED` names what none of them reaches, and the
first entry is the one §02 leaves genuinely open.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbotab import packs as P

DATA = Path(__file__).resolve().parent / "sample_data"

#: **The eight signatures with a shipped fixture, and what each must say.**
#:
#: `(fixture, expected signature keys in order, whether the card asks)`. The
#: keys are a LIST because two of §02's rows are genuinely undecidable from a
#: matrix, and a classifier that returned one name for them would be picking a
#: side while wearing a badge that says it has not.
SIGNATURE_FIXTURES = {
    "raw counts": ("genomics_expression.csv", [P.RAW_COUNTS], False),
    "estimated counts": ("genomics_estimated_counts.csv",
                         [P.ESTIMATED_COUNTS, P.FPKM], True),
    "CPM or TPM": ("genomics_cpm.csv", [P.CPM_OR_TPM], True),
    "TMM-scaled CPM": ("genomics_tmm_cpm.csv", [P.TMM_SCALED_CPM], False),
    "FPKM": ("genomics_fpkm.csv", [P.FPKM, P.ESTIMATED_COUNTS], True),
    "VST": ("genomics_vst.csv", [P.VST], False),
    "microarray": ("genomics_microarray.csv", [P.MICROARRAY], False),
    "log-ratio or z-scored": ("wide_assay.csv", [P.LOG_RATIO], False),
}

#: The target each fixture is driven with, so Explore opens and the card is
#: reachable through the door rather than only through the route.
TARGETS = {"wide_assay.csv": "responder"}

#: NOT COVERED BY A FIXTURE, said out loud — a sweep that reports only what it
#: covered has not reported its coverage.
#:
#: **rlog, §02 row 7.** `genomics_vst.csv.md` records why: a true rlog matrix
#: needs a floor BELOW zero and none is shipped. It is not declared unreachable
#: — the branch exists and `test_the_rlog_branch_is_reachable_without_a_fixture`
#: drives it on a constructed frame — but a constructed frame is a weaker
#: instrument than a fixture and saying so is the point of this list.
#:
#: **A REAL GENE-ID VOCABULARY.** Every shipped matrix names its columns
#: `gene_0001`, and only `genomics_microarray.csv` carries identifiers a real
#: export would — Affymetrix probe-set ids. §01's Ensembl, RefSeq, Illumina and
#: Agilent patterns are implemented and none of them is exercised by a fixture.
#:
#: **A TRANSPOSED MATRIX.** §01's own convention is genes in rows, so the
#: commonest real export is the one shape none of these has. That is
#: `orientation.py`'s question rather than this reading's, and it fires first —
#: but nothing here checks that the two compose.
#:
#: **A MATRIX WITH NO COVARIATE.** All eight carry `age` beside the features, so
#: the block selector's drop path is exercised on every one of them and its
#: no-op path only incidentally.
#:
#: **SINGLE-CELL.** The pack opens by putting it out of scope and requiring it
#: to be DETECTED AND REFUSED — zero fraction above 80–90% with more than a
#: thousand columns and a median non-zero count of three or less. No fixture is
#: single-cell and this reading does not implement that refusal; it would call
#: such a matrix raw counts.
SHAPES_NOT_COVERED = [
    "rlog (§02 row 7) — no shipped fixture has a floor below zero; the branch "
    "is driven on a constructed frame instead",
    "a real gene-ID vocabulary — Ensembl, RefSeq, Illumina and Agilent are "
    "implemented and only Affymetrix is exercised",
    "a transposed (genes-in-rows) matrix — orientation.py's question, and "
    "nothing here checks the two compose",
    "a matrix with no covariate column beside the features — all eight carry "
    "`age`, so the block selector's no-op path is only incidentally covered",
    "single-cell — the pack requires it detected and REFUSED, no fixture is "
    "single-cell, and this reading would call one raw counts",
]


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


# ═══════════ 1 · EVERY SIGNATURE FIRES, ON A FIXTURE ═══════════

@pytest.mark.parametrize("signature", sorted(SIGNATURE_FIXTURES))
def test_each_signature_fires_on_its_fixture(signature):
    """§02's table, row by row, against the matrix built for that row.

    The assertion is on the KEYS and their ORDER, not on a substring of the
    label: where two names come back the order is the evidence talking, and a
    test that read the sentence could not tell a reversed pair from a correct
    one.
    """
    fixture, expected, asks = SIGNATURE_FIXTURES[signature]
    card = P.data_type_card(load(fixture))
    assert card is not None and card["read"], (
        f"{fixture} produced no reading at all")
    reading = card["classification"]
    assert reading["keys"] == expected, (
        f"{fixture} reads as {reading['keys']}, not {expected}")
    assert reading["requires_input"] is asks, (
        f"{fixture}: requires_input is {reading['requires_input']}, "
        f"expected {asks} — §02 says ASK for the rows that overlap and the "
        f"app may not silently pick")
    assert reading["coaching"], "a classification with no coaching is a label"


def test_the_two_undecidable_rows_return_both_names_and_ask():
    """§00: *"Formally undecidable from the matrix alone."*

    The sharpest thing this classifier does is decline twice. `genomics_cpm.csv`
    is CPM or TPM and no test on the numbers separates them; the estimated-count
    and FPKM fixtures are separated only by a statistic §02 does not name. Both
    come back with the question the user can actually answer.
    """
    cpm = P.data_type_card(load("genomics_cpm.csv"))["classification"]
    assert cpm["keys"] == [P.CPM_OR_TPM]
    assert "CPM" in cpm["label"] and "TPM" in cpm["label"], cpm["label"]
    assert cpm["question"] and "pipeline" in cpm["question"]

    for fixture, first in (("genomics_estimated_counts.csv", P.ESTIMATED_COUNTS),
                           ("genomics_fpkm.csv", P.FPKM)):
        reading = P.data_type_card(load(fixture))["classification"]
        assert len(reading["keys"]) == 2, reading["keys"]
        assert reading["keys"][0] == first, (
            f"{fixture} ranks {reading['keys']} — the library-size spread is "
            f"the separator and it has been read the wrong way round")
        assert reading["question"], f"{fixture} picked without asking"


def test_the_negative_rule_is_a_rule_and_not_a_reading():
    """§02's one stated rule: *"any negative value rules out raw counts, CPM,
    TPM and FPKM."*

    Asserted as a PROPERTY over every negative-valued frame this file can build
    rather than on the one fixture that has negatives, because a rule checked
    against one example is a coincidence. The constructed frames put negatives
    into matrices whose other statistics say `raw counts` and `CPM` as loudly
    as they can, so the rule has something to overrule.

    **And what comes back is silence, which is the sharper result.** A count
    matrix with one negative cell is not any of the nine shapes: the rule takes
    four rows away and the remaining five do not fit a matrix whose ceiling is
    twenty-six thousand. The app says it read the numbers and matched nothing.
    That is the governing rule doing exactly what it says — *the app may be
    silent* — where the alternative is naming the second-best row.
    """
    counts = load("genomics_expression.csv")
    genes = [c for c in counts.columns if c.startswith("gene_")]

    def keys_of(frame):
        card = P.data_type_card(frame)
        return card["classification"]["keys"] if card["read"] else []

    # A count matrix, one cell of which is negative. Every other reading —
    # whole numbers, a ceiling past ten thousand, varying library sizes — still
    # says raw counts, and the rule overrules all of it.
    spoiled = counts.copy()
    spoiled.loc[spoiled.index[0], genes[0]] = -1
    assert P.RAW_COUNTS not in keys_of(spoiled), (
        f"one negative cell and the app still reads raw counts: "
        f"{keys_of(spoiled)}")

    # A CPM matrix, likewise. Every sample still sums to a million.
    spoiled_cpm = load("genomics_cpm.csv")
    spoiled_cpm.loc[spoiled_cpm.index[0], genes[0]] = -0.5
    ruled = keys_of(spoiled_cpm)
    assert P.CPM_OR_TPM not in ruled and P.FPKM not in ruled, ruled

    # AND THE RULE IS NAMED WHERE IT ACTED. A reading that quietly dropped four
    # rows would be indistinguishable from a reading that never considered them.
    card = P.data_type_card(spoiled)
    row = next(r for r in card["evidence"]["rows"] if r["key"] == "negatives")
    assert "rules out" in row["statement"], row


def _rlog_frame() -> pd.DataFrame:
    """A VST matrix with its floor pushed below zero — §02 row 7's shape.

    Constructed rather than shipped, and named in `SHAPES_NOT_COVERED` for it.
    DESeq2's rlog shrinks low counts toward a per-gene mean, so the zeros land
    on a shared SMALL NEGATIVE floor rather than on a positive one; everything
    else about the matrix is what VST leaves behind.
    """
    frame = load("genomics_vst.csv")
    genes = [c for c in frame.columns if c.startswith("gene_")]
    frame[genes] = frame[genes].to_numpy(dtype=float) - 2.4
    return frame


def test_the_rlog_branch_is_reachable_without_a_fixture():
    """Row 7 of nine, handled rather than declared unreachable.

    And the separator is checked in BOTH directions, which is the half that
    matters: a rule that only ever sees the positive case is a rule nothing has
    tried to break. `wide_assay.csv` is the other side — negatives too, a
    ceiling in the same band — and it must not read as rlog.
    """
    reading = P.data_type_card(_rlog_frame())["classification"]
    assert reading["keys"] == [P.RLOG], reading["keys"]
    assert reading["confidence"] == "medium", (
        "no shipped fixture is rlog, so the reading may not claim high "
        "confidence in it")
    other = P.data_type_card(load("wide_assay.csv"))["classification"]
    assert other["keys"] == [P.LOG_RATIO], other["keys"]


# ═══════════ 2 · THE CAPABILITY MATRIX ═══════════

def test_every_signature_has_a_complete_capability_matrix():
    """All nine, not the eight with a fixture. §02's Presentation asks for a
    matrix showing *"which downstream steps are now enabled, disabled, or
    require input"*, and a signature whose row is missing is a card that would
    render a hole on the day somebody uploads that matrix."""
    for key in P.SIGNATURES:
        rows = P.capability_rows(key)
        assert [r["key"] for r in rows] == list(P.CAPABILITIES), key
        for row in rows:
            assert row["state"] in (P.ENABLED, P.DISABLED, P.REQUIRES_INPUT)
            assert row["evidence_status"] in P.EVIDENCE_STATUSES, (key, row)
            assert row["source"].startswith("research/GENOMICS_PACK.md#")


def test_a_capability_row_can_be_rendered_from_itself():
    """**`GUIDED-207`, at the granularity it was filed at.**

    The row it names is *a payload field that NAMES what an interface must
    construct instead of DESCRIBING it* — trap #1 at FIELD granularity, whose
    detector is not *does anything import this* but **could an interface build
    a control from this alone**. A capability matrix is the exact shape:
    `disabled_because: "count_model"` makes the page hold a copy of the rule,
    the two drift, and the copy the user reads is the untested one.

    So this asks that question of every row. **Structurally, not by length** —
    the first draft asserted `len(because) > 80` and that is a tuned number
    dressed as a rule: it went red on *"RMA already normalizes across arrays.
    Scaling again would be normalizing twice"*, which is a complete and correct
    sentence at seventy-nine characters, and the only ways out were to pad the
    prose or to lower the bound. Both are the test editing the product to suit
    itself.

    What separates a sentence from a name is not size. It is that a name is a
    token from a vocabulary the reader has to look up, and the vocabularies are
    right here to check against.
    """
    vocabulary = (set(P.CAPABILITIES) | set(P.SIGNATURES)
                  | {P.ENABLED, P.DISABLED, P.REQUIRES_INPUT})
    for key in P.SIGNATURES:
        for row in P.capability_rows(key):
            where = f"{key}/{row['key']}"
            assert row["label"] and " " in row["label"], where
            assert row["state_label"] and " " in row["state_label"], (
                f"{where}: `state` is a code and the page must not translate "
                f"it — a phrase travels beside it")
            because = (row["because"] or "").strip()
            # A NAME IS A TOKEN FROM A VOCABULARY THE READER MUST LOOK UP. Both
            # vocabularies this payload uses are checkable, so check them.
            assert because not in vocabulary, (
                f"{where}: `because` is {because!r}, which is a NAME from this "
                f"payload's own vocabulary. That is `GUIDED-207` exactly — the "
                f"page would have to hold the sentence, and the copy a user "
                f"reads would be the one nothing tests.")
            assert because.endswith((".", "!")), (
                f"{where}: `because` does not end a sentence: {because!r}")
            assert len(because.split()) >= 8, (
                f"{where}: `because` is {len(because.split())} words. A clause "
                f"this short is a label with punctuation, and the page would "
                f"still have to explain it.")
            # NO CAPITAL-LETTER CHECK. The first draft had one and it went red
            # on *"log2 of the value plus an offset is the usual next step…"* —
            # which is right, because `log2` is a function name and capitalizing
            # it would be the test correcting the domain. Sentence case is not
            # what separates a sentence from a name here.


def test_what_the_research_closes_is_closed_on_the_fixture_that_carries_it():
    """The five branches of §02's coaching, each on the matrix it is about.

    This is the assertion the whole card exists for. Not *does a row exist* —
    *does the row say what the research says*, on the data that triggers it.
    """
    def states(fixture):
        card = P.data_type_card(load(fixture))
        return {r["key"]: r["state"] for r in card["capabilities"]["rows"]}

    # Raw counts — the only input a count model can estimate precision from.
    assert states("genomics_expression.csv")["count_model"] == P.ENABLED

    # TPM/CPM/FPKM — the negative-binomial route is closed, and feeding them to
    # a count model runs silently while its p-values are wrong.
    for fixture in ("genomics_cpm.csv", "genomics_tmm_cpm.csv",
                    "genomics_fpkm.csv"):
        assert states(fixture)["count_model"] == P.DISABLED, fixture

    # FPKM specifically — not comparable across samples EVEN IN PRINCIPLE.
    # This is the one row that separates FPKM from its two siblings, so it is
    # asserted against them rather than alone.
    assert states("genomics_fpkm.csv")["cross_sample_comparison"] == P.DISABLED
    assert states("genomics_tmm_cpm.csv")["cross_sample_comparison"] == P.ENABLED

    # VST — never the input to a DE test, and that covers the Gaussian route
    # too, because limma is a DE test.
    vst = states("genomics_vst.csv")
    assert vst["count_model"] == P.DISABLED
    assert vst["gaussian_workflow"] == P.DISABLED
    assert vst["pca_and_clustering"] == P.ENABLED

    # Microarray — the whole count toolchain does not apply; limma is the tool.
    array = states("genomics_microarray.csv")
    assert array["count_model"] == P.DISABLED
    assert array["gaussian_workflow"] == P.ENABLED


def test_the_research_sentences_travel_with_the_rows_that_carry_them():
    """The citations §02 names, in the sentence a person reads.

    A capability row that says *"not comparable across samples"* and does not
    say who established it is an assertion without a record — and this is the
    surface where the pack is most likely to be doubted, because the claim is
    counter-intuitive.
    """
    fpkm = {r["key"]: r for r in
            P.data_type_card(load("genomics_fpkm.csv"))["capabilities"]["rows"]}
    because = fpkm["cross_sample_comparison"]["because"]
    assert "Wagner" in because and "Dillies" in because, because
    assert "even in principle" in because, because

    counts = {r["key"]: r for r in
              P.data_type_card(load("genomics_expression.csv"))
              ["capabilities"]["rows"]}
    assert "measurement precision" in counts["count_model"]["because"]

    cpm = {r["key"]: r for r in
           P.data_type_card(load("genomics_cpm.csv"))["capabilities"]["rows"]}
    assert "silently" in cpm["count_model"]["because"], (
        "the sharpest fact about a normalized matrix in a count model is that "
        "NOTHING GOES WRONG visibly")


# ═══════════ 3 · WHAT THE CARD SERVES ═══════════

def test_no_list_the_card_serves_is_cut_and_every_one_says_its_bound():
    """**`GUIDED-209`.** *A server-side list cut to a literal bound before it is
    served, with nothing saying so.*

    Both of this card's lists are whole, and both carry `n` and `showing` so a
    later loop that starts truncating has to move a number a reader can see.
    Asserted as `showing == n` rather than as *there is no slice*, because the
    second is a claim about the file and this is a claim about the payload.
    """
    for fixture, _keys, _asks in SIGNATURE_FIXTURES.values():
        card = P.data_type_card(load(fixture))
        for name in ("evidence", "capabilities"):
            block = card[name]
            assert block["n"] == len(block["rows"]), (fixture, name)
            assert block["showing"] == block["n"], (
                f"{fixture}: the {name} list is cut to {block['showing']} of "
                f"{block['n']} and the card does not say so")
        assert len(card["capabilities"]["rows"]) == len(P.CAPABILITIES)


def test_the_percent_integer_the_card_shows_is_the_one_it_means():
    """**The measured trap, decided rather than left ambiguous.**

    Over ALL cells, *% integer* reads about 15% on the CPM, FPKM and VST
    matrices — because a zero is a whole number and those matrices are 15%
    zeros. That number describes the zeros. The card shows the reading over the
    NON-ZERO cells and says on the row itself which one it is and why the other
    is misleading.
    """
    card = P.data_type_card(load("genomics_cpm.csv"))
    row = next(r for r in card["evidence"]["rows"] if r["key"] == "integrality")
    measured = card["measured"]
    assert measured["pct_integer"] < 0.01, measured["pct_integer"]
    assert 0.1 < measured["pct_integer_all"] < 0.2, measured["pct_integer_all"]
    assert "non-zero" in row["value"]
    assert "NON-ZERO" in row["statement"] and "zero is a whole" in row["statement"]


def test_a_library_size_over_a_centred_matrix_is_not_reported():
    """Trap 9 — *returning a value where you should return nothing.*

    `wide_assay.csv` is centred on zero, so its row sums average almost nothing
    and their coefficient of variation reads 900%. That number is arithmetic and
    it is about nothing. It is withheld, with the reason in a sentence rather
    than by leaving a key out — an absent key and a question nobody asked look
    the same.
    """
    finding = P._genomics_data_type(load("wide_assay.csv"))
    assert finding["params"]["library_size_cv"] is None
    assert "negative" in finding["params"]["library_size_cv_note"]
    keys = {r["key"] for r in
            P.data_type_card(load("wide_assay.csv"))["evidence"]["rows"]}
    assert "library_size" not in keys

    counts = P._genomics_data_type(load("genomics_expression.csv"))
    assert counts["params"]["library_size_cv"] == pytest.approx(0.273, abs=0.01)


def test_the_covariate_beside_the_matrix_is_named_where_it_is_dropped():
    """The block selector's whole risk, made visible.

    `age` decides the maximum of the VST and microarray matrices — 79 against
    14.2 — and barely moves a row total, so it has to come out or the reading is
    of a matrix that does not exist. It comes out **named**: if one of those
    columns is a gene, the classification above it is wrong and this row is
    where a reader sees it.
    """
    for fixture in ("genomics_vst.csv", "genomics_microarray.csv"):
        card = P.data_type_card(load(fixture))
        assert card["block"]["excluded"] == ["age"], fixture
        row = next(r for r in card["evidence"]["rows"] if r["key"] == "excluded")
        assert "age" in row["value"]
        assert "wrong" in row["statement"]
    # AND IT DROPS NOTHING WHERE THERE IS NO CONTINUOUS MAJORITY TO BE OUTSIDE
    # OF. Raw counts are integers too, so the rule has nothing to separate and
    # the honest answer is to keep everything.
    counts = P.data_type_card(load("genomics_expression.csv"))
    assert counts["block"]["excluded"] == []
    assert counts["block"]["n_excluded"] == 0


# ═══════════ 4 · WHERE IT MUST SAY NOTHING ═══════════

#: Tables the genomics lens does not describe. The metabolomics panel is the
#: near miss and the reason the zero clause exists: 395 continuous columns with
#: varying totals is the estimated-counts row's shape exactly, and a matrix with
#: no zero anywhere in it is not a transcript quantification.
NOT_EXPRESSION = ("metabolomics_untargeted.csv", "survey_instrument.csv")


@pytest.mark.parametrize("fixture", NOT_EXPRESSION)
def test_the_reader_declines_rather_than_naming_the_nearest_shape(fixture):
    """Guard #2, on the surface where breaking it is most expensive.

    *A pack that fires on the wrong data asserts something false in the one
    place the app has promised it never will — and it does so authoritatively.*
    A classification is the most authoritative thing this pack says, so an
    unmatched matrix comes back as an ANSWER that names no shape, and the
    finding is withheld entirely.
    """
    card = P.data_type_card(load(fixture))
    assert card is not None, f"{fixture} is wide enough to be read"
    assert card["read"] is False, (
        f"{fixture} was classified as "
        f"{card.get('classification', {}).get('keys')}")
    assert "not going to name the nearest one" in card["reason"]
    assert P._genomics_data_type(load(fixture)) is None, (
        f"the {fixture} reading produced a finding, so the pack asserts a data "
        f"type for a table it could not classify")


@pytest.mark.parametrize("fixture", ("clinic_visits.csv", "nhanes_dietary.csv"))
def test_a_narrow_table_produces_no_card_at_all(fixture):
    """Not the same answer as *"I read it and it matched nothing"*, and the
    difference is worth keeping: there is no matrix here to read."""
    assert P.data_type_card(load(fixture)) is None


def test_the_genomics_lens_is_no_longer_refused_on_a_matrix_that_is_not_counts():
    """**The defect this part found, and it is in the mechanism built to catch a
    false reading.**

    `contradiction()`'s third direction raised a 409 against any wide genomics
    table whose values are fractional: *"495 of its measurement columns are not
    counts … Counts and concentrations are different objects."* **Six of the
    nine shapes §02 describes for an expression matrix are non-integer.** So the
    premise was false, and it blocked the genomics lens on seven of the eight
    genomics fixtures in this tree — which is to say the pack could not be
    reached at all on the matrices it was built this loop to read.

    The claim now needs the reader to have found no signature AT ALL. Both
    directions are asserted, because a fix that silenced the check everywhere
    would pass the first half of this on its own.
    """
    for fixture, _keys, _asks in SIGNATURE_FIXTURES.values():
        assert P.contradiction(load(fixture), [P.GENOMICS]) is None, (
            f"{fixture} is one of §02's nine shapes and the app tells the user "
            f"their genomics lens is probably wrong")
    # AND IT STILL FIRES WHERE IT SHOULD. 395 continuous columns, no zero
    # anywhere, and nothing in §02's table matches.
    still = P.contradiction(load("metabolomics_untargeted.csv"), [P.GENOMICS])
    assert still is not None
    assert still["kind"] == "stated_genomics_but_values_are_not_counts"
    assert "not any of the other eight" in still["message"]


# ═══════════ 5 · IT REACHES A PERSON ═══════════

def _driven(client, fixture):
    """Upload, answer the lens and the target, exactly as a user does."""
    with (DATA / fixture).open("rb") as handle:
        pid = client.post("/project", files={
            "file": (fixture, handle, "text/csv")}).json()["id"]
    for kind, payload in (("set_lens", {"lens": [P.GENOMICS]}),
                          ("set_target",
                           {"column": TARGETS.get(fixture, "condition")})):
        answered = client.post(f"/project/{pid}/decision",
                               json={"kind": kind, "payload": payload})
        assert answered.status_code == 200, (fixture, kind, answered.text[:400])
    return pid


@pytest.mark.parametrize("signature", sorted(SIGNATURE_FIXTURES))
def test_every_signature_reaches_an_upload(signature):
    """`GUIDED-058`'s rule: the check is not *does something import this*, it is
    *does an upload reach it.* Driven through the API — a file, a lens answer, a
    target — on all eight."""
    from fastapi.testclient import TestClient

    from turbotab import api

    fixture, expected, _asks = SIGNATURE_FIXTURES[signature]
    client = TestClient(api.app)
    pid = _driven(client, fixture)

    served = client.get(f"/project/{pid}").json()["findings"]
    reading = [f for f in served if f["id"] == "pack::genomics::data_type"]
    assert len(reading) == 1, (
        f"{fixture}: the data-type reading is not in the served findings — "
        f"{[f['id'] for f in served if f['source'] == 'pack']}")
    assert reading[0]["params"]["signatures"] == expected
    assert reading[0]["evidence"]["source"].startswith(
        "research/GENOMICS_PACK.md#")

    card = client.get(f"/project/{pid}/genomics/data_type")
    assert card.status_code == 200, card.text[:300]
    assert card.json()["classification"]["keys"] == expected


def test_the_card_is_refused_before_the_lens_is_answered():
    """A 409, and it says why rather than 404-ing.

    What a number IS is a claim about the assay that produced it. The app does
    not infer the field from column names — `wide_assay.csv` is 45 continuous
    columns centred on zero and nothing in them says whether they are
    expression, spectra or sensor readings.
    """
    from fastapi.testclient import TestClient

    from turbotab import api

    client = TestClient(api.app)
    with (DATA / "genomics_cpm.csv").open("rb") as handle:
        pid = client.post("/project", files={
            "file": ("genomics_cpm.csv", handle, "text/csv")}).json()["id"]
    refused = client.get(f"/project/{pid}/genomics/data_type")
    assert refused.status_code == 409, refused.status_code
    assert "lens" in refused.json()["detail"]


@pytest.mark.parametrize("signature", sorted(SIGNATURE_FIXTURES))
def test_the_card_renders_on_the_page(signature):
    """**`GUIDED-142`'s class, asked before it can recur.**

    Five packs and eighteen detectors were computed correctly, served correctly
    and rendered nowhere. So this drives the page's own controller under node
    against responses captured from a real API drive, and reads the card back
    out of the DOM.

    It asserts on STRUCTURE — the classification label, the capability keys as
    attributes, the badge classes — rather than on a substring of a sentence,
    for `FEATURE_PARITY.md`'s reason: a substring of a message is a wildcard
    wearing an assertion's clothes.
    """
    from fastapi.testclient import TestClient

    from turbotab import api
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")

    fixture, expected, _asks = SIGNATURE_FIXTURES[signature]
    client = TestClient(api.app)
    pid = _driven(client, fixture)

    project = client.get(f"/project/{pid}").json()
    card = client.get(f"/project/{pid}/genomics/data_type").json()
    routes = {
        f"/project/{pid}": project,
        f"/project/{pid}/genomics/data_type": card,
        f"/project/{pid}/interview?step=data":
            client.get(f"/project/{pid}/interview?step=data").json(),
        f"/project/{pid}/interview?step=explore":
            client.get(f"/project/{pid}/interview?step=explore").json(),
        f"/project/{pid}/evidence/missingness": {"cards": []},
        f"/project/{pid}/capabilities":
            client.get(f"/project/{pid}/capabilities").json(),
    }
    out = PH.run(
        "__emit({box: (__harness.html('genomicsBox') || ''),"
        " out: (__harness.html('gxOut') || '')});",
        routes=routes, search=f"?project={pid}")
    html = out["out"]
    assert html, (
        "the genomics box rendered nothing at all — the card is computed, "
        "served, and invisible to a person")
    assert out["box"], "the card is not attached to the box the page declares"

    label = card["classification"]["label"]
    assert label in html, f"the classification is not on the page: {html[:400]}"

    # EVERY CAPABILITY ROW IS THERE, addressed by key, and carrying the state
    # the server decided. A matrix missing a row would read as a shorter shelf.
    rendered = dict(re.findall(
        r'data-cap-key="([^"]+)" data-cap-state="([^"]+)"', html))
    assert rendered == {r["key"]: r["state"]
                        for r in card["capabilities"]["rows"]}, rendered

    # AND THE SENTENCES, because the row is the point and the chip is not.
    for row in card["capabilities"]["rows"]:
        assert row["state_label"] in html, row["key"]

    # THE BOUND IS STATED (`GUIDED-209`) — both lists say how many of how many.
    assert 'data-cap-for="genomics-capabilities"' in html
    assert 'data-cap-of="' + str(card["capabilities"]["n"]) + '"' in html
    assert 'data-cap-for="genomics-evidence"' in html

    # THE BADGE TRAVELS. A pack claim without one is the uniform confidence
    # `DOMAIN_SCIENCE.md` §01.1 exists to end.
    statuses = set(re.findall(r'class="badge (\w+)"', html))
    assert {r["evidence_status"].lower()
            for r in card["capabilities"]["rows"]} <= statuses, statuses


def test_the_page_shows_no_genomics_card_under_another_lens():
    """A pack changes what is drawn. On a table nobody described as genomic the
    surface is not there, rather than there and empty."""
    from fastapi.testclient import TestClient

    from turbotab import api
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")

    client = TestClient(api.app)
    with (DATA / "metabolomics_untargeted.csv").open("rb") as handle:
        pid = client.post("/project", files={
            "file": ("metabolomics_untargeted.csv", handle,
                     "text/csv")}).json()["id"]
    for kind, payload in (("set_lens", {"lens": [P.METABOLOMICS]}),
                          ("set_target", {"column": "responder"})):
        assert client.post(f"/project/{pid}/decision",
                           json={"kind": kind, "payload": payload}
                           ).status_code == 200

    project = client.get(f"/project/{pid}").json()
    out = PH.run(
        "__emit({out: (__harness.html('gxOut') || ''),"
        " box: (__harness.html('genomicsBox') || '')});",
        routes={f"/project/{pid}": project,
                f"/project/{pid}/interview?step=data":
                    client.get(f"/project/{pid}/interview?step=data").json(),
                f"/project/{pid}/interview?step=explore":
                    client.get(f"/project/{pid}/interview?step=explore").json(),
                f"/project/{pid}/evidence/missingness": {"cards": []},
                f"/project/{pid}/capabilities":
                    client.get(f"/project/{pid}/capabilities").json()},
        search=f"?project={pid}")
    assert not out["out"], out["out"][:300]

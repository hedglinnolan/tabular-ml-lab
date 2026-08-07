"""L43-B · every §A5/§B6 hit that was filed rather than fixed, as a red test.

`LOOP.md` §06's first check, and `AGENT_ONBOARD.md` §08's first: **a hit that
becomes prose is a hit that gets forgotten.** So each of the twenty-one
findings this loop filed and did not fix is a `pytest.mark.xfail(strict=True)`
naming its row.

**Why strict.** A non-strict xfail is a comment with a decorator on it. Strict
means the test fails *in the other direction* the moment the code is fixed and
the row is not closed — so the ledger and the tree cannot drift apart silently.
`GUIDED-119` and `GUIDED-138` are the precedent; `GUIDED-138`'s is currently
red and doing its job.

**These are not assertions of what the fix should be.** Each one states the
defect as a condition on shipped code, quoted from the audit's own
`failing_assertion`. Where the fix is a judgment — how to phrase a disclosure,
whether to gate or to compute — the row's `act` carries the registry's
requirement and this file carries only the fact that the defect is still there.

The audit: **132 requirements checked across seven §A5/§B6 clusters, 40
candidate hits, 28 surviving adversarial refutation, 7 refuted, 27
uncheckable.** Two criticals were fixed this loop (`AUDIT-014`, `AUDIT-015`);
these are the rest.
"""
from __future__ import annotations

import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _source(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="ignore")


def _ledger():
    return {r["id"]: r for r in json.loads(
        (ROOT / "docs" / "turbotab" / "data" / "findings.json")
        .read_text(encoding="utf-8"))}


def test_every_row_this_file_names_is_open_and_every_open_one_is_named():
    """The two lists are one fact. A row closed without its xfail going away
    leaves a strict xfail that now passes — which pytest reports as a failure,
    correctly, and this test says why before that happens."""
    named = {m for m in MARKED}
    led = _ledger()
    missing = sorted(i for i in named if i not in led)
    assert not missing, f"these are marked here and not in the ledger: {missing}"
    closed = sorted(i for i in named if led[i]["status"] not in ("OPEN", "PARTIAL"))
    assert not closed, (
        f"these are marked xfail here and CLOSED in the ledger: {closed}. "
        f"Either the fix landed and the marker should go, or the row was "
        f"closed without one.")


MARKED = [
    # AUDIT-019 closed at L51-C. The seal's candidate-parameter count is now
    # taken over the frame `training.feature_frame` builds, so the identifier
    # columns the app refuses to encode are out of it — 45 rather than 344 on
    # `survey_instrument.csv`, with what left the count named and priced.
    # `turbotab/test_the_seal_counts_the_parameters_the_models_are_handed.py`.
    "AUDIT-022",
    "AUDIT-023",
    "AUDIT-025",
    # AUDIT-030 closed at L45-A2. The ruling was made at the L44 adjudication and
    # applied here: the Methods section no longer calls the held-out comparison a
    # validation one, and it states what selecting among N on those rows costs.
    # `tests/test_the_methods_section_names_the_set_it_compared_on.py`.
    "AUDIT-031",
    "AUDIT-032",
    "AUDIT-034",
    "AUDIT-035"
]


def test_audit_019_is_closed_and_its_regression_test_is_the_one_named_here():
    """`AUDIT-019` — fixed at L51-C, and no longer an xfail.

    Same shape as `AUDIT-030` below: a closed row keeping its strict marker is
    an xfail that now passes, which pytest reports as a failure nobody can
    read. What stays is the pointer.

    The defect was that `resolution.candidate_parameters` dropped the target
    and the group column and nothing else, while `training.feature_frame` had
    dropped `identifiers.excluded` since `GUIDED-108` — so the seal's methods
    sentence reported 344 candidate predictor parameters on
    `survey_instrument.csv` where the models are handed 45, and §A5.4's whole
    point is that this count is the input to Riley's minimum sample size.
    """
    row = _ledger()["AUDIT-019"]
    assert row["status"] == "FIXED", (
        f"AUDIT-019 is {row['status']} in the ledger and the code is fixed. "
        f"The fix and the row close together — the subagent that made the fix "
        f"is not the writer of findings.json, so this test is red on purpose "
        f"until the row is set: docs/turbotab/tools/ledger.py set AUDIT-019 "
        f"--status FIXED --test turbotab/"
        f"test_the_seal_counts_the_parameters_the_models_are_handed.py")
    assert "test_the_seal_counts_the_parameters_the_models_are_handed" in (
        row.get("test") or ""), (
        "AUDIT-019 is closed against a test this file cannot name")


@pytest.mark.xfail(strict=True, reason="AUDIT-022 — filed at L43-B, not fixed this loop")
def test_audit_022_the_generated_manuscript_lists_the_sample_size_as_a_strength_for_every():
    """The generated manuscript lists the sample size as a Strength for every N, with no criterion — and can print it as a strength and a limitation in the same section

    Where: `pages/10_Report_Export.py:1722`
    Registry: §A5.4 requires that sample-size adequacy come from Riley's criteria-based calculation for the candidate parameter count, prevalence and anticipated R². Placing an uncomputed N under a heading that asserts it is a methodological strength is an adequacy claim the app never evaluated. This is the AUDIT
    """
    row = _ledger()["AUDIT-022"]
    assert row["status"] in ("OPEN", "PARTIAL")
    pytest.fail(
        "AUDIT-022 is still open: " + row["item"][:160])

@pytest.mark.xfail(strict=True, reason="AUDIT-023 — filed at L43-B, not fixed this loop")
def test_audit_023_applying_feature_selection_overwrites_the_candidate_feature_list__so_t():
    """Applying feature selection overwrites the candidate feature list, so the sufficiency prose reports the KEPT count while calling them 'candidate predictors'

    Where: `pages/04_Feature_Selection.py:440`
    Registry: §A5.4, flagged ⚠: 'CANDIDATE PREDICTORS COUNT TOWARD SAMPLE SIZE EVEN IF THEY ARE LATER DROPPED. If you screen 40 variables and keep 8, you must size for 40 — data-driven selection consumes degrees of freedom whether or not it appears in the final model. This is the sample-size mistake PROBAST most 
    """
    row = _ledger()["AUDIT-023"]
    assert row["status"] in ("OPEN", "PARTIAL")
    pytest.fail(
        "AUDIT-023 is still open: " + row["item"][:160])

@pytest.mark.xfail(strict=True, reason="AUDIT-025 — filed at L43-B, not fixed this loop")
def test_audit_025_the_theory_reference_tells_the_user_the_feature_selection_page_offers():
    """The Theory Reference tells the user the Feature Selection page offers VIF-based filtering as one of its selection methods; no such method exists

    Where: `/Users/nhedglin/tabular-ml-lab/pages/11_Theory_Reference.py:1222-1223`
    Registry: Governing rule: the app may be SILENT and it may REFUSE, but it must never ASSERT SOMETHING FALSE. Check 5 of this cluster asks whether a collinearity-based drop exists under another name; the audit rule that absence is not a hit is explicitly conditioned on nothing in the app claiming it — here the
    """
    row = _ledger()["AUDIT-025"]
    assert row["status"] in ("OPEN", "PARTIAL")
    pytest.fail(
        "AUDIT-025 is still open: " + row["item"][:160])

def test_audit_030_is_closed_and_its_regression_test_is_the_one_named_here():
    """`AUDIT-030` — ruled at L44, applied at L45-A2, and no longer an xfail.

    The strict xfail above it is gone rather than left to pass, which is the
    whole point of `test_every_row_this_file_names_is_open_and_every_open_one_is_named`
    — a closed row keeping its marker is a strict xfail that now passes, which
    pytest reports as a failure and which nobody can read.

    What stays here is the pointer. The row's regression tests live in
    `tests/`, because the defect is Classic-door — `pages/06`, `ml/narrative_engine.py`
    and `utils/workflow_provenance.py` — and this file is the registry index
    rather than the test.
    """
    row = _ledger()["AUDIT-030"]
    assert row["status"] == "FIXED", (
        f"AUDIT-030 reopened and this file no longer marks it: {row['status']}")
    assert "test_the_methods_section_names_the_set_it_compared_on" in (
        row.get("test") or ""), (
        "AUDIT-030 is closed against a test this file cannot name")


@pytest.mark.xfail(strict=True, reason="AUDIT-031 — filed at L43-B, not fixed this loop")
def test_audit_031_the_manuscript_s_auto_generated_strengths_list_asserts_the_dataset_con():
    """The manuscript's auto-generated Strengths list asserts the dataset contained "no leakage candidates" on the strength of a numeric-only |r|>0.95 scan

    Where: `pages/02_EDA.py:540-554`
    Registry: §A5.5 Anti-patterns: "and — specific to EHR — using a variable recorded *because* the outcome happened (a 'palliative care consult' predicting death)." A variable of exactly that shape is either non-numeric (never scanned) or numeric with correlation far below 0.95 (never flagged). The registry's go
    """
    row = _ledger()["AUDIT-031"]
    assert row["status"] in ("OPEN", "PARTIAL")
    pytest.fail(
        "AUDIT-031 is still open: " + row["item"][:160])

@pytest.mark.xfail(strict=True, reason="AUDIT-032 — filed at L43-B, not fixed this loop")
def test_audit_032_running_the_leakage_diagnostic_marks_the_leakage_blocker_resolved__the():
    """Running the leakage diagnostic marks the leakage BLOCKER resolved; the report then calls it "addressed" and the manuscript drops the caveat, while the column is still a model feature

    Where: `pages/02_EDA.py:1575-1589`
    Registry: §A5.5 Anti-patterns names leakage predictors as a thing that must not be silently carried into a model. The governing rule is that the app may be silent and may refuse but must never assert something false: reporting a still-present leakage predictor under "Addressed observations" and in an "N were 
    """
    row = _ledger()["AUDIT-032"]
    assert row["status"] in ("OPEN", "PARTIAL")
    pytest.fail(
        "AUDIT-032 is still open: " + row["item"][:160])

@pytest.mark.xfail(strict=True, reason="AUDIT-034 — filed at L43-B, not fixed this loop")
def test_audit_034_the_recorded_reverse_coding_sentence_promises_a_flip_and_a_scoring_ste():
    """The recorded reverse-coding sentence promises a flip and a scoring step the app never performs, and it is exported into the manuscript

    Where: `turbotab/api.py:538-543`
    Registry: §B6's anti-pattern list and the governing rule. The app may be silent and may refuse, but must not assert something false. §B1.2's own framing (which this pack cites) is that reverse-coding comes from the codebook and matters because it changes the score; asserting the flip was applied when it was n
    """
    row = _ledger()["AUDIT-034"]
    assert row["status"] in ("OPEN", "PARTIAL")
    pytest.fail(
        "AUDIT-034 is still open: " + row["item"][:160])

@pytest.mark.xfail(strict=True, reason="AUDIT-035 — filed at L43-B, not fixed this loop")
def test_audit_035_the_purpose_question_tells_the_user_its_answer_decides_whether_a_scale():
    """The purpose question tells the user its answer decides whether a scale is scored or used item by item; nothing in the app reads it for that

    Where: `turbotab/purpose.py:71`
    Registry: §B6 Item-level vs scale-level [DISPUTED]: "for PREDICTION, item-level with penalization is reasonable and should be compared against the scale score in optimism-corrected internal validation — LET THE VALIDATION DECIDE. For INFERENCE about the construct, use the scale score and either correct for at
    """
    row = _ledger()["AUDIT-035"]
    assert row["status"] in ("OPEN", "PARTIAL")
    pytest.fail(
        "AUDIT-035 is still open: " + row["item"][:160])


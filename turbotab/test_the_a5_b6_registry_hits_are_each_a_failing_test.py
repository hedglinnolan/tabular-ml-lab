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
    "AUDIT-019",
    "AUDIT-020",
    "AUDIT-021",
    "AUDIT-022",
    "AUDIT-023",
    "AUDIT-024",
    "AUDIT-025",
    "AUDIT-026",
    "AUDIT-027",
    "AUDIT-028",
    "AUDIT-029",
    # AUDIT-030 closed at L45-A2. The ruling was made at the L44 adjudication and
    # applied here: the Methods section no longer calls the held-out comparison a
    # validation one, and it states what selecting among N on those rows costs.
    # `tests/test_the_methods_section_names_the_set_it_compared_on.py`.
    "AUDIT-031",
    "AUDIT-032",
    "AUDIT-033",
    "AUDIT-034",
    "AUDIT-035"
]


@pytest.mark.xfail(strict=True, reason="AUDIT-019 — filed at L43-B, not fixed this loop")
def test_audit_019_the_seal_s_methods_sentence_states_a_candidate_parameter_count_that_in():
    """The seal's methods sentence states a candidate-parameter count that includes the identifier columns the app refuses to give the model — 344 instead of 45 on the repo's own survey fixture

    Where: `turbotab/resolution.py:112`
    Registry: §A5.4: 'Compute Riley et al.'s minimum sample size for model development. Inputs: number of candidate predictor PARAMETERS — COUNT PARAMETERS, NOT VARIABLES.' The count that matters is the parameters the model may spend; the app's own badge for this function (turbotab/resolution.py:85-96, SOURCES['c
    """
    row = _ledger()["AUDIT-019"]
    assert row["status"] in ("OPEN", "PARTIAL")
    pytest.fail(
        "AUDIT-019 is still open: " + row["item"][:160])

@pytest.mark.xfail(strict=True, reason="AUDIT-020 — filed at L43-B, not fixed this loop")
def test_audit_020_events_per_variable_is_computed_over_raw_columns__so_a_5_level_factor():
    """Events-per-variable is computed over raw COLUMNS, so a 5-level factor counts as 1 parameter instead of 4 — and the resulting 'adequate signal' sentence is exported into the report

    Where: `ml/dataset_profile.py:367`
    Registry: §A5.4: 'number of candidate predictor PARAMETERS — count parameters, not variables; a 4-knot spline is 3 parameters, a 5-level factor is 4.' The app's own Guided-door implementation gets this right (turbotab/resolution.py:120-130 charges nunique−1) and badges it SETTLED against §A5.4, so this is a c
    """
    row = _ledger()["AUDIT-020"]
    assert row["status"] in ("OPEN", "PARTIAL")
    pytest.fail(
        "AUDIT-020 is still open: " + row["item"][:160])

@pytest.mark.xfail(strict=True, reason="AUDIT-021 — filed at L43-B, not fixed this loop")
def test_audit_021_the_app_states_epv____10_as__the_guideline__for_sample_size__in_both_d():
    """The app states EPV >= 10 as 'the guideline' for sample size, in both doors, against a [SETTLED] position that the rule is superseded

    Where: `ml/model_coach.py:416`
    Registry: §A5.4, marked [SETTLED that EPV≥10 is superseded]: 'The events-per-variable rule of 10 is a legacy heuristic that both under- and over-estimates requirements depending on prevalence and expected model strength; use the criteria-based calculation.' The repo's own design doc agrees — docs/turbotab/DOM
    """
    row = _ledger()["AUDIT-021"]
    assert row["status"] in ("OPEN", "PARTIAL")
    pytest.fail(
        "AUDIT-021 is still open: " + row["item"][:160])

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

@pytest.mark.xfail(strict=True, reason="AUDIT-024 — filed at L43-B, not fixed this loop")
def test_audit_024_the_classic_feature_selection_page_offers_univariable_p_value_screenin():
    """The Classic Feature Selection page offers univariable p-value screening and RFE-CV, both ON BY DEFAULT, and states none of the [SETTLED] objection anywhere in shipped code

    Where: `/Users/nhedglin/tabular-ml-lab/pages/04_Feature_Selection.py:170-178`
    Registry: §A5.5 Modeling practice: "Avoid univariable pre-screening of predictors by p-value. It is one of PROBAST's explicit high-risk-of-bias signals: it discards variables that matter only in combination, and it invalidates the p-values in the final model." [SETTLED]. And: "Avoid stepwise selection. It pro
    """
    row = _ledger()["AUDIT-024"]
    assert row["status"] in ("OPEN", "PARTIAL")
    pytest.fail(
        "AUDIT-024 is still open: " + row["item"][:160])

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

@pytest.mark.xfail(strict=True, reason="AUDIT-026 — filed at L43-B, not fixed this loop")
def test_audit_026_the_methods_section_asserts_n_fold_cross_validation_was_used_for_inter():
    """The Methods section asserts N-fold cross-validation was used for internal validation whenever the checkbox was ticked, even when no CV ran at all

    Where: `ml/narrative_engine.py:1040`
    Registry: §A5.5: 'Internal validation must resample the entire modeling pipeline — imputation, transformation, selection, tuning. Bootstrap optimism correction is the recommended default ...; repeated k-fold CV is acceptable. A single train/test split is the weakest option and is discouraged at typical clinic
    """
    row = _ledger()["AUDIT-026"]
    assert row["status"] in ("OPEN", "PARTIAL")
    pytest.fail(
        "AUDIT-026 is still open: " + row["item"][:160])

@pytest.mark.xfail(strict=True, reason="AUDIT-027 — filed at L43-B, not fixed this loop")
def test_audit_027_the__rank_them_for_me__panel_tells_the_user_that_what_is_actually_sele():
    """The 'Rank them for me' panel tells the user that what is actually selected is refitted inside each training fold — this door selects once over the training rows

    Where: `turbotab/selection.py:202`
    Registry: §A5.5 [SETTLED]: 'Internal validation must resample the entire modeling pipeline — imputation, transformation, selection, tuning.' Telling the researcher that selection is refitted inside each training fold is a claim that selection IS inside a resampling loop. It is not; there is one fit on one par
    """
    row = _ledger()["AUDIT-027"]
    assert row["status"] in ("OPEN", "PARTIAL")
    pytest.fail(
        "AUDIT-027 is still open: " + row["item"][:160])

@pytest.mark.xfail(strict=True, reason="AUDIT-028 — filed at L43-B, not fixed this loop")
def test_audit_028_every_imputation_declaration_writes__within_each_training_fold__into_t():
    """Every imputation declaration writes 'within each training fold' into the recorded methods sentence, in a door that fits once over the training rows

    Where: `turbotab/missingness.py:522`
    Registry: §A5.5 [SETTLED that the full pipeline must be inside the loop]: internal validation must resample imputation and transformation, and 'a single train/test split is the weakest option and is discouraged at typical clinical sample sizes'. A methods sentence saying the median was computed within each tr
    """
    row = _ledger()["AUDIT-028"]
    assert row["status"] in ("OPEN", "PARTIAL")
    pytest.fail(
        "AUDIT-028 is still open: " + row["item"][:160])

@pytest.mark.xfail(strict=True, reason="AUDIT-029 — filed at L43-B, not fixed this loop")
def test_audit_029_the_generated_methods_section_tells_the_reader_that_cross_validation_w():
    """The generated Methods section tells the reader that cross-validation was run on already-preprocessed data — the code re-fits preprocessing inside every fold and has since STATE-059

    Where: `ml/publication.py:1378-1381`
    Registry: §A5.5 [SETTLED]: internal validation must resample the entire pipeline — imputation, transformation, selection, tuning. The manuscript is the artifact a reviewer reads; asserting that the app's CV scored pre-preprocessed data misdescribes the analysis in the one document where the description is the
    """
    row = _ledger()["AUDIT-029"]
    assert row["status"] in ("OPEN", "PARTIAL")
    pytest.fail(
        "AUDIT-029 is still open: " + row["item"][:160])

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

@pytest.mark.xfail(strict=True, reason="AUDIT-033 — filed at L43-B, not fixed this loop")
def test_audit_033_the_task_type_detector_tells_the_user__in_both_doors__that_an_ordinal():
    """The task-type detector tells the user, in both doors, that an ordinal score should be modeled as regression — B6 marks that SETTLED wrong

    Where: `ml/triage.py:83-86`
    Registry: §B6 Coaching [SETTLED]: "For an ordinal outcome, use a cumulative link (proportional odds) model rather than a LINEAR MODEL ON THE SCORE or a dichotomization into 'responder/non-responder.' The PO model generalizes the Wilcoxon and Kruskal-Wallis tests while allowing covariate adjustment, handles ar
    """
    row = _ledger()["AUDIT-033"]
    assert row["status"] in ("OPEN", "PARTIAL")
    pytest.fail(
        "AUDIT-033 is still open: " + row["item"][:160])

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


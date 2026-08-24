"""Drive 9 · the findings the re-drive opened, and what each repair has to hold.

`docs/audit/DRIVE9_RECHECK.md`. One governing rule across all of them: **the app
may be silent and it may refuse, but it must never assert something false.**

- **D9-01** (`DRIVE-072`, critical) — a consensus selection screened 19 numeric
  candidates, kept 6, carried 8 non-numeric predictors through, and reduced 27
  predictors to 14. The Methods draft said *"All 14 candidate predictors were
  retained for final modeling"* while the Evidence Map one panel away said
  *"consensus: 27 → 14 predictors"*. `_resolve_workflow_feature_counts` read
  `logged_steps['Feature Selection'][-1]`, which is the *Apply* entry — the
  ledger files both under one step name — and the Apply entry carries no
  before/after counts, so both collapsed to the post-apply list.
- **D9-03** (`DRIVE-073`, high) — applying a selection clears `dataset_profile`
  and nothing recomputed it, so page 06 lost the class-imbalance card, the
  rebalancing control and the model-suitability badges without a word on screen,
  and page 10's manifest read *"Dataset profile: Not computed"*.
- **D9-04** — TRIPOD item 9 (*missing data*) ticked by a target dtype recode
  that left every blank blank, and 15a + 19a both ticked by one explainability
  line. A tick must come from the section it certifies.
- **D9-05** — the draft counted *"the 1 test"* while its own Evidence Map said
  *"2 test(s)"*: distinct comparisons against recorded runs, neither named.
- **D9-06** — Limitations spliced coach cards as `finding.: rationale`, and the
  validator's no-coaching-language check passed over *"A reviewer would question
  why the more complex model was selected."*
- **D9-09** — *"Found 1 worth checking, 1 note."* above *"Also worth a look (2)"*.
- **D9-10** — *"Selected 6 features using Lasso Regression, rfe"*: one display
  label and one raw method key in one sentence.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def clean_session():
    st.session_state.clear()
    yield
    st.session_state.clear()


def _source(relative: str) -> str:
    return (REPO / relative).read_text(encoding="utf-8")


# ── D9-01 · the counts come from the selection record ────────────────────────
#
# The drive's own numbers: 27 predictors offered, 19 of them numeric and
# therefore rankable, 6 met the consensus threshold, 8 non-numeric predictors
# carried through, 14 in the modeling set.

_SCREENING_ENTRY = {
    "step": "Feature Selection",
    "action": "Selected 6 features using LASSO, RFE-CV",
    "details": {"methods": ["lasso", "rfe"], "methods_completed": ["lasso", "rfe"],
                "n_features_before": 19, "n_features_after": 6,
                "consensus_threshold": 2},
}
# The ledger files this under `Feature Selection` too, and it is LAST.
_APPLIED_ENTRY = {
    "step": "Feature Selection",
    "action": "Applied consensus feature selection",
    "details": {"method": "consensus", "n_features_selected": 14,
                "n_consensus_ranked": 6,
                "carried_through_unranked": ["gender", "meds_chol",
                                             "imputed_weight", "imputed_height",
                                             "imputed_bmi", "imputed_waist",
                                             "imputed_bp_sys", "imputed_bp_di"]},
}


#: 19 rankable numeric predictors, 8 non-numeric ones the methods never see.
_NUMERIC = [f"num_{i:02d}" for i in range(19)]
_CATEGORICAL = [f"cat_{i:02d}" for i in range(8)]
_OFFERED = _NUMERIC + _CATEGORICAL          # 27 predictors offered at upload
_APPLIED = _NUMERIC[:6] + _CATEGORICAL      # 6 consensus + 8 carried = 14


def _applied_provenance():
    """Provenance as page 04 leaves it after Run then Apply."""
    from utils.workflow_provenance import WorkflowProvenance

    prov = WorkflowProvenance()
    prov.record_upload("meds_hbp", "classification", _OFFERED, 21849)
    prov.record_feature_selection(
        method="consensus", n_before=19, n_after=6,
        features_kept=list(_NUMERIC[:6]),
        consensus_methods=["lasso", "rfe"],
        candidates_screened=list(_NUMERIC),
    )
    prov.record_feature_selection(
        method="consensus", n_before=19, n_after=14,
        features_kept=list(_APPLIED),
        consensus_methods=["lasso", "rfe"],
    )
    return prov


def test_d9_01_the_apply_entry_does_not_overwrite_the_screening_counts():
    """`fs_entries[-1]` is the Apply entry; it carries no before/after counts."""
    from ml.publication import _resolve_workflow_feature_counts

    st.session_state["workflow_provenance"] = _applied_provenance()
    st.session_state["selected_features"] = list(_APPLIED)

    counts = _resolve_workflow_feature_counts(
        list(_APPLIED),
        logged_steps={"Feature Selection": [_SCREENING_ENTRY, _APPLIED_ENTRY]},
        # page 04's apply overwrote this with the post-selection list
        data_config={"feature_cols": list(_APPLIED)},
    )

    assert counts["selected"] == 14, counts
    assert counts["candidate"] == 27, (
        "the screened set collapsed to the applied count, so the draft reports "
        f"no reduction: {counts}")
    assert counts["original"] == 27, (
        "`original` came from a `feature_cols` the apply overwrote: %s" % counts)
    assert counts["candidate"] != counts["selected"], counts


def test_d9_01_the_screening_step_keeps_its_own_two_numbers():
    """19 ranked → 6 kept and 8 carried through are facts about the SELECTION."""
    from ml.publication import _resolve_workflow_feature_counts

    st.session_state["workflow_provenance"] = _applied_provenance()
    counts = _resolve_workflow_feature_counts(
        list(_APPLIED),
        logged_steps={"Feature Selection": [_SCREENING_ENTRY, _APPLIED_ENTRY]},
        data_config={"feature_cols": list(_APPLIED)},
    )
    assert counts["ranked"] == 19, counts
    assert counts["ranked_kept"] == 6, counts
    assert counts["carried_unranked"] == 8, counts
    assert counts["ranked_kept"] + counts["carried_unranked"] == counts["selected"]


def test_d9_01_the_ledger_fallback_identifies_entries_by_payload_not_position():
    """With no provenance the two records are still told apart."""
    from ml.publication import _resolve_workflow_feature_counts

    counts = _resolve_workflow_feature_counts(
        None,
        logged_steps={"Feature Selection": [_SCREENING_ENTRY, _APPLIED_ENTRY]},
        data_config={},
    )
    assert counts["selected"] == 14, counts
    assert counts["candidate"] == 19, (
        "the screening entry's `n_features_before` was lost to the Apply entry "
        f"sitting last under the same step name: {counts}")


def test_d9_01_the_draft_reports_the_reduction_that_happened():
    """The Methods paragraph and the Evidence Map must state the same funnel."""
    from ml.narrative_engine import NarrativeEngine

    engine = NarrativeEngine(
        _applied_provenance(), None,
        manuscript_context={
            "feature_counts": {"original": 27, "candidate": 27, "selected": 14,
                               "engineered": 0, "ranked": 19, "ranked_kept": 6,
                               "carried_unranked": 8},
            "feature_names_for_manuscript": list(_APPLIED),
        },
    )
    engine._apply_manuscript_context()
    paragraph = engine._gen_predictor_variables()

    assert "retained all 14 candidate predictors" not in paragraph, paragraph
    assert "All 14 candidate predictors were retained" not in paragraph, paragraph
    assert "ranked 19 candidate predictors and retained 6" in paragraph, paragraph
    assert "8 non-ranked predictors carried through" in paragraph, paragraph
    assert "from 27 to 14" in paragraph, paragraph

    evidence_map = engine.generate_evidence_map()
    assert "27 → 14 predictors" in evidence_map, evidence_map


# ── D9-03 · the profile is recomputed where it is read ───────────────────────

def _study(n=400):
    import numpy as np

    rng = np.random.default_rng(9)
    return pd.DataFrame({
        "age": rng.integers(20, 80, n),
        "bmi": rng.normal(27, 5, n),
        "bp_sys": rng.normal(130, 15, n),
        "kcal": rng.normal(2000, 400, n),
        "gender": rng.choice(["male", "female"], n),
        # ~12% positives: imbalanced enough for the page-06 card to exist
        "y": (rng.random(n) < 0.12).astype(int),
    })


def _configured_session(features):
    from utils.session_state import DataConfig

    df = _study()
    st.session_state["raw_data"] = df
    st.session_state["data_config"] = DataConfig(
        target_col="y", feature_cols=list(features), task_type="classification")
    st.session_state["selected_features"] = list(features)
    return df


def test_d9_03_applying_a_selection_leaves_a_profile_that_is_recomputed():
    """`reset_downstream_results` clears it; the consumer must not go silent."""
    from utils.session_state import ensure_dataset_profile

    _configured_session(["age", "bmi", "bp_sys", "kcal", "gender"])
    assert st.session_state.get("dataset_profile") is None

    profile = ensure_dataset_profile()
    assert profile is not None, (
        "the profile was never recomputed, so page 06 loses the imbalance card, "
        "the rebalancing control and the viability badges with nothing on screen")
    assert set(profile.feature_profiles) == {"age", "bmi", "bp_sys", "kcal", "gender"}
    assert profile.target_profile is not None
    assert st.session_state["dataset_profile_scope"]["n_rows"] == profile.n_rows


def test_d9_03_a_profile_describing_the_old_feature_set_is_replaced():
    """A profile that survived a selection would describe columns that are gone."""
    from utils.session_state import ensure_dataset_profile

    _configured_session(["age", "bmi", "bp_sys", "kcal", "gender"])
    stale = ensure_dataset_profile()
    assert stale is not None

    # page 04's apply: 5 predictors → 2
    st.session_state["selected_features"] = ["age", "bmi"]
    st.session_state["data_config"].feature_cols = ["age", "bmi"]
    fresh = ensure_dataset_profile()

    assert set(fresh.feature_profiles) == {"age", "bmi"}, (
        "the profile still describes the pre-selection feature set, so every "
        "p/n ratio and suitability badge computed from it is about a model "
        "nobody is fitting")


def test_d9_03_the_pages_that_read_the_profile_ask_for_one():
    """Every consumer on the post-selection path recomputes before it reads."""
    for page in ("pages/05_Preprocess.py", "pages/06_Train_and_Compare.py",
                 "pages/10_Report_Export.py"):
        text = _source(page)
        assert "ensure_dataset_profile" in text, (
            f"{page} still reads `dataset_profile` without recomputing it; "
            "applying a feature selection makes its panels vanish silently")


# ── D9-04 · a TRIPOD tick names the evidence the item asks for ───────────────

def _entry(**kw):
    from utils.insight_ledger import Insight

    base = dict(id="x", source_page="01_Upload_and_Audit", category="data_quality",
                severity="info", finding="", implication="Logged methodology decision",
                resolved=True, resolved_by="", resolved_on_page="01_Upload_and_Audit")
    base.update(kw)
    return Insight(**base)


def test_d9_04_a_dtype_recode_does_not_certify_the_missing_data_item():
    """TRIPOD 9 is how missing values were HANDLED; this left them all blank."""
    from utils.insight_ledger import tripod_keys_certified

    recode = _entry(
        id="method_data_cleaning_recode",
        finding="Recoded outcome 'meds_hbp' from True/False to 1/0 "
                "(True → 1, False → 0); blank values left blank",
        resolved_by="Recoded outcome 'meds_hbp' from True/False to 1/0 "
                    "(True → 1, False → 0); blank values left blank",
        resolution_details={"source": "target_dtype_repair", "column": "meds_hbp",
                            "n_true": 5527, "n_false": 770, "n_missing": 15552,
                            "action_type": "data_cleaning"},
    )
    # The category still hands it the key — the gate is what refuses it.
    assert recode.tripod_keys == ["missing_data"], recode.tripod_keys
    assert "missing_data" not in tripod_keys_certified(recode), (
        "a dtype recode ticked 'Describe how missing data were handled'")

    config = _entry(
        id="method_upload_and_audit",
        finding="Configured classification task with 27 features, target: meds_hbp",
        resolved_by="Configured classification task with 27 features, target: meds_hbp",
        resolution_details={"target": "meds_hbp", "n_features": 27},
    )
    assert "missing_data" not in tripod_keys_certified(config)


def test_d9_04_the_imputation_record_does_certify_it():
    """The tick survives — it moves to the record that actually describes it."""
    from utils.insight_ledger import tripod_keys_certified

    preprocessing = _entry(
        id="method_preprocessing", category="methodology",
        source_page="05_Preprocess", resolved_on_page="05_Preprocess",
        finding="Configured preprocessing pipeline",
        resolved_by="Configured preprocessing pipeline",
        resolution_details={"imputation": "median", "scaling": "standard",
                            "action_type": "preprocessing"},
    )
    assert "missing_data" in tripod_keys_certified(preprocessing)


def test_d9_04_one_explainability_line_certifies_neither_15a_nor_19a():
    """An importance ranking is not the model, and it is not the Discussion."""
    from utils.insight_ledger import tripod_keys_certified

    explain = _entry(
        id="method_explainability", category="explainability",
        source_page="07_Explainability", resolved_on_page="07_Explainability",
        finding="Ran permutation_importance, shap on 3 models",
        resolved_by="Ran permutation_importance, shap on 3 models",
        resolution_details={"analyses": ["permutation_importance", "shap"],
                            "models": ["logreg", "rf", "histgb_clf"]},
    )
    assert explain.tripod_keys == ["full_model", "interpretation"], explain.tripod_keys
    certified = tripod_keys_certified(explain)
    assert "full_model" not in certified, (
        "a permutation-importance run ticked 'Present the full prediction model "
        "to allow predictions for individuals'")
    assert "interpretation" not in certified, (
        "the same line ticked the Discussion's overall interpretation")


def test_d9_04_each_tick_can_name_the_entry_that_certified_it():
    from utils.insight_ledger import InsightLedger

    ledger = InsightLedger()
    ledger.upsert(_entry(
        id="method_data_cleaning_recode",
        finding="Recoded outcome 'meds_hbp' to 1/0; blank values left blank",
        resolved_by="Recoded outcome 'meds_hbp' to 1/0; blank values left blank",
        resolution_details={"action_type": "data_cleaning"}))
    ledger.upsert(_entry(
        id="method_preprocessing", category="methodology",
        source_page="05_Preprocess", resolved_on_page="05_Preprocess",
        finding="Configured preprocessing pipeline",
        resolved_by="Configured preprocessing pipeline",
        resolution_details={"imputation": "median"}))

    evidence = ledger.get_tripod_evidence()
    assert evidence["missing_data"].id == "method_preprocessing", (
        "the note beside the tick came from the first entry holding the key "
        "rather than from the one that certified it")
    assert ledger.get_tripod_status()["missing_data"] is True


def test_d9_04_the_missing_data_tick_reads_where_the_strategy_lives():
    """`preprocessing_config` has held no `numeric_imputation` since per-model."""
    page = _source("pages/10_Report_Export.py")
    block = page[page.index("prep_config = st.session_state.get('preprocessing_config'"):]
    block = block[:block.index("done, total = tracker.get_progress()")]
    assert "preprocessing_config_by_model" in block, (
        "the missing-data tick still reads a key the per-model preprocessing "
        "config does not carry, so it never fires and the item is certified by "
        "whatever else happens to hold the key")
    assert "get_tripod_evidence" in page, page[:0]


# ── D9-05 · both counts name their universe ──────────────────────────────────

def _override_pair():
    from utils.workflow_provenance import WorkflowProvenance

    prov = WorkflowProvenance()
    prov.record_upload("y", "classification", ["glucose", "gender"], 6297)
    for parametric, overridden, p in ((False, False, 4e-154), (True, True, 3e-36)):
        prov.record_statistical_test(
            test_name="t-test (ind.)" if parametric else "Mann-Whitney U",
            variable="glucose", statistic=1.0, p_value=p,
            details={"parametric": parametric,
                     "assumption_basis": "Shapiro-Wilk p<0.001",
                     "assumption_overridden": overridden},
        )
    return prov


def test_d9_05_the_draft_and_its_evidence_map_do_not_contradict_each_other():
    """Two records of one comparison: 1 and 2 are both right, unlabeled they clash."""
    from ml.narrative_engine import NarrativeEngine

    engine = NarrativeEngine(_override_pair(), None)
    paragraph = engine._gen_statistical_validation()
    evidence_map = engine.generate_evidence_map()

    assert "across the 1 test reported here" in paragraph, paragraph
    assert "2 test runs recorded" in paragraph, (
        "the sentence counts comparisons and never says so, so the Evidence "
        f"Map's record count reads as a contradiction: {paragraph!r}")

    row = [line for line in evidence_map.splitlines()
           if "statistical-test record" in line]
    assert row, evidence_map
    assert "2 recorded test runs" in row[0] and "1 distinct comparison" in row[0], (
        f"the evidence row states a bare count of records: {row[0]!r}")


# ── D9-06 · coach register stays out of the manuscript ───────────────────────

def _coach_card():
    from utils.insight_ledger import Insight

    return Insight(
        id="train_prefer_simpler", source_page="06_Train_and_Compare",
        category="model_selection", severity="warning",
        finding=("Logistic Regression performed within 0.7% of Histogram Gradient "
                 "Boosting (F1 0.8552 vs 0.8495). A reviewer would question why "
                 "the more complex model was selected."),
        implication=("When models perform comparably, parsimony favors the simpler, "
                     "more interpretable model."),
        manuscript_text=("the simpler Logistic Regression performed within 0.7% of "
                         "the more complex Histogram Gradient Boosting "
                         "(F1 0.8552 vs 0.8495), so parsimony considerations favor "
                         "the simpler specification"),
    )


def test_d9_06_the_limitations_list_is_not_spliced_from_finding_and_rationale():
    """`f"{finding}: {implication}"` produced a full stop followed by a colon."""
    from utils.insight_ledger import InsightLedger

    ledger = InsightLedger()
    ledger.add(_coach_card())
    limitations = ledger.discussion_points_for_manuscript()["limitations"]

    assert limitations, limitations
    assert any("parsimony considerations favor" in text for text in limitations)
    assert not any("A reviewer would" in text for text in limitations), limitations
    assert not any(".: " in text for text in limitations), limitations

    page = _source("pages/10_Report_Export.py")
    assert 'f"{_ui.finding}: {_ui.implication}"' not in page, (
        "page 10 still splices coach cards into the Limitations list")
    assert "discussion_points_for_manuscript" in page


def test_d9_06_the_discussion_prose_uses_the_manuscript_register_sentence():
    from ml.narrative_engine import NarrativeEngine
    from utils.insight_ledger import InsightLedger
    from utils.workflow_provenance import WorkflowProvenance

    ledger = InsightLedger()
    ledger.add(_coach_card())
    engine = NarrativeEngine(WorkflowProvenance(), ledger)
    text = engine._discussion_model_pattern(
        "classification",
        {"logreg": {"F1": 0.8552}, "histgb_clf": {"F1": 0.8495}},
    )
    assert "A reviewer would question" not in text, text
    assert "parsimony considerations favor the simpler specification" in text, text


def test_d9_06_the_validator_sees_the_coaching_registers_it_missed():
    """The check reported PASS over the sentence sitting in the Discussion."""
    from ml.manuscript_validator import validate_manuscript_bundle

    report_text = (
        "## Discussion (Draft)\n\n### Principal Findings\n\n"
        "Logistic Regression achieved F1 0.8552. A reviewer would question why "
        "the more complex model was selected.\n"
    )
    result = validate_manuscript_bundle(
        manuscript_context={}, methods_text="", report_text=report_text,
        latex_text="", task_type="classification",
    )
    failed = {check.name for check in result.failed_checks}
    assert "No coaching language patterns remain in export text" in failed, (
        "reviewer-anticipation coaching still passes the check that claims to "
        "look for it")


def test_d9_06_the_apps_own_advice_log_is_not_read_as_manuscript_prose():
    """The decision appendix records what the app advised; that is not a defect."""
    from ml.manuscript_validator import validate_manuscript_bundle

    report_text = (
        "## Key Observations and Resolutions\n\n"
        "- [WARNING] A reviewer would question why the more complex model was "
        "selected.\n\n"
        "## Discussion (Draft)\n\nThe model achieved F1 0.8552.\n"
    )
    result = validate_manuscript_bundle(
        manuscript_context={}, methods_text="", report_text=report_text,
        latex_text="", task_type="classification",
    )
    failed = {check.name for check in result.failed_checks}
    assert "No coaching language patterns remain in export text" not in failed


# ── D9-09 · one counting vocabulary in the structural review ─────────────────

def test_d9_09_the_expander_label_speaks_the_headline_vocabulary():
    """"Found 1 worth checking, 1 note." above "Also worth a look (2)"."""
    text = _source("utils/import_ui.py")
    assert 'f"Also worth a look ({len(rest)})"' not in text, (
        "the expander counts non-critical findings as one total while the "
        "headline breaks them out by severity")
    assert "summarize(rest)" in text, (
        "the two lines must be composed by the same summarizer")


def test_d9_09_the_two_lines_agree_on_the_same_findings():
    from ml.import_doctor import summarize

    class _F:
        def __init__(self, severity):
            self.severity = severity

    rest = [_F("warning"), _F("info")]
    phrase = summarize(rest).removeprefix("Found ").rstrip(".")
    assert phrase == "1 worth checking, 1 note", phrase
    assert "2" not in phrase


# ── D9-10 · one naming rule for a method list ───────────────────────────────

def test_d9_10_a_selection_method_list_is_named_by_one_rule():
    """`lasso` is a model key too, and the model renamer took only that one."""
    from utils.insight_ledger import _clean_for_manuscript, feature_selection_method_label

    rendered = "Selected 6 features using " + ", ".join(
        feature_selection_method_label(m) for m in ("lasso", "rfe"))
    assert rendered == "Selected 6 features using LASSO, RFE-CV", rendered

    cleaned = _clean_for_manuscript(rendered)
    assert cleaned == rendered, (
        f"the manuscript cleaner rewrote a selection method as a model: {cleaned!r}")
    assert "Lasso Regression" not in cleaned


def test_d9_10_the_model_key_rule_still_applies_to_model_lists():
    """The protection must not switch off the renaming it sits beside."""
    from utils.insight_ledger import _clean_for_manuscript

    cleaned = _clean_for_manuscript(
        "Preprocessing was tuned for 3 models: HISTGB_CLF, LOGREG, RF.")
    assert "Logistic Regression" in cleaned and "Random Forest" in cleaned, cleaned
    assert "LOGREG" not in cleaned, cleaned


def test_d9_10_page_04_writes_the_labels_rather_than_the_raw_keys():
    page = _source("pages/04_Feature_Selection.py")
    block = page[page.index("methods_used = "):page.index("consensus_threshold': consensus_threshold")]
    assert "feature_selection_method_label" in block, (
        "the action sentence is still composed from raw method keys, so the "
        "audit trail prints one display label beside one internal key")
    assert not re.search(r'", "\.join\(methods_completed\)', block), block

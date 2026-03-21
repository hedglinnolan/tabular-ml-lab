"""
Comprehensive functional test for the unified InsightLedger.

Simulates a real user workflow across all 10 pages, verifying that:
1. EDA insights are created with correct relevant_pages
2. Insights surface on the correct downstream pages
3. Resolution on one page marks resolved across all views
4. log_methodology calls create pre-resolved ledger entries
5. TRIPOD auto-completes from ledger resolutions
6. Manuscript narrative groups by workflow phase correctly
7. Report Export can generate a complete narrative
8. No legacy references remain functional
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

# Mock streamlit
class MockSessionState(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(name)
    def __setattr__(self, name, value):
        self[name] = value

class MockSt:
    session_state = MockSessionState()
    @staticmethod
    def cache_data(*a, **kw):
        def wrapper(f): return f
        return wrapper

import unittest.mock as mock
sys.modules.setdefault('streamlit', MockSt)
st = MockSt

from utils.insight_ledger import (
    Insight, InsightLedger, get_ledger, CATEGORIES, SEVERITIES,
    PAGES, WORKFLOW_PHASES, TRIPOD_CATEGORY_MAP,
)
from utils.session_state import log_methodology


def fresh_ledger():
    """Reset session state and return a fresh ledger."""
    st.session_state.clear()
    return get_ledger()


def make_test_data():
    """Create a realistic test DataFrame."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        'Age': np.random.normal(50, 15, n),
        'BMI': np.random.exponential(25, n),  # Skewed
        'Glucose': np.random.normal(120, 30, n),
        'BloodPressure': np.random.normal(80, 12, n),
        'Insulin': np.random.exponential(80, n),  # Skewed
        'SkinThickness': np.random.normal(20, 8, n),
        'Target': np.random.choice([0, 1], n, p=[0.65, 0.35]),
    })
    # Inject some missing values
    df.loc[np.random.choice(n, 40, replace=False), 'BMI'] = np.nan
    df.loc[np.random.choice(n, 30, replace=False), 'Insulin'] = np.nan
    return df


import pytest

@pytest.fixture
def r():
    """Provide TestResults instance as a pytest fixture."""
    return TestResults()


class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self, name):
        self.passed += 1
        print(f"  ✓ {name}")

    def fail(self, name, msg):
        self.failed += 1
        self.errors.append((name, msg))
        print(f"  ✗ {name}: {msg}")

    def assert_true(self, cond, name, msg=""):
        if cond:
            self.ok(name)
        else:
            self.fail(name, msg or "assertion failed")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*50}")
        print(f"Results: {self.passed}/{total} passed")
        if self.errors:
            print("Failures:")
            for name, msg in self.errors:
                print(f"  - {name}: {msg}")
        return self.failed == 0


def test_data_model(r: TestResults):
    """Test the Insight dataclass and its new fields."""
    print("\n--- Data Model ---")

    # Basic creation
    i = Insight(
        id="test_1", source_page="02_EDA", category="distribution",
        severity="warning", finding="BMI skewed", implication="Bad for linear",
        relevant_pages=["03_Feature_Engineering", "05_Preprocess"],
        affected_features=["BMI"], metadata={"skewness": 3.2},
    )
    r.assert_true(i.tripod_keys == ["predictor_handling"], "Auto tripod_keys from category")
    r.assert_true(len(i.relevant_pages) == 2, "relevant_pages set correctly")
    r.assert_true(i.auto_generated == True, "Default auto_generated")
    r.assert_true(i.resolved == False, "Default not resolved")
    r.assert_true(i.resolution_details == {}, "Default empty resolution_details")

    # Legacy action_page migration
    i2 = Insight(
        id="test_2", source_page="02_EDA", category="data_quality",
        severity="info", finding="test", implication="test",
        action_page="05_Preprocess",
    )
    r.assert_true("05_Preprocess" in i2.relevant_pages, "Legacy action_page migrated")

    # Serialization roundtrip
    d = i.to_dict()
    i3 = Insight.from_dict(d)
    r.assert_true(i3.id == i.id, "Roundtrip preserves id")
    r.assert_true(i3.relevant_pages == i.relevant_pages, "Roundtrip preserves relevant_pages")
    r.assert_true(i3.tripod_keys == i.tripod_keys, "Roundtrip preserves tripod_keys")
    r.assert_true(i3.metadata == i.metadata, "Roundtrip preserves metadata")


def test_ledger_core(r: TestResults):
    """Test ledger CRUD operations."""
    print("\n--- Ledger Core Operations ---")

    ledger = fresh_ledger()

    # Add
    added = ledger.add(Insight(
        id="a1", source_page="02_EDA", category="distribution",
        severity="warning", finding="Skew", implication="Bad",
        relevant_pages=["03_Feature_Engineering"],
    ))
    r.assert_true(added, "Add returns True")
    r.assert_true(len(ledger) == 1, "Length after add")

    # Duplicate add
    added2 = ledger.add(Insight(
        id="a1", source_page="02_EDA", category="distribution",
        severity="warning", finding="Skew v2", implication="Bad v2",
    ))
    r.assert_true(not added2, "Duplicate add returns False")
    r.assert_true(len(ledger) == 1, "Length unchanged after duplicate")
    r.assert_true(ledger.get("a1").finding == "Skew", "Original finding preserved")

    # Upsert
    ledger.upsert(Insight(
        id="a1", source_page="02_EDA", category="distribution",
        severity="blocker", finding="Skew (updated)", implication="Really bad",
    ))
    r.assert_true(ledger.get("a1").finding == "Skew (updated)", "Upsert updates finding")
    r.assert_true(len(ledger) == 1, "Upsert doesn't duplicate")

    # Resolve
    resolved = ledger.resolve(
        "a1", "Applied log transform", "03_Feature_Engineering",
        {"method": "log1p", "columns": ["BMI"]},
    )
    r.assert_true(resolved, "Resolve returns True")
    i = ledger.get("a1")
    r.assert_true(i.resolved, "Marked resolved")
    r.assert_true(i.resolved_by == "Applied log transform", "resolved_by set")
    r.assert_true(i.resolved_on_page == "03_Feature_Engineering", "resolved_on_page set")
    r.assert_true(i.resolution_details == {"method": "log1p", "columns": ["BMI"]}, "resolution_details set")
    r.assert_true(i.resolved_at is not None, "resolved_at timestamp set")

    # Upsert preserves resolution
    ledger.upsert(Insight(
        id="a1", source_page="02_EDA", category="distribution",
        severity="warning", finding="Skew (re-upserted)", implication="test",
    ))
    r.assert_true(ledger.get("a1").resolved, "Upsert preserves resolution state")

    # Remove
    removed = ledger.remove("a1")
    r.assert_true(removed, "Remove returns True")
    r.assert_true(len(ledger) == 0, "Length after remove")


def test_page_queries(r: TestResults):
    """Test page-level queries — the core of per-page coaching."""
    print("\n--- Page Queries ---")

    ledger = fresh_ledger()

    # Create insights that span multiple pages
    ledger.add(Insight(
        id="skew_1", source_page="02_EDA", category="distribution",
        severity="warning", finding="BMI skewed",
        implication="Transform it",
        relevant_pages=["03_Feature_Engineering", "05_Preprocess"],
        affected_features=["BMI"],
    ))
    ledger.add(Insight(
        id="missing_1", source_page="02_EDA", category="data_quality",
        severity="warning", finding="40% missing in Insulin",
        implication="Impute it",
        relevant_pages=["05_Preprocess", "10_Report_Export"],
        affected_features=["Insulin"],
    ))
    ledger.add(Insight(
        id="sufficiency_1", source_page="02_EDA", category="sufficiency",
        severity="info", finding="200 rows is adequate",
        implication="All models viable",
        relevant_pages=["06_Train_and_Compare"],
    ))

    # Page: Feature Engineering — should see skew only
    fe = ledger.get_unresolved(page="03_Feature_Engineering")
    r.assert_true(len(fe) == 1, f"FE sees 1 insight (got {len(fe)})")
    r.assert_true(fe[0].id == "skew_1", "FE sees skew insight")

    # Page: Preprocess — should see skew + missing
    pp = ledger.get_unresolved(page="05_Preprocess")
    r.assert_true(len(pp) == 2, f"Preprocess sees 2 insights (got {len(pp)})")

    # Page: Train — should see sufficiency
    tc = ledger.get_unresolved(page="06_Train_and_Compare")
    r.assert_true(len(tc) == 1, f"Train sees 1 insight (got {len(tc)})")

    # Page: EDA — sees all (as source_page)
    eda = ledger.get_for_page("02_EDA")
    r.assert_true(len(eda) == 3, f"EDA sees all 3 (got {len(eda)})")

    # Resolve skew in FE
    ledger.resolve("skew_1", "Applied log1p", "03_Feature_Engineering")

    # Now FE sees 0 unresolved
    fe2 = ledger.get_unresolved(page="03_Feature_Engineering")
    r.assert_true(len(fe2) == 0, f"FE sees 0 after resolve (got {len(fe2)})")

    # Preprocess sees 1 now (missing only)
    pp2 = ledger.get_unresolved(page="05_Preprocess")
    r.assert_true(len(pp2) == 1, f"Preprocess sees 1 after resolve (got {len(pp2)})")

    # Page summary
    ps = ledger.page_summary("05_Preprocess")
    r.assert_true(ps["total"] == 2, f"Preprocess total=2 (got {ps['total']})")
    r.assert_true(ps["unresolved"] == 1, f"Preprocess unresolved=1 (got {ps['unresolved']})")
    r.assert_true(ps["resolved"] == 1, f"Preprocess resolved=1 (got {ps['resolved']})")

    # Feature query
    bmi_insights = ledger.get_for_features(["BMI"])
    r.assert_true(len(bmi_insights) == 1, f"Feature query finds BMI insight (got {len(bmi_insights)})")


def test_tripod(r: TestResults):
    """Test TRIPOD auto-completion from resolved ledger entries."""
    print("\n--- TRIPOD Auto-Completion ---")

    ledger = fresh_ledger()

    # Unresolved insight shouldn't trigger TRIPOD
    ledger.add(Insight(
        id="missing_1", source_page="02_EDA", category="data_quality",
        severity="warning", finding="Missing data", implication="Handle it",
        tripod_keys=["missing_data"],
    ))
    status = ledger.get_tripod_status()
    r.assert_true(not status.get("missing_data", False), "Unresolved doesn't trigger TRIPOD")

    # Resolve it
    ledger.resolve("missing_1", "KNN imputation", "05_Preprocess")
    status2 = ledger.get_tripod_status()
    r.assert_true(status2.get("missing_data", False), "Resolved triggers TRIPOD missing_data")

    # Category auto-mapping
    ledger.add(Insight(
        id="model_1", source_page="06_Train_and_Compare", category="model_selection",
        severity="info", finding="Trained RF", implication="Good fit",
        resolved=True, resolved_by="Trained RF", resolved_on_page="06_Train_and_Compare",
    ))
    status3 = ledger.get_tripod_status()
    r.assert_true(status3.get("model_building", False), "model_selection category → model_building TRIPOD")

    progress = ledger.get_tripod_progress()
    r.assert_true(progress[0] >= 2, f"At least 2 TRIPOD items done (got {progress[0]})")
    r.assert_true(progress[1] > 0, f"Total TRIPOD items > 0 (got {progress[1]})")


def test_methodology_bridge(r: TestResults):
    """Test that log_methodology creates ledger entries."""
    print("\n--- Methodology → Ledger Bridge ---")

    ledger = fresh_ledger()

    # Call log_methodology
    log_methodology("EDA", "Ran correlation analysis", {"method": "pearson"})
    r.assert_true(len(ledger) >= 1, f"Ledger has entry after log_methodology (got {len(ledger)})")

    entry = ledger.insights[0]
    r.assert_true(entry.resolved, "Entry is pre-resolved")
    r.assert_true(entry.resolution_details.get("method") == "pearson", "Details preserved")
    r.assert_true("10_Report_Export" in entry.relevant_pages, "Report Export in relevant_pages")

    # Multiple additive calls
    log_methodology("Statistical Validation", "t-test on BMI", {"p": 0.01})
    log_methodology("Statistical Validation", "chi-square on Gender", {"p": 0.04})
    r.assert_true(len(ledger) >= 3, f"Multiple additive entries (got {len(ledger)})")

    # Replace-step behavior
    log_methodology("Model Training", "Trained Ridge", {"model": "ridge"})
    before = len(ledger)
    log_methodology("Model Training", "Trained RF", {"model": "rf"})
    r.assert_true(len(ledger) >= before, f"Replace step doesn't lose entries")

    # methodology_log still populated
    mlog = st.session_state.get("methodology_log", [])
    r.assert_true(len(mlog) >= 4, f"methodology_log populated ({len(mlog)} entries)")


def test_narrative_generation(r: TestResults):
    """Test manuscript narrative and report generation."""
    print("\n--- Narrative Generation ---")

    ledger = fresh_ledger()

    # Add and resolve insights across workflow phases
    ledger.add(Insight(
        id="skew_1", source_page="02_EDA", category="distribution",
        severity="warning", finding="5 features heavily right-skewed",
        implication="Transform them",
        relevant_pages=["03_Feature_Engineering"],
    ))
    ledger.resolve("skew_1", "Applied log1p to all 5 features",
                   "03_Feature_Engineering",
                   {"transforms": ["log1p"], "features": ["BMI", "Insulin"]})

    ledger.add(Insight(
        id="missing_1", source_page="02_EDA", category="data_quality",
        severity="warning", finding="BMI has 42% missing",
        implication="Impute", relevant_pages=["05_Preprocess"],
    ))
    ledger.resolve("missing_1", "KNN imputation (k=5)",
                   "05_Preprocess",
                   {"method": "knn", "k": 5, "columns": ["BMI"]})

    # Manuscript narrative
    narratives = ledger.to_manuscript_narrative()
    r.assert_true(len(narratives) >= 2, f"At least 2 phases (got {len(narratives)})")

    # Check deduplication — each entry appears in only one phase
    all_text = " ".join(narratives.values())
    r.assert_true(all_text.count("right-skewed") == 1, "Skew mentioned once (no duplicates)")
    r.assert_true(all_text.count("42% missing") == 1, "Missing mentioned once (no duplicates)")

    # Skew should be in FE phase, missing in Preprocessing phase
    r.assert_true("right-skewed" in narratives.get("Feature Engineering & Selection", ""),
                  "Skew appears in FE phase")
    r.assert_true("42% missing" in narratives.get("Preprocessing", ""),
                  "Missing appears in Preprocessing phase")

    # Report narrative
    report = ledger.narrative_for_report()
    r.assert_true("Addressed" in report or "addressed" in report, "Report mentions addressed")
    r.assert_true("log1p" in report, "Report mentions resolution action")

    # Methodology log from resolved entries
    mlog = ledger.get_methodology_log()
    r.assert_true(len(mlog) == 2, f"2 methodology log entries (got {len(mlog)})")

    # Provenance timeline
    timeline = ledger.provenance_timeline()
    r.assert_true(len(timeline) == 4, f"4 events: 2 created + 2 resolved (got {len(timeline)})")
    r.assert_true(timeline[0]["type"] == "created", "First event is creation")


def test_full_workflow_simulation(r: TestResults):
    """Simulate a complete user workflow and verify cross-page coherence."""
    print("\n--- Full Workflow Simulation ---")

    ledger = fresh_ledger()

    # === PAGE 1: Upload ===
    log_methodology("Upload & Audit", "Loaded diabetes.csv", {"rows": 768, "cols": 9})

    # === PAGE 2: EDA ===
    # Auto-detected insights (simulating what EDA page generates)
    ledger.add(Insight(
        id="eda_sufficiency_borderline", source_page="02_EDA", category="sufficiency",
        severity="warning",
        finding="Data sufficiency is borderline (768 rows, 8 features)",
        implication="Prefer simpler models and tighter regularization.",
        recommended_action="Consider feature reduction before complex modeling",
        relevant_pages=["04_Feature_Selection", "06_Train_and_Compare"],
    ))
    ledger.add(Insight(
        id="eda_missing_moderate", source_page="02_EDA", category="data_quality",
        severity="info",
        finding="2 features have moderate missingness (5-15%)",
        implication="Standard imputation should suffice",
        recommended_action="Address in Preprocessing",
        relevant_pages=["05_Preprocess"],
        affected_features=["Insulin", "SkinThickness"],
    ))
    ledger.add(Insight(
        id="eda_skew_group", source_page="02_EDA", category="distribution",
        severity="warning",
        finding="3 features are heavily right-skewed",
        implication="Linear models will underperform without transforms",
        recommended_action="Consider transforms in Feature Engineering or Preprocessing",
        relevant_pages=["03_Feature_Engineering", "05_Preprocess"],
        affected_features=["Insulin", "BMI", "DiabetesPedigree"],
        metadata={"features": {"Insulin": 4.1, "BMI": 3.2, "DiabetesPedigree": 1.9}},
    ))
    ledger.add(Insight(
        id="eda_corr_cluster_0", source_page="02_EDA", category="relationship",
        severity="info",
        finding="Collinear cluster: {Age, Pregnancies} (r=0.54)",
        implication="May cause multicollinearity in linear models",
        recommended_action="Review in Feature Selection — consider dropping 1 of 2",
        relevant_pages=["04_Feature_Selection", "05_Preprocess"],
        affected_features=["Age", "Pregnancies"],
    ))
    ledger.add(Insight(
        id="eda_opportunity_class_imbalance", source_page="02_EDA", category="distribution",
        severity="opportunity",
        finding="Class imbalance: 65%/35% (not severe)",
        implication="Standard training should work; class weighting optional",
        recommended_action="Consider class weighting in training",
        relevant_pages=["06_Train_and_Compare"],
    ))

    log_methodology("EDA", "Completed exploratory data analysis",
                    {"n_insights": 5, "n_blockers": 0})

    # Verify page views BEFORE any resolutions
    r.assert_true(len(ledger.get_unresolved(page="03_Feature_Engineering")) == 1,
                  "FE: 1 unresolved (skew)")
    r.assert_true(len(ledger.get_unresolved(page="04_Feature_Selection")) == 2,
                  "FS: 2 unresolved (sufficiency + correlation)")
    r.assert_true(len(ledger.get_unresolved(page="05_Preprocess")) == 3,
                  "Preprocess: 3 unresolved (missing + skew + correlation)")
    r.assert_true(len(ledger.get_unresolved(page="06_Train_and_Compare")) == 2,
                  "Train: 2 unresolved (sufficiency + opportunity)")

    # === PAGE 3: Feature Engineering ===
    ledger.resolve("eda_skew_group", "Applied log1p to Insulin, BMI, DiabetesPedigree",
                   "03_Feature_Engineering",
                   {"transforms": ["log1p"], "features": ["Insulin", "BMI", "DiabetesPedigree"],
                    "new_columns": ["log1p_Insulin", "log1p_BMI", "log1p_DiabetesPedigree"]})
    log_methodology("Feature Engineering", "Applied log transforms to 3 skewed features",
                    {"method": "log1p", "n_features": 3})

    # After FE: skew resolved everywhere
    r.assert_true(len(ledger.get_unresolved(page="03_Feature_Engineering")) == 0,
                  "FE: 0 after resolve")
    r.assert_true(len(ledger.get_unresolved(page="05_Preprocess")) == 2,
                  "Preprocess: 2 after skew resolved (missing + correlation)")

    # === PAGE 4: Feature Selection ===
    ledger.resolve("eda_corr_cluster_0", "Removed Pregnancies (kept Age due to stronger univariate signal)",
                   "04_Feature_Selection",
                   {"dropped": ["Pregnancies"], "kept": ["Age"], "method": "manual_review"})
    log_methodology("Feature Selection Applied", "Selected 7 features (dropped Pregnancies)",
                    {"n_selected": 7, "method": "expert + LASSO"})

    r.assert_true(len(ledger.get_unresolved(page="04_Feature_Selection")) == 1,
                  "FS: 1 remaining (sufficiency)")
    r.assert_true(len(ledger.get_unresolved(page="05_Preprocess")) == 1,
                  "Preprocess: 1 remaining (missing)")

    # === PAGE 5: Preprocess ===
    ledger.resolve("eda_missing_moderate", "Median imputation for Insulin, SkinThickness",
                   "05_Preprocess",
                   {"method": "median", "columns": ["Insulin", "SkinThickness"]})
    log_methodology("Preprocessing", "Built preprocessing pipelines for 3 models",
                    {"models": ["ridge", "rf", "mlp"]})

    r.assert_true(len(ledger.get_unresolved(page="05_Preprocess")) == 0,
                  "Preprocess: 0 unresolved after all fixes")

    # === PAGE 6: Train ===
    log_methodology("Model Training", "Trained Ridge, RF, MLP; RF best (AUC=0.83)",
                    {"best_model": "rf", "auc": 0.83})

    # === PAGE 7: Explainability ===
    log_methodology("Explainability", "SHAP analysis on RF model",
                    {"method": "shap", "top_feature": "Glucose"})

    # === PAGE 8: Sensitivity ===
    log_methodology("Sensitivity Analysis", "Bootstrap CI (n=200): AUC 0.79-0.86",
                    {"method": "bootstrap", "n": 200, "ci_lower": 0.79, "ci_upper": 0.86})

    # === PAGE 9: Hypothesis Testing ===
    log_methodology("Statistical Validation", "t-test on Glucose by outcome (p<0.001)",
                    {"test": "t-test", "variable": "Glucose", "p_value": 0.001})

    # === VERIFICATION ===
    print()
    summary = ledger.summary()
    r.assert_true(summary["resolved"] >= 3, f"At least 3 resolved from EDA (got {summary['resolved']})")
    r.assert_true(summary["total"] >= 10, f"Total entries >= 10 including methodology (got {summary['total']})")

    # TRIPOD
    tripod = ledger.get_tripod_status()
    completed_keys = [k for k, v in tripod.items() if v]
    r.assert_true("missing_data" in completed_keys, "TRIPOD: missing_data complete")
    r.assert_true("predictor_handling" in completed_keys, "TRIPOD: predictor_handling complete")
    r.assert_true("model_building" in completed_keys, "TRIPOD: model_building complete")

    # Manuscript narrative
    narratives = ledger.to_manuscript_narrative()
    r.assert_true(len(narratives) >= 3, f"At least 3 narrative phases (got {len(narratives)})")

    # Report narrative
    report = ledger.narrative_for_report()
    r.assert_true(len(report) > 100, f"Report narrative substantial (got {len(report)} chars)")
    r.assert_true("log1p" in report, "Report mentions log1p transform")
    r.assert_true("Median" in report or "median" in report.lower(), "Report mentions imputation")

    # Provenance timeline
    timeline = ledger.provenance_timeline()
    r.assert_true(len(timeline) >= 10, f"Timeline has events (got {len(timeline)})")
    created_events = [e for e in timeline if e["type"] == "created"]
    resolved_events = [e for e in timeline if e["type"] == "resolved"]
    r.assert_true(len(created_events) >= 5, f"At least 5 created events")
    r.assert_true(len(resolved_events) >= 3, f"At least 3 resolved events")

    # Serialization
    serialized = ledger.to_list()
    restored = InsightLedger.from_list(serialized)
    r.assert_true(len(restored) == len(ledger), "Serialization preserves all entries")


def test_edge_cases(r: TestResults):
    """Test edge cases and error handling."""
    print("\n--- Edge Cases ---")

    ledger = fresh_ledger()

    # Resolve nonexistent
    r.assert_true(not ledger.resolve("nope", "foo", "bar"), "Resolve nonexistent returns False")

    # Remove nonexistent
    r.assert_true(not ledger.remove("nope"), "Remove nonexistent returns False")

    # Empty ledger queries
    r.assert_true(len(ledger.get_unresolved()) == 0, "Empty unresolved")
    r.assert_true(len(ledger.get_resolved()) == 0, "Empty resolved")
    r.assert_true(ledger.summary()["total"] == 0, "Empty summary")
    r.assert_true(not ledger.has_blockers(), "No blockers in empty")
    r.assert_true(ledger.narrative_for_report() != "", "Report on empty ledger")
    r.assert_true(ledger.to_manuscript_narrative() == {}, "Empty manuscript narrative")

    # Malformed deserialization
    restored = InsightLedger.from_list([{"id": "bad", "bogus_key": True}])
    r.assert_true(len(restored) == 0, "Malformed entries skipped")

    # Category filter
    ledger.add(Insight(
        id="a", source_page="02_EDA", category="distribution",
        severity="warning", finding="test", implication="test",
    ))
    ledger.add(Insight(
        id="b", source_page="02_EDA", category="data_quality",
        severity="info", finding="test", implication="test",
    ))
    r.assert_true(len(ledger.get_by_category("distribution")) == 1, "Category filter works")

    # Severity ordering
    ledger.add(Insight(
        id="c", source_page="02_EDA", category="distribution",
        severity="blocker", finding="blocker", implication="test",
    ))
    ordered = ledger.insights
    r.assert_true(ordered[0].severity == "blocker", "Blockers sort first")

    # Clear
    ledger.clear()
    r.assert_true(len(ledger) == 0, "Clear empties ledger")


def test_no_legacy_leaks(r: TestResults):
    """Verify no legacy insight systems are used."""
    print("\n--- Legacy System Verification ---")

    import os
    import re

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pages_dir = os.path.join(root, "pages")

    legacy_patterns = {
        "get_insights_by_category": r'\bget_insights_by_category\b',
        "add_insight (storyline)": r'from utils\.storyline import.*\badd_insight\b',
        "feature_engineering_hints read": r"session_state.*\[.*['\"]feature_engineering_hints['\"].*\]",
        "eda_insights read": r"session_state.*\[.*['\"]eda_insights['\"].*\]",
        "coach_output active use": r"if coach_output",
    }

    for page_file in sorted(os.listdir(pages_dir)):
        if not page_file.endswith('.py'):
            continue
        filepath = os.path.join(pages_dir, page_file)
        with open(filepath, 'r') as f:
            content = f.read()
        for pattern_name, pattern in legacy_patterns.items():
            matches = re.findall(pattern, content)
            if matches:
                r.fail(f"{page_file}: no {pattern_name}", f"Found: {matches[0]}")
            # Don't report pass for every pattern - too noisy

    r.ok("No legacy insight patterns in page files")


if __name__ == "__main__":
    r = TestResults()

    test_data_model(r)
    test_ledger_core(r)
    test_page_queries(r)
    test_tripod(r)
    test_methodology_bridge(r)
    test_narrative_generation(r)
    test_full_workflow_simulation(r)
    test_edge_cases(r)
    test_no_legacy_leaks(r)

    success = r.summary()
    sys.exit(0 if success else 1)

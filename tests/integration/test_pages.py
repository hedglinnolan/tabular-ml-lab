"""
Tier 2: Streamlit AppTest page rendering tests.

Each test loads a page via Streamlit's AppTest framework with injected
session state and verifies:
  1. The page renders without exceptions
  2. Key UI elements are present (widgets, disclosures, banners)
  3. Classification and regression paths both work

These catch widget bugs, session_state collisions, and missing imports
that unit tests can't reach.
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from streamlit.testing.v1 import AppTest
from tests.integration.conftest import (
    build_test_dataframe, build_classification_dataframe,
    inject_data_state, inject_trained_state,
)


# ── Helpers ──────────────────────────────────────────────────────────

def assert_no_exception(at, page_name):
    """Assert AppTest didn't raise during render."""
    if at.exception:
        msgs = [str(e.value)[:300] for e in at.exception]
        pytest.fail(f"{page_name} raised: {'; '.join(msgs)}")


def all_text(at):
    """Concatenate all visible text from markdown, info, warning, caption."""
    parts = []
    for attr in ('markdown', 'info', 'warning', 'caption', 'error'):
        for el in getattr(at, attr, []):
            parts.append(str(getattr(el, 'value', '')))
    return ' '.join(parts).lower()


# ── Page Rendering (Regression) ─────────────────────────────────────

class TestPageRendering:
    """Every page renders without crashing when data is loaded (regression)."""

    @pytest.fixture(scope="class")
    def df(self):
        return build_test_dataframe()

    def test_upload_page(self, df):
        at = AppTest.from_file("pages/01_Upload_and_Audit.py", default_timeout=30)
        inject_data_state(at, df)
        at.run()
        assert_no_exception(at, "Upload & Audit")

    def test_eda_page(self, df):
        at = AppTest.from_file("pages/02_EDA.py", default_timeout=30)
        inject_data_state(at, df)
        at.run()
        assert_no_exception(at, "EDA")

    def test_feature_engineering_page(self, df):
        at = AppTest.from_file("pages/03_Feature_Engineering.py", default_timeout=30)
        inject_data_state(at, df)
        at.run()
        assert_no_exception(at, "Feature Engineering")

    def test_feature_selection_page(self, df):
        at = AppTest.from_file("pages/04_Feature_Selection.py", default_timeout=30)
        inject_data_state(at, df)
        at.run()
        assert_no_exception(at, "Feature Selection")

    def test_preprocess_page(self, df):
        at = AppTest.from_file("pages/05_Preprocess.py", default_timeout=30)
        inject_data_state(at, df)
        at.run()
        assert_no_exception(at, "Preprocess")

    def test_train_page(self, df):
        at = AppTest.from_file("pages/06_Train_and_Compare.py", default_timeout=30)
        inject_data_state(at, df)
        at.run()
        assert_no_exception(at, "Train & Compare")

    def test_explainability_page(self, df):
        at = AppTest.from_file("pages/07_Explainability.py", default_timeout=30)
        inject_data_state(at, df)
        inject_trained_state(at, df)
        at.run()
        assert_no_exception(at, "Explainability")

    def test_sensitivity_page(self, df):
        at = AppTest.from_file("pages/08_Sensitivity_Analysis.py", default_timeout=30)
        inject_data_state(at, df)
        inject_trained_state(at, df)
        at.run()
        assert_no_exception(at, "Sensitivity Analysis")

    def test_hypothesis_page(self, df):
        at = AppTest.from_file("pages/09_Hypothesis_Testing.py", default_timeout=30)
        inject_data_state(at, df)
        at.run()
        assert_no_exception(at, "Hypothesis Testing")

    def test_report_export_page(self, df):
        at = AppTest.from_file("pages/10_Report_Export.py", default_timeout=30)
        inject_data_state(at, df)
        inject_trained_state(at, df)
        at.run()
        assert_no_exception(at, "Report Export")

    def test_theory_page(self):
        at = AppTest.from_file("pages/11_Theory_Reference.py", default_timeout=30)
        at.run()
        assert_no_exception(at, "Theory Reference")


# ── Page Rendering (Classification) ─────────────────────────────────

class TestClassificationPageRendering:
    """Pages that behave differently for classification render without crashing."""

    @pytest.fixture(scope="class")
    def clf_df(self):
        return build_classification_dataframe()

    def test_eda_classification(self, clf_df):
        at = AppTest.from_file("pages/02_EDA.py", default_timeout=30)
        inject_data_state(at, clf_df, target_col='condition', task_type='classification')
        at.run()
        assert_no_exception(at, "EDA (classification)")

    def test_train_classification(self, clf_df):
        at = AppTest.from_file("pages/06_Train_and_Compare.py", default_timeout=30)
        inject_data_state(at, clf_df, target_col='condition', task_type='classification')
        at.run()
        assert_no_exception(at, "Train (classification)")

    def test_feature_selection_classification(self, clf_df):
        at = AppTest.from_file("pages/04_Feature_Selection.py", default_timeout=30)
        inject_data_state(at, clf_df, target_col='condition', task_type='classification')
        at.run()
        assert_no_exception(at, "Feature Selection (classification)")


# ── UI Element Checks ────────────────────────────────────────────────

class TestUIElements:
    """Verify key UI elements, disclosures, and banners are present."""

    @pytest.fixture(scope="class")
    def df(self):
        return build_test_dataframe()

    def test_feature_selection_categorical_disclosure(self, df):
        """#36: Feature selection should disclose categorical exclusion."""
        at = AppTest.from_file("pages/04_Feature_Selection.py", default_timeout=30)
        inject_data_state(at, df)
        at.run()
        assert_no_exception(at, "Feature Selection")
        text = all_text(at)
        assert "non-numeric" in text or "excluded from ranking" in text or "categorical" in text, \
            "Categorical exclusion disclosure missing (#36)"

    def test_preprocess_execution_banner(self, df):
        """Preprocess page should show execution order banner."""
        at = AppTest.from_file("pages/05_Preprocess.py", default_timeout=30)
        inject_data_state(at, df)
        at.run()
        assert_no_exception(at, "Preprocess")
        info_texts = [el.value for el in at.info]
        assert any("not applied yet" in t for t in info_texts), \
            "Preprocess execution order banner missing"

    def test_hypothesis_fwer_warning(self, df):
        """#43: Multiple tests should trigger family-wise error rate warning."""
        at = AppTest.from_file("pages/09_Hypothesis_Testing.py", default_timeout=30)
        inject_data_state(at, df)
        at.session_state['custom_table1_tests'] = [
            {'variable': 'age', 'test': 't-test', 'statistic': 't=2.1', 'p_value': 0.03, 'note': ''},
            {'variable': 'bmi', 'test': 't-test', 'statistic': 't=1.8', 'p_value': 0.07, 'note': ''},
            {'variable': 'chol', 'test': 'ANOVA', 'statistic': 'F=3.2', 'p_value': 0.04, 'note': ''},
        ]
        at.run()
        assert_no_exception(at, "Hypothesis Testing")
        warning_texts = [el.value.lower() for el in at.warning]
        assert any("multiple comparisons" in t or "family-wise" in t for t in warning_texts), \
            "FWER warning missing for 3+ tests (#43)"

    def test_theory_clipping_vs_trimming(self):
        """#25: Theory reference should include clipping vs trimming section."""
        at = AppTest.from_file("pages/11_Theory_Reference.py", default_timeout=30)
        at.run()
        assert_no_exception(at, "Theory Reference")
        text = all_text(at)
        assert "trimming" in text or "clipping" in text, \
            "Clipping vs trimming section missing (#25)"

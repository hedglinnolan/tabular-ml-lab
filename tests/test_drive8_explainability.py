"""Drive 8's Classic surfacing findings — one test per finding.

Each fails if its fix is reverted:

- `DRIVE-065` — page 07 handed the RAW feature frame to a bare estimator
  whenever the held-out rows could not be re-read by label, so every
  permutation and SHAP run on a dataset with a categorical predictor died on
  "could not convert string to float: 'female'". The page then printed
  "✅ Explainability analysis complete" unconditionally and wrote a methodology
  entry reading "Ran  on 3 models", which ticked two TRIPOD items from an
  empty run.
- `DRIVE-066` — the Methods' sample-to-feature ratio was computed over every
  uploaded row (809:1 = 21,849/27) inside a draft describing a 6,297-row
  analysis cohort.
- `DRIVE-069` — the seal chip said "15% (n=945, stratified)" and never said
  15% OF WHAT; the Methods draft had the basis ("of eligible observations")
  all along.
- `DRIVE-070` — the model badges quoted the profile's row count (20,904)
  beside a 4,407-row training set, and refused SVC on that basis; the coaching
  panel listed the TARGET among ">30% missing features" and advised dropping
  or imputing it.
- finding 12 — the ≥2-opens warning asserted that a choice had been made after
  seeing a held-out number, over a second opening the workflow makes by design.
- finding 13 / `MISC-092` — "VIF … changes nothing. No open observation is
  waiting on it", printed by the run that had just closed two of them.
- finding 19 — numeric/categorical counted three ways on one screen.
- finding 25 — "scored against 1 time(s) already" beside a chip that had
  already said the count.
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PAGE_02 = "pages/02_EDA.py"
PAGE_06 = "pages/06_Train_and_Compare.py"
PAGE_07 = "pages/07_Explainability.py"


def source(path):
    with open(os.path.join(PROJECT_ROOT, path)) as fh:
        return fh.read()


@pytest.fixture(autouse=True)
def clean():
    st.session_state.clear()
    yield
    st.session_state.clear()


class _Captured:
    """Collects what a render call emits, in bare mode."""

    def __init__(self, monkeypatch):
        self.warnings = []
        self.captions = []
        self.infos = []
        monkeypatch.setattr(st, "warning",
                            lambda msg, **kw: self.warnings.append(str(msg)))
        monkeypatch.setattr(st, "caption",
                            lambda msg, **kw: self.captions.append(str(msg)))
        monkeypatch.setattr(st, "info",
                            lambda msg, **kw: self.infos.append(str(msg)))

    @property
    def text(self):
        return " ".join(self.warnings + self.captions + self.infos)


# ══════════════════════════════════════════════════════════════════════════
# DRIVE-065 · the explainability run, driven through the page
# ══════════════════════════════════════════════════════════════════════════

def _categorical_study(n=200, seed=0):
    """A frame with a categorical predictor — the shape that broke."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.normal(50, 12, n),
        "bmi": rng.normal(27, 4, n),
        "gender": rng.choice(["male", "female"], n),
    })
    df["condition"] = (
        df["age"] + (df["gender"] == "female") * 9 + rng.normal(0, 4, n) > 55
    ).astype(int)
    return df


def _explainability_app(df, *, with_fitted_pipeline: bool):
    """Page 07 with one trained model, at the state page 06 leaves.

    `with_fitted_pipeline=False` is the forced-failure case: an estimator with
    no preprocessing recorded for it, which is what a bare estimator over a
    raw frame really looks like.
    """
    from tests.integration.conftest import inject_data_state

    feat = ["age", "bmi", "gender"]
    n_train = int(len(df) * 0.7)
    X, y = df[feat], df["condition"]
    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_test, y_test = X.iloc[n_train:], y.iloc[n_train:]

    prep = Pipeline([("ct", ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler())]),
         ["age", "bmi"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["gender"]),
    ]))])
    X_train_t = prep.fit_transform(X_train)
    model = LogisticRegression(max_iter=500).fit(X_train_t, y_train)

    at = AppTest.from_file(PAGE_07, default_timeout=180)
    inject_data_state(at, df, target_col="condition", task_type="classification")
    at.session_state["X_train"] = X_train
    at.session_state["y_train"] = y_train
    at.session_state["X_test"] = X_test
    at.session_state["y_test"] = y_test.values
    at.session_state["feature_names"] = feat
    at.session_state["selected_features"] = feat
    at.session_state["trained_models"] = {"logreg": model}
    at.session_state["fitted_estimators"] = {"logreg": model}
    at.session_state["model_results"] = {"logreg": {
        "metrics": {"accuracy": 0.8},
        "y_test": y_test.values,
        "y_test_pred": model.predict(prep.transform(X_test)),
    }}
    if with_fitted_pipeline:
        from ml.pipeline import get_feature_names_after_transform
        at.session_state["fitted_preprocessing_pipelines"] = {"logreg": prep}
        at.session_state["feature_names_by_model"] = {
            "logreg": get_feature_names_after_transform(prep, feat)}
    return at


def _run_analyses(at):
    at.run()
    at.checkbox(key="run_pdp").set_value(False)
    at.run()
    button = [b for b in at.button if "Run Selected" in str(b.label)][0]
    button.click().run()
    return at


class TestDrive065ExplainabilityRunsThroughTheFittedPipelines:
    def test_categorical_predictors_no_longer_kill_every_analysis(self):
        """The defect, driven: a one-hot-encoded model and a raw test frame.

        `_get_pipeline_and_data` dropped the fitted preprocessing pipeline and
        returned the bare estimator whenever the held-out rows could not be
        re-read by label. The estimator was fitted on the pipeline's OUTPUT, so
        every analysis raised on the numeric cast of 'female'.
        """
        at = _run_analyses(_explainability_app(_categorical_study(),
                                               with_fitted_pipeline=True))
        assert not at.exception
        assert at.session_state["permutation_importance"], (
            "permutation importance produced nothing on a model whose "
            "pipeline one-hot encodes its only categorical predictor")
        assert at.session_state["shap_results"], "SHAP produced nothing"
        issues = " ".join(el.value for el in at.get("text"))
        assert "could not convert string to float" not in issues

    def test_the_banner_reports_per_analysis_outcomes(self):
        at = _run_analyses(_explainability_app(_categorical_study(),
                                               with_fitted_pipeline=True))
        banner = " ".join(el.value for el in at.success)
        assert "Explainability analysis complete" in banner
        assert "permutation importance (1/1 models)" in banner, (
            "the green banner names no analysis, so it cannot be checked "
            "against what ran")
        assert "SHAP (1/1 models)" in banner

    def test_a_failed_run_gets_no_green_banner(self):
        """The forced-failure case: nothing computed, nothing claimed."""
        at = _run_analyses(_explainability_app(_categorical_study(),
                                               with_fitted_pipeline=False))
        assert not at.exception
        assert not at.session_state["permutation_importance"]
        assert not at.session_state["shap_results"]
        assert not [el for el in at.success
                    if "Explainability analysis complete" in el.value], (
            "every analysis failed and the page reported success")
        errors = " ".join(el.value for el in at.error)
        assert "produced no results" in errors
        assert "0 of 2 analyses completed" in errors

    def test_a_failed_run_records_no_methodology_entry(self):
        """The empty run ticked TRIPOD 15a and 19a through this entry."""
        at = _run_analyses(_explainability_app(_categorical_study(),
                                               with_fitted_pipeline=False))
        log = at.session_state["methodology_log"]
        explain = [e for e in log if e.get("step") == "Explainability"]
        assert explain == [], (
            f"an empty explainability run was logged as work done: {explain}")

    def test_a_successful_run_records_only_the_models_with_results(self):
        at = _run_analyses(_explainability_app(_categorical_study(),
                                               with_fitted_pipeline=True))
        log = at.session_state["methodology_log"]
        explain = [e for e in log if e.get("step") == "Explainability"]
        assert explain, "a run that produced results recorded nothing"
        action = explain[-1]["action"]
        assert "Ran  on" not in action, (
            "the analysis name is empty — the record is describing an empty run")
        assert explain[-1]["details"]["models"] == ["logreg"]

    def test_the_pipeline_is_never_dropped_from_the_returned_estimator(self):
        """The root cause, read at the source.

        Both exits of the fitted-pipeline branch must return the composed
        pipeline. A `return estimator, X_test, ...` inside it is the defect.
        """
        text = source(PAGE_07)
        start = text.index("def _get_pipeline_and_data")
        body = text[start:text.index("def _to_dense_numpy", start)]
        branch_start = body.index("if name in st.session_state.get("
                                  "'fitted_preprocessing_pipelines'")
        fallback = body.index("return full_pipeline, X_test, y_test, X_test")
        branch = body[branch_start:fallback]
        assert "return estimator, X_test, y_test, X_test" not in branch, (
            "the branch that HAS a fitted preprocessing pipeline returns the "
            "bare estimator, which is handed the raw frame downstream")


# ══════════════════════════════════════════════════════════════════════════
# DRIVE-066 · the sample-to-feature ratio's denominator
# ══════════════════════════════════════════════════════════════════════════

class TestDrive066TheRatioDescribesTheAnalysisCohort:
    def test_the_page_divides_by_the_target_complete_rows(self):
        text = source(PAGE_02)
        assert "n_p_ratio = _analysis_n / max(regime.n_features, 1)" in text, (
            "the sample-to-feature ratio is computed over every uploaded row")
        assert "_analysis_n = (int(df[target_col].notna().sum())" in text

    def test_both_sentences_name_the_denominator(self):
        text = source(PAGE_02)
        block = text[text.index('id="eda_opportunity_high_np"'):]
        block = block[:block.index("# Classification: balanced classes")]
        assert "_ANALYSIS_POP_PROSE" in block, (
            "the manuscript sentence quotes a ratio without saying what "
            "population it was computed on")
        assert block.count("_ANALYSIS_POP_PROSE") >= 2, (
            "the on-screen finding and the manuscript text must both name it")

    def test_the_narrative_rewrite_still_matches_the_finding(self):
        """`ml/narrative_engine.py` captures what is between the parentheses.

        The denominator travels INSIDE them, so the draft carries the ratio
        and its population together instead of the raw insight text leaking.
        """
        finding = ("Large sample-to-feature ratio (233:1 — 6,297 observations "
                   "with a recorded outcome over 27 predictors) — plenty of "
                   "data relative to complexity.")
        out = re.sub(
            r"Large sample-to-feature ratio \(([^)]+)\) — plenty of data "
            r"relative to complexity\.",
            r"The sample-to-feature ratio was \1, supporting model estimation "
            r"relative to predictor dimensionality.",
            finding, flags=re.IGNORECASE)
        assert out.startswith("The sample-to-feature ratio was 233:1")
        assert "6,297 observations with a recorded outcome" in out

    def test_the_sufficiency_sentences_use_the_same_population(self):
        text = source(PAGE_02)
        assert "_suff_ratio = _analysis_n / max(_cands.screened, 1)" in text
        for stale in ("({regime.n_rows:,} rows, {_cands.screened} candidate",
                      "({regime.n_rows:,} observations, {_cand_text})"):
            assert stale not in text, (
                f"a sufficiency sentence still counts every uploaded row: {stale}")


# ══════════════════════════════════════════════════════════════════════════
# DRIVE-069 · 15% of what
# ══════════════════════════════════════════════════════════════════════════

def _seal(n=400, missing=0, seed=3):
    """A sealed lockbox over `n` rows, `missing` of them without an outcome."""
    from utils.session_state import DataConfig
    from utils.test_lockbox import ensure_lockbox

    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"age": rng.integers(20, 80, n),
                       "y": rng.integers(0, 2, n).astype(float)})
    if missing:
        df.loc[df.index[:missing], "y"] = np.nan
    st.session_state["data_config"] = DataConfig(
        target_col="y", feature_cols=["age"], task_type="classification")
    return df, ensure_lockbox(df, "y", "classification")


class TestDrive069TheChipStatesItsDenominator:
    def test_the_chip_says_fifteen_percent_of_what(self, monkeypatch):
        from utils.test_lockbox import render_lockbox_status

        df, lb = _seal(n=400, missing=250)
        assert lb["n_total"] == 150, "eligibility is y.notna()"
        cap = _Captured(monkeypatch)
        render_lockbox_status()
        assert "of eligible rows" in cap.text
        assert "150 rows with a value for `y`" in cap.text, (
            "the chip states a percentage and never the base it is a "
            "percentage OF — a reader computes it against the upload")

    def test_a_record_without_the_field_says_nothing_rather_than_guessing(
            self, monkeypatch):
        from utils.test_lockbox import get_lockbox, render_lockbox_status

        _seal(n=400, missing=250)
        get_lockbox().pop("n_total")
        cap = _Captured(monkeypatch)
        render_lockbox_status()
        assert "of eligible rows" not in cap.text
        assert "rows with a value for" not in cap.text


# ══════════════════════════════════════════════════════════════════════════
# DRIVE-070 · the badges' n, and the target in the missingness advice
# ══════════════════════════════════════════════════════════════════════════

class _TargetProfile:
    def __init__(self, task_type="classification"):
        self.task_type = task_type
        self.minority_class_size = 600
        self.is_imbalanced = False
        self.class_balance_ratio = 1.2
        self.has_outliers = False
        self.outlier_rate = 0.0


class _Profile:
    """The shape `model_viability` reads, with the study-wide row count."""

    def __init__(self, n_rows, n_features, task_type="classification"):
        self.n_rows = n_rows
        self.n_features = n_features
        self.p_n_ratio = n_features / n_rows
        self.target_profile = _TargetProfile(task_type)
        self.events_per_variable = 24.2
        self.n_features_with_missing = 0
        self.highly_skewed_features = []


class TestDrive070TheBadgesQuoteTheTrainingSize:
    def test_the_realized_training_size_wins_over_the_profile(self):
        from ml.model_coach import model_viability

        profile = _Profile(n_rows=20904, n_features=27)
        v = model_viability(profile, n_train=4407)
        assert "4,407" in v["nn"][1]
        assert "20,904" not in " ".join(c for _, c in v.values()), (
            "a badge quotes a row count no model on this run is fitted on")

    def test_svc_is_not_refused_on_a_size_the_run_does_not_have(self):
        from ml.model_coach import model_viability

        profile = _Profile(n_rows=20904, n_features=27)
        assert model_viability(profile)["svc"][0] == "poor", (
            "sanity: 20,904 is over the kernel-cost threshold")
        assert model_viability(profile, n_train=4407)["svc"][0] != "poor", (
            "SVC refused for being slow at a size this study never trains on")

    def test_the_session_supplies_the_number_when_the_caller_does_not(self):
        from ml.model_coach import model_viability, realized_training_n

        assert realized_training_n() is None, "no split drawn yet"
        st.session_state["X_train"] = pd.DataFrame({"a": range(4407)})
        assert realized_training_n() == 4407
        v = model_viability(_Profile(n_rows=20904, n_features=27))
        assert "4,407" in v["nn"][1]

    def test_the_profile_still_answers_before_a_split_exists(self):
        from ml.model_coach import model_viability

        v = model_viability(_Profile(n_rows=20904, n_features=27))
        assert "20,904" in v["nn"][1]

    def test_page_06_passes_the_training_size_explicitly(self):
        """The session fallback is for other callers; the badge page says it."""
        text = source(PAGE_06)
        assert "n_train=len(X_train)," in text, (
            "the Train & Compare badges leave the coach to guess which n it "
            "is describing")


class TestDrive070TheTargetIsNotAFeature:
    def test_the_missingness_card_excludes_the_outcome(self):
        text = source(PAGE_02)
        block = text[text.index("# Missing data — synthesize into severity"):]
        block = block[:block.index('id="eda_missing_moderate"')]
        assert "_missing_candidates" in block, (
            "the >30%-missing card is built straight from "
            "signals.high_missing_cols, which includes the target")
        assert "not (_has_target and c == target_col)" in text

    def test_a_71_percent_missing_target_raises_no_feature_card(self):
        """The exact shape of the drive: the outcome is the worst column."""
        from ml.eda_recommender import compute_dataset_signals

        rng = np.random.default_rng(1)
        n = 1000
        df = pd.DataFrame({
            "age": rng.normal(50, 10, n),
            "meds_hbp": rng.integers(0, 2, n).astype(float),
        })
        df.loc[df.index[:712], "meds_hbp"] = np.nan
        signals = compute_dataset_signals(df, "meds_hbp", "classification",
                                          "cross_sectional", None,
                                          feature_cols=["age"])
        assert "meds_hbp" in signals.high_missing_cols, (
            "sanity: the target really is the high-missing column")
        candidates = [c for c in signals.high_missing_cols if c != "meds_hbp"]
        assert candidates == [], (
            "with the target removed there is no feature to advise about, so "
            "no card may be raised")


# ══════════════════════════════════════════════════════════════════════════
# finding 12 · the open counter says what it measured
# ══════════════════════════════════════════════════════════════════════════

class TestFinding12TheOpenCounterDoesNotAssertAChoiceItCannotSee:
    def test_the_by_design_pair_is_not_a_red_warning(self, monkeypatch):
        from utils.test_lockbox import record_lockbox_open, render_lockbox_status

        _seal()
        record_lockbox_open("Train & Compare")
        record_lockbox_open("Sensitivity Analysis (seed sweep, re-split over "
                            "the sealed rows)")
        cap = _Captured(monkeypatch)
        render_lockbox_status()
        assert cap.warnings == [], (
            "one final scoring run plus the page's own disclosed seed sweep "
            f"raised a warning: {cap.warnings}")
        joined = " ".join(cap.infos)
        assert "accessed 2 times" in joined
        assert "by design" in joined
        assert "reads better than it will on new data" not in " ".join(
            cap.warnings), (
            "the causal claim survived on a session that does not support it")

    def test_the_by_design_notice_still_names_the_sources_and_the_risk(
            self, monkeypatch):
        from utils.test_lockbox import record_lockbox_open, render_lockbox_status

        _seal()
        record_lockbox_open("Train & Compare")
        record_lockbox_open("Sensitivity Analysis (seed sweep, re-split over "
                            "the sealed rows)")
        cap = _Captured(monkeypatch)
        render_lockbox_status()
        joined = " ".join(cap.infos)
        assert "Train & Compare" in joined
        assert "Sensitivity Analysis" in joined
        assert "a choice made from now on" in joined, (
            "the conditional risk is the part worth keeping")

    def test_repeated_scoring_stays_loud(self, monkeypatch):
        from utils.test_lockbox import record_lockbox_open, render_lockbox_status

        _seal()
        record_lockbox_open("Train & Compare")
        record_lockbox_open("Train & Compare")
        cap = _Captured(monkeypatch)
        render_lockbox_status()
        assert cap.warnings, "two scoring runs against the sealed set are the "\
                             "thing this warning exists for"
        joined = " ".join(cap.warnings)
        assert "opened 2 times" in joined
        assert "2 were scoring runs" in joined
        assert "reads better than it will on new data" in joined

    def test_a_third_scoring_run_after_a_sweep_is_loud_again(self, monkeypatch):
        from utils.test_lockbox import record_lockbox_open, render_lockbox_status

        _seal()
        record_lockbox_open("Train & Compare")
        record_lockbox_open("Sensitivity Analysis (seed sweep)")
        record_lockbox_open("Train & Compare")
        cap = _Captured(monkeypatch)
        render_lockbox_status()
        assert cap.warnings, "the second scoring run must re-arm the warning"
        assert "Train & Compare (2×)" in " ".join(cap.warnings)


# ══════════════════════════════════════════════════════════════════════════
# finding 13 / MISC-092 · the diagnostic that DOES close something
# ══════════════════════════════════════════════════════════════════════════

class TestFinding13TheVifDisclosureStatesItsCarveOut:
    def test_a_run_that_closed_observations_does_not_claim_it_changed_nothing(
            self):
        from ml.eda_actions import diagnostic_disclosure

        said = diagnostic_disclosure("VIF (Multicollinearity)", n_open=0,
                                     n_closed=2)
        assert "No open observation is waiting on it" not in said, (
            "the sentence denies a carve-out the same press just applied")
        assert "2 observations" in said
        assert "recorded as addressed by it" in said

    def test_the_read_only_case_is_unchanged(self):
        from ml.eda_actions import diagnostic_disclosure

        said = diagnostic_disclosure("Leakage Detection", n_open=0)
        assert said == ("Leakage Detection reads the data and reports; it "
                        "changes nothing. No open observation is waiting on it.")

    def test_open_and_closed_are_both_reported(self):
        from ml.eda_actions import diagnostic_disclosure

        said = diagnostic_disclosure("VIF (Multicollinearity)", n_open=1,
                                     n_closed=2)
        assert "2 observations" in said
        assert "A further 1 observation" in said
        assert "stays **open**" in said

    def test_the_page_counts_what_the_carve_out_resolved(self):
        text = source(PAGE_02)
        block = text[text.index("_n_closed_here = 0"):]
        block = block[:block.index("_disclosure = _resolve_insights")]
        assert "_n_closed_here += 1" in block, (
            "the VIF resolver closes insights and counts none of them")
        assert "_resolve_insights_from_eda_result(\n" in text
        assert "action_id, result, title, _n_closed_here)" in text


# ══════════════════════════════════════════════════════════════════════════
# finding 19 · one counting rule
# ══════════════════════════════════════════════════════════════════════════

class TestFinding19OneCountingRuleForColumnTypes:
    def test_the_page_builds_the_two_lists_once(self):
        text = source(PAGE_02)
        assert text.count(
            "numeric_features = [c for c in feature_cols") == 1
        assert ("numeric_features = [f for f in feature_cols if f in "
                "df.columns and pd.api.types.is_numeric_dtype(df[f])]"
                not in text), (
            "the distribution filter recomputes the split with a rule that "
            "calls a bool column numeric, and offered 'Numeric (25)' beside a "
            "tile reading 19")

    def test_the_tiles_read_the_shared_lists(self):
        text = source(PAGE_02)
        assert 'st.metric("Numeric", f"{len(numeric_features)}"' in text
        assert 'st.metric("Categorical", f"{len(cat_features)}"' in text
        assert "regime.n_numeric" not in text
        assert "regime.n_categorical" not in text

    def test_the_rule_is_the_pipelines_own_and_is_stated(self):
        text = source(PAGE_02)
        assert "from data_processor import get_numeric_columns" in text, (
            "page 05 splits columns with get_numeric_columns; a second rule "
            "here is a second answer to one question")
        assert "_TYPE_COUNT_RULE" in text

    def test_a_bool_column_counts_as_categorical_on_both_surfaces(self):
        """The rule, applied: this is why 19 and 25 differed."""
        from data_processor import get_numeric_columns

        df = pd.DataFrame({"age": [1.0, 2.0], "flag": [True, False],
                           "sex": ["m", "f"]})
        numeric = set(get_numeric_columns(df))
        assert "flag" not in numeric, (
            "the pipeline one-hot encodes a bool column, so it is categorical")
        assert pd.api.types.is_numeric_dtype(df["flag"]), (
            "sanity: the rule this replaces would have called it numeric")


# ══════════════════════════════════════════════════════════════════════════
# finding 25 · the plural placeholder, and the count said twice
# ══════════════════════════════════════════════════════════════════════════

class TestFinding25TheReopenNoticeIsWrittenInWords:
    def test_no_plural_placeholder_ships(self):
        from utils.test_lockbox import reopen_notice

        assert "time(s)" not in reopen_notice(1)
        assert "time(s)" not in reopen_notice(3)
        assert "time(s)" not in source(PAGE_06)

    def test_the_notice_does_not_repeat_the_count_the_chip_gives(
            self, monkeypatch):
        from utils.test_lockbox import (record_lockbox_open, render_lockbox_status,
                                        reopen_notice)

        _seal()
        record_lockbox_open("Train & Compare")
        cap = _Captured(monkeypatch)
        render_lockbox_status(reopen_notice())
        chip = " ".join(cap.captions)
        assert "opened once, at Train & Compare" in chip
        assert "already been scored against once" in chip
        assert "1 time" not in chip

    def test_the_zero_case_is_unchanged(self):
        from utils.test_lockbox import reopen_notice

        assert reopen_notice(0) == ("Training on this page opens the held-out "
                                    "test set.")

    def test_page_06_uses_the_shared_composer(self):
        text = source(PAGE_06)
        assert "reopen_notice" in text
        assert "they have been scored " not in text

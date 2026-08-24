"""Where the paper describes the intended behavior, the app now matches it.

One test per item of the Wave-3 alignment sprint. Each fails if its fix is
reverted:

- pages/10  "explainability visualizations, diagnostic plots, calibration
            figures … automatically incorporated or exported alongside" — the
            plot switch defaulted OFF, so a default export was figureless, and
            calibration produced metrics with no figure at all.
- pages/09  "tests are selected according to … distributional assumptions" —
            the parametric/non-parametric switch defaulted to PARAMETRIC
            whatever the data looked like.
- pages/08  "for every trained model, the application summarizes mean
            performance, SD, range, CV" — one user-picked model per run.
- STATE-041 `exploratory_mode` travelled only as a deferred widget key that
            page 01 alone claimed, so `is_exploratory()` read False on every
            other page of a restored exploratory session.
- MINE-030  seeds whose fit raised were swallowed, and the REQUESTED seed count
            travelled to the manuscript as the size of the analysis.
- SWEEP-023 `or 42` rewrote random seed 0 in the reproducibility manifest.

Pages are Streamlit scripts: importing one executes it. Tests that need page
behavior lift the named function (or constant) out of the page's AST and
execute that definition alone, against the real modules it collaborates with.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st

from tests.test_session_manager import fake_session  # noqa: F401

REPO = Path(__file__).resolve().parent.parent
PAGE_08 = REPO / "pages" / "08_Sensitivity_Analysis.py"
PAGE_09 = REPO / "pages" / "09_Hypothesis_Testing.py"
PAGE_10 = REPO / "pages" / "10_Report_Export.py"


@pytest.fixture(autouse=True)
def clean_session():
    st.session_state.clear()
    yield
    st.session_state.clear()


def source(path) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_from_page(path, names, extra_globals=None):
    """Execute the named top-level defs/assignments of a page, alone.

    Importing a Streamlit page runs the whole script (and needs a session); the
    functions under test are ordinary module-level code, so they are compiled
    from the page's own AST into a namespace holding the collaborators they
    close over.
    """
    tree = ast.parse(source(path))
    wanted = set(names)
    picked = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted:
            picked.append(node)
        elif isinstance(node, ast.Assign):
            targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if targets & wanted:
                picked.append(node)
    missing = wanted - {getattr(n, "name", None) for n in picked} - {
        t.id for n in picked if isinstance(n, ast.Assign)
        for t in n.targets if isinstance(t, ast.Name)}
    assert not missing, f"{Path(path).name} no longer defines {sorted(missing)}"

    namespace = dict(extra_globals or {})
    exec(compile(ast.Module(body=picked, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def call_args(path, func_name, first_arg):
    """Return the keywords of the call to `func_name` whose first positional
    argument is the literal `first_arg`."""
    for node in ast.walk(ast.parse(source(path))):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
        if name != func_name or not node.args:
            continue
        if isinstance(node.args[0], ast.Constant) and node.args[0].value == first_arg:
            return {kw.arg: kw.value for kw in node.keywords}
    return None


# ── pages/10: figures ship with the numbers ───────────────────────────────

def test_page10_exports_figures_by_default_including_calibration():
    """(a) the plot switch defaults ON, and (b) calibration reliability
    diagrams are in the exportable set for classification runs."""
    from ml.calibration import calibration_classification, plot_calibration_curve

    # (a) The default export carries figures.
    kwargs = call_args(PAGE_10, "checkbox", "Include plots in zip")
    assert kwargs is not None, "the 'Include plots in zip' switch is gone"
    assert kwargs["value"].value is True, (
        "the default export contains no figures at all")
    cal_kwargs = call_args(PAGE_10, "checkbox", "Calibration plots")
    assert cal_kwargs is not None and cal_kwargs["value"].value is True, (
        "calibration figures are not in the default export set")

    # (b) The page's own adapter turns a stored calibration result into a
    # figure — for a dataclass and for the dict a restored session carries,
    # and not for regression, which has no reliability bins.
    ns = load_from_page(PAGE_10, ["calibration_result_for_plot"], {"np": np})
    adapt = ns["calibration_result_for_plot"]

    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 300)
    y_proba = np.clip(y_true * 0.6 + rng.normal(0.2, 0.2, 300), 0.01, 0.99)
    cal = calibration_classification(y_true, y_proba, model_name="ridge")

    assert adapt(cal) is not None
    assert plot_calibration_curve(adapt(cal)) is not None

    import dataclasses
    as_dict = dataclasses.asdict(cal)
    as_dict = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
               for k, v in as_dict.items()}
    restored = adapt(as_dict)
    assert restored is not None, "a restored session's calibration dict exports nothing"
    assert plot_calibration_curve(restored) is not None

    assert adapt({"model_name": "ridge", "task_type": "regression"}) is None
    assert adapt(None) is None

    # And the export loop actually writes them into the package.
    text = source(PAGE_10)
    assert "if export_plots_calibration:" in text
    assert 'zip_file.writestr(f"plots/calibration/{name}_calibration.png", plot_bytes)' in text


# ── pages/09: the data picks the test ─────────────────────────────────────

def test_page09_defaults_the_test_family_from_a_normality_precheck():
    """The parametric default follows Shapiro-Wilk on the selected columns,
    the reason is shown, and the choice is recorded with the result."""
    from ml.stats_tests import normality_check

    ns = load_from_page(
        PAGE_09,
        ["_parametric_default", "_NORMALITY_MIN_N", "_NORMALITY_ALPHA"],
        {"np": np, "normality_check": normality_check,
         "Dict": dict, "List": list, "Tuple": tuple},
    )
    default = ns["_parametric_default"]

    rng = np.random.default_rng(7)
    normal = rng.normal(50, 10, 400)
    skewed = rng.exponential(2.0, 400) ** 2

    ok, reason = default({"a": normal, "b": rng.normal(52, 10, 400)})
    assert ok is True, f"normal data was pushed to the non-parametric test ({reason})"
    assert "p=" in reason and "Shapiro" in reason

    not_ok, reason = default({"a": normal, "b": skewed})
    assert not_ok is False, "a skewed group still defaulted to the t-test"
    assert "b:" in reason and "p=" in reason, (
        "the screen cannot say WHY it chose the non-parametric test")

    # Too few observations to test is not evidence of normality.
    tiny, reason = default({"a": np.array([1.0, 2.0, 3.0])})
    assert tiny is False and "too few" in reason

    text = source(PAGE_09)
    # No test site defaults to parametric regardless of the data any more.
    for label in ("Use parametric test (t-test)",
                  "Use parametric test (ANOVA)",
                  "Use parametric test (paired t-test)"):
        kwargs = call_args(PAGE_09, "_parametric_choice", label)
        assert kwargs is None, "the label moved off the helper's keyword form"
    assert text.count("_parametric_choice(") == 4, (  # 1 def + 3 call sites
        "a test site is not going through the assumption pre-check")
    assert 'value=default' in text, "the checkbox no longer starts from the pre-check"
    # The choice is recorded wherever the result is.
    assert text.count("'assumption_basis': assumption_basis") >= 6, (
        "the selection basis is missing from a result dict, the methodology "
        "log, or the provenance record")


# ── pages/08: every trained model, not one ────────────────────────────────

def test_page08_seed_analysis_covers_every_trained_model():
    """The seed sweep loops over all seed-compatible models by default and
    presents the per-model mean/SD/range/CV table; the NN exclusion stands."""
    ns = load_from_page(
        PAGE_08,
        ["_seed_col", "_seed_summary_table", "_SEED_MODEL_SUFFIX"],
        {"pd": pd, "re": __import__("re")},
    )
    seed_col, summary = ns["_seed_col"], ns["_seed_summary_table"]

    assert seed_col("rmse", "ridge", "ridge") == "rmse"
    assert seed_col("rmse", "gbm", "ridge") == "rmse [gbm]"

    df_seeds = pd.DataFrame({
        "seed": [0, 1, 2, 3],
        "rmse": [1.0, 1.2, 0.8, 1.0],            # primary model, unsuffixed
        "rmse [gbm]": [2.0, 2.0, 2.0, 2.0],
        "rmse [rf]": [3.0, np.nan, 3.5, 4.5],
    })
    table = summary(df_seeds, "rmse", "ridge")
    assert set(table["Model"]) == {"RIDGE", "GBM", "RF"}, (
        "the summary covers fewer models than were re-seeded")
    assert list(table.columns) == ["Model", "Seeds", "Mean", "SD", "Min", "Max",
                                   "Range", "CV (%)"]
    gbm = table.set_index("Model").loc["GBM"]
    assert gbm["SD"] == 0 and gbm["CV (%)"] == 0 and gbm["Range"] == 0
    rf = table.set_index("Model").loc["RF"]
    assert rf["Seeds"] == 3, "a failed seed was counted in the model's summary"

    text = source(PAGE_08)
    scope = call_args(PAGE_08, "radio", "Models to re-seed")
    assert scope is not None, "the run no longer chooses a model scope"
    assert scope["index"].value == 0 and scope["options"].elts[0].value == "all", (
        "the default run no longer covers all trained models")
    assert "for model_key in models_to_seed:" in text, (
        "the seed loop is not per-model")
    # The NN exclusion and its on-screen disclosure are untouched.
    assert "_seed_compatible = [k for k in model_keys if k != 'nn']" in text
    assert ("ℹ️ Neural Network excluded from sensitivity analysis (PyTorch "
            "models cannot be cloned for re-seeding).") in text


# ── STATE-041: the quarantine regime arrives before the pages read it ─────

def test_state_041_exploratory_mode_is_claimed_before_any_page_reads_it():
    """Restore an exploratory session and land on a page that is not 01:
    `is_exploratory()` must be True before page 01 ever renders."""
    from utils import session_manager, theme
    from utils.test_lockbox import is_exploratory

    # What a restore leaves behind (pinned by
    # tests/test_session_manager.py::test_exploratory_mode_travels_via_widget_state
    # and exercised there against the real archive round trip).
    st.session_state["_pending_widget_state_restore"] = {
        "exploratory_mode": True, "workflow_mode_selector": "advanced"}
    assert "exploratory_mode" in session_manager._SAFE_WIDGET_KEYS
    assert is_exploratory() is False, "nothing has claimed the value yet"

    warnings = []
    real_warning = st.warning
    st.warning = lambda msg, *a, **k: warnings.append(msg)
    try:
        # The claim every page makes, before its own content.
        theme.apply_pending_exploratory_mode()
    finally:
        st.warning = real_warning

    assert is_exploratory() is True, (
        "a restored exploratory session still reads as quarantined off page 01")
    assert st.session_state.get("exploratory_used") is True, (
        "the manuscript watermark did not arrive with the mode")
    assert warnings and "exploratory mode" in warnings[0], (
        "quarantine-off arrived silently")
    # The unrelated deferred key is left for its own owner.
    assert st.session_state["_pending_widget_state_restore"] == {
        "workflow_mode_selector": "advanced"}

    # A session saved with quarantine ON restores unchanged and undisclosed.
    st.session_state.clear()
    st.session_state["_pending_widget_state_restore"] = {"exploratory_mode": False}
    theme.apply_pending_exploratory_mode()
    assert is_exploratory() is False
    assert "exploratory_used" not in st.session_state

    # And the claim is wired into the function every page calls.
    theme_src = source(REPO / "utils" / "theme.py")
    body = theme_src.split("def render_sidebar_workflow")[1]
    assert "apply_pending_exploratory_mode()" in body, (
        "the central claim is defined but never called")


# ── MINE-030: the analysis reports the seeds it achieved ─────────────────

def test_mine_030_page08_reports_achieved_seeds_not_requested():
    """Failed seeds are disclosed on screen and excluded from the count that
    reaches ml/publication.py as the size of the analysis."""
    text = source(PAGE_08)

    assert "failures_by_model[model_key] = [" in text, (
        "per-seed failures are not collected per model")
    assert "seeds failed for" in text, (
        "failed seeds are not disclosed the way the dropout loop discloses its own")

    # `n_seeds` is what the manuscript prints; it must be the achieved count.
    tree = ast.parse(text)
    logged = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "log_methodology"):
            keys = {kw.arg: kw.value for kw in node.keywords}
            details = keys.get("details")
            if not isinstance(details, ast.Dict):
                continue
            names = [k.value for k in details.keys if isinstance(k, ast.Constant)]
            if "n_seeds" in names:
                logged = dict(zip(names, details.values))
                break
    assert logged is not None, "the seed run no longer logs its size"
    assert {"n_seeds_requested", "n_seeds_succeeded", "n_seeds_failed"} <= set(logged), (
        "the achieved and requested seed counts are not both recorded")
    n_seeds_expr = ast.unparse(logged["n_seeds"])
    assert "_n_failed" in n_seeds_expr, (
        f"n_seeds still reports the requested count ({n_seeds_expr}) — "
        f"ml/publication.py prints it as the size of the analysis")
    assert ast.unparse(logged["n_seeds_succeeded"]) == n_seeds_expr


# ── SWEEP-023: seed 0 is a seed ──────────────────────────────────────────

def test_sweep_023_seed_zero_survives_export_to_the_manifest():
    """Seed 0 must reach the reproducibility manifest as 0, not 42."""
    from utils.session_state import SplitConfig

    class _FakeState(dict):
        pass

    class _FakeSt:
        session_state = _FakeState()

    ns = load_from_page(
        PAGE_10, ["_build_reproducibility_manifest"],
        {"st": _FakeSt(), "pd": pd, "get_data": lambda: None,
         "Dict": dict, "Any": object},
    )
    manifest = ns["_build_reproducibility_manifest"]

    _FakeSt.session_state.clear()
    _FakeSt.session_state["split_config"] = SplitConfig(
        train_size=0.7, val_size=0.15, test_size=0.15, random_state=0)
    assert manifest()["random_seed"] == 0, (
        "the manifest a reviewer re-runs the study from names a different seed")

    _FakeSt.session_state.clear()
    _FakeSt.session_state["random_seed"] = 0
    assert manifest()["random_seed"] == 0

    _FakeSt.session_state.clear()
    _FakeSt.session_state["split_config"] = SplitConfig(
        train_size=0.7, val_size=0.15, test_size=0.15, random_state=7)
    assert manifest()["random_seed"] == 7

    # Nothing recorded at all still falls back, and does not raise.
    _FakeSt.session_state.clear()
    assert manifest()["random_seed"] == 42


# ── MISC-102 · calibration PROSE reaches the manuscript, not just figures ──

class TestMisc102CalibrationProseReachesTheManuscript:
    """Page 06 calibrates every eligible model and stores the records; the
    composer carried a Calibration block no caller ever fed, and the LaTeX
    export printed a placeholder asking its own author for numbers the session
    already held."""

    @staticmethod
    def _records():
        from ml.calibration import CalibrationResult

        return {
            "rf": CalibrationResult(model_name="RF", task_type="classification",
                                    brier_score=0.12, ece=0.03,
                                    weak_slope=0.87, weak_intercept=-0.05),
            "ridge": CalibrationResult(model_name="RIDGE", task_type="regression",
                                       calibration_slope=0.98,
                                       calibration_intercept=0.40),
            "blank": CalibrationResult(model_name="B", task_type="classification"),
        }

    def test_the_composer_block_is_reachable_for_several_models(self):
        from ml.publication import generate_methods_section

        text = generate_methods_section(
            data_config={"feature_cols": ["a"], "target_col": "y"},
            preprocessing_config={}, model_configs={"rf": {}}, split_config={},
            n_total=100, n_train=70, n_val=15, n_test=15,
            feature_names=["a"], target_name="y", task_type="classification",
            metrics_used=["AUC"],
            selected_model_results={"rf": {"metrics": {"AUC": 0.80}},
                                    "ridge": {"metrics": {"AUC": 0.70}}},
            calibration_results=self._records(),
        )

        assert "### Calibration" in text, "the Calibration block never rendered"
        assert "Brier score = 0.1200" in text and "ECE = 0.0300" in text
        assert "weak calibration slope = 0.870" in text
        # The SECOND model is there too — one model's calibration is not a
        # statement about the others.
        assert "calibration slope = 0.980, intercept = 0.400" in text
        # A record with no computed metric contributes no sentence.
        assert text.count("**") >= 2

    def test_a_records_dict_with_nothing_in_it_opens_no_section(self):
        from ml.calibration import CalibrationResult
        from ml.publication import generate_methods_section

        text = generate_methods_section(
            data_config={"feature_cols": ["a"], "target_col": "y"},
            preprocessing_config={}, model_configs={}, split_config={},
            n_total=100, n_train=70, n_val=15, n_test=15,
            feature_names=["a"], target_name="y", task_type="classification",
            metrics_used=["AUC"],
            selected_model_results={"rf": {"metrics": {"AUC": 0.8}}},
            calibration_results={"rf": CalibrationResult(model_name="RF",
                                                         task_type="classification")},
        )
        assert "### Calibration" not in text, (
            "an empty artifact opened a section that then said nothing")

    def test_page10_hands_over_the_records_for_included_models_only(self):
        from typing import Any, Dict, List

        ns = load_from_page(
            PAGE_10,
            ["_calibration_records_for_manuscript", "_calibration_metrics_as_dict"],
            {"st": st, "np": np, "Dict": Dict, "Any": Any, "List": List},
        )
        st.session_state["calibration_results"] = self._records()

        kept = ns["_calibration_records_for_manuscript"](["rf", "ridge"])
        assert set(kept) == {"rf", "ridge"}
        assert ns["_calibration_records_for_manuscript"](["rf"]).keys() == {"rf"}, (
            "a manuscript may not report calibration for a model it excludes")
        assert ns["_calibration_records_for_manuscript"]([]) == {}

        metrics = ns["_calibration_metrics_as_dict"](self._records()["rf"])
        assert metrics["brier_score"] == 0.12
        assert "model_name" not in metrics and "task_type" not in metrics
        # A restored session's plain dict reads the same way.
        assert ns["_calibration_metrics_as_dict"](
            {"model_name": "rf", "ece": 0.03})["ece"] == 0.03

    def test_the_export_summary_carries_every_included_model(self):
        """The bridge between the stored records and the LaTeX subsection."""
        from typing import Any, Dict, List, Optional

        ns = load_from_page(
            PAGE_10,
            ["_build_explainability_summary_for_export",
             "_calibration_metrics_as_dict"],
            {"st": st, "np": np, "Dict": Dict, "Any": Any, "List": List,
             "Optional": Optional},
        )
        st.session_state["calibration_results"] = self._records()

        summary = ns["_build_explainability_summary_for_export"](
            {"manuscript_primary_model": "rf", "included_models": ["rf", "ridge"]})

        assert set(summary["calibration_by_model"]) == {"rf", "ridge"}, (
            "only the primary model's calibration reached the PDF")
        assert summary["calibration_metrics"]["brier_score"] == 0.12
        assert "blank" not in summary["calibration_by_model"]

    def test_the_latex_calibration_subsection_carries_every_model(self):
        from ml.latex_report import generate_latex_report

        tex = generate_latex_report(
            task_type="classification",
            model_results={"rf": {"metrics": {"AUC": 0.8}}},
            explainability_summary={
                "calibration_by_model": {
                    "rf": {"brier_score": 0.12, "ece": 0.03},
                    "ridge": {"calibration_slope": 0.98,
                              "calibration_intercept": 0.40},
                },
                "calibration_metrics": {"brier_score": 0.12},
            },
        )
        assert "PLACEHOLDER: Report calibration" not in tex, (
            "the export asked its author for numbers the session held")
        assert "Brier score = 0.1200" in tex
        assert "calibration slope = 0.980" in tex
        assert "Random Forest" in tex and "Ridge Regression" in tex
        assert "Calibration Metrics" not in tex, (
            "the primary model's metrics are stated twice")

    def test_the_placeholder_stays_when_nothing_was_computed(self):
        from ml.latex_report import generate_latex_report

        for task in ("classification", "regression"):
            tex = generate_latex_report(
                task_type=task, model_results={"rf": {"metrics": {"AUC": 0.8}}})
            assert "PLACEHOLDER: Report calibration" in tex, (
                "an absence is a real absence and keeps saying so")


# ── MISC-103 · the exploratory limitation survives the fallback path ──────

class TestMisc103TheExploratoryLimitationSurvivesTheFallback:
    """It reached the draft through NarrativeEngine alone. When the engine
    raises — or provenance is empty — pages/10 falls back to the composer, and
    the one limitation about the whole study disappeared without a word."""

    @staticmethod
    def _methods(**kwargs):
        from ml.publication import generate_methods_section

        return generate_methods_section(
            data_config={"feature_cols": ["a"], "target_col": "y"},
            preprocessing_config={}, model_configs={}, split_config={},
            n_total=100, n_train=70, n_val=15, n_test=15,
            feature_names=["a"], target_name="y", task_type="regression",
            metrics_used=["RMSE"], **kwargs)

    def test_the_fallback_composer_states_it(self):
        from ml.publication import EXPLORATORY_LIMITATION_SENTENCE

        text = self._methods(manuscript_context={"exploratory_mode": True})
        assert EXPLORATORY_LIMITATION_SENTENCE in text
        assert EXPLORATORY_LIMITATION_SENTENCE not in self._methods(
            manuscript_context={"exploratory_mode": False})

    def test_the_session_watermark_is_read_when_the_context_is_silent(self):
        from ml.publication import EXPLORATORY_LIMITATION_SENTENCE

        st.session_state["exploratory_used"] = True
        assert EXPLORATORY_LIMITATION_SENTENCE in self._methods(), (
            "'exploratory_used' is sticky precisely so toggling the mode off "
            "cannot launder results computed with it on")

    def test_the_validator_fails_a_draft_that_omits_it(self):
        from ml.manuscript_validator import validate_manuscript_bundle

        report = validate_manuscript_bundle(
            {"exploratory_mode": True}, methods_text="A clean held-out study.",
            report_text="", latex_text="", task_type="regression")
        check = self._check(report)
        assert check.status == "FAIL" and check.scored
        assert not report.passed, "a failing check must gate the export"

    def test_the_validator_accepts_either_producer_wording(self):
        from ml.manuscript_validator import validate_manuscript_bundle
        from ml.publication import EXPLORATORY_LIMITATION_SENTENCE

        narrative = ("Limitations: the analysis was run in exploratory mode: "
                     "the held-out test set was not quarantined from feature "
                     "engineering and selection, so reported performance may be "
                     "optimistically biased.")
        for draft in (EXPLORATORY_LIMITATION_SENTENCE, narrative):
            report = validate_manuscript_bundle(
                {"exploratory_mode": True}, methods_text=draft, report_text="",
                latex_text="", task_type="regression")
            assert self._check(report).status == "PASS"

    def test_a_clean_study_is_not_asked_for_a_sentence_it_does_not_owe(self):
        """The roster the panel counts is not padded with a check that is not
        about this manuscript — and a clean draft is never gated on it."""
        from ml.manuscript_validator import validate_manuscript_bundle

        report = validate_manuscript_bundle(
            {"exploratory_mode": False}, methods_text="x", report_text="",
            latex_text="", task_type="regression")
        assert not [c for c in report.checks if "Exploratory" in c.name]

    @staticmethod
    def _check(report):
        named = [c for c in report.checks if "Exploratory" in c.name]
        assert named, "the exploratory-limitation check is gone"
        return named[0]


# ── MISC-104 · four manuscript numbers, each over one universe ────────────

class TestMisc104TheManuscriptNumbersNameTheirUniverse:

    @staticmethod
    def _methods_with_log(log):
        from ml.publication import generate_methods_section

        st.session_state["methodology_log"] = log
        return generate_methods_section(
            data_config={"feature_cols": ["a"] * 5, "target_col": "y"},
            preprocessing_config={}, model_configs={}, split_config={},
            n_total=100, n_train=70, n_val=15, n_test=15,
            feature_names=["a", "b", "c", "sex", "site"], target_name="y",
            task_type="regression", metrics_used=["RMSE"])

    def test_the_selection_sentence_counts_both_universes(self):
        text = self._methods_with_log([
            {"step": "Feature Selection", "action": "ran",
             "details": {"methods": ["lasso", "rfe"],
                         "methods_completed": ["lasso", "rfe"],
                         "n_features_before": 10, "n_features_after": 3,
                         "consensus_threshold": 2}},
            {"step": "Feature Selection Applied", "action": "applied",
             "details": {"method": "consensus", "n_features_selected": 5,
                         "n_consensus_ranked": 3,
                         "carried_through_unranked": ["sex", "site"],
                         "consensus_threshold": 2}},
        ])

        assert "ranked 10 numeric candidate predictors and retained 3" in text
        assert "2 non-ranked feature(s) (sex and site) were carried through" in text
        assert "giving 5 predictors in the final modeling set" in text
        assert "reduced the feature set from 10 to 5" not in text, (
            "10 is the numeric universe and 5 includes carried categoricals")

    def test_the_threshold_denominator_is_the_methods_that_ran(self):
        text = self._methods_with_log([
            {"step": "Feature Selection", "action": "ran",
             "details": {"methods": ["lasso", "rfe", "univariate"],
                         "methods_completed": ["lasso", "rfe"],
                         "n_features_before": 10, "n_features_after": 3,
                         "consensus_threshold": 2}},
            {"step": "Feature Selection Applied", "action": "applied",
             "details": {"method": "consensus", "n_features_selected": 3,
                         "consensus_threshold": 2}},
        ])

        assert "at least 2 of the 2 method(s) that completed" in text
        assert "3 were requested; 1 did not complete" in text
        assert "at least 2 of 3 methods" not in text, (
            "the threshold came from the methods that RAN; the denominator "
            "came from the methods that were REQUESTED")
        # The named methods are the ones that voted.
        assert "univariate screening" not in text

    def test_an_unrecorded_completion_gets_no_denominator_at_all(self):
        text = self._methods_with_log([
            {"step": "Feature Selection", "action": "ran",
             "details": {"methods": ["lasso", "rfe"], "n_features_before": 10,
                         "n_features_after": 3, "consensus_threshold": 2}},
            {"step": "Feature Selection Applied", "action": "applied",
             "details": {"method": "consensus", "n_features_selected": 3,
                         "consensus_threshold": 2}},
        ])
        assert "at least 2 of the methods that completed." in text
        assert "of 2 methods" not in text

    def test_page04_records_which_methods_completed(self):
        text = source(REPO / "pages" / "04_Feature_Selection.py")
        assert "'methods_completed': list(methods_completed)" in text, (
            "without this key the manuscript cannot tell requested from run")
        assert "methods_completed.append(method)" in text

    def test_the_seed_stability_export_names_its_model(self):
        from typing import Any, Dict, List, Tuple

        ns = load_from_page(
            PAGE_10,
            ["_build_sensitivity_summary_for_export", "_seed_stability_records",
             "_seed_stability_by_model"],
            {"st": st, "np": np, "pd": pd, "Dict": Dict, "Any": Any,
             "List": List, "Tuple": Tuple,
             "_report_ledger": _NoLedger()},
        )

        st.session_state["sensitivity_seed_results"] = pd.DataFrame({
            "seed": [0, 1, 2, 3],
            "rmse": [1.00, 1.10, 0.95, 1.05],
            "rmse [ridge]": [1.30, 1.25, 1.35, 1.28],
        })
        st.session_state["methodology_log"] = [
            {"step": "Sensitivity Analysis", "action": "Ran seed stability analysis",
             "details": {"model": "ridge", "metric": "rmse", "n_seeds": 4}},
            {"step": "Sensitivity Analysis", "action": "Ran seed stability analysis",
             "details": {"model": "rf", "metric": "rmse", "n_seeds": 4}},
        ]

        summary = ns["_build_sensitivity_summary_for_export"]()
        stability = summary["seed_stability"]
        assert stability["model"] == "rf", (
            "the unsuffixed columns belong to the primary model, which page 08 "
            "logs last")
        assert stability["metric"] == "rmse"
        assert stability["n_seeds"] == 4
        by_model = {row["model"]: row for row in stability["by_model"]}
        assert set(by_model) == {"rf", "ridge"}, (
            "page 08 sweeps every model; the export carried one")
        assert by_model["ridge"]["mean"] == pytest.approx(1.295)

    def test_the_seed_sentence_is_unnamed_when_nothing_recorded_it(self):
        from typing import Any, Dict, List, Tuple

        ns = load_from_page(
            PAGE_10,
            ["_build_sensitivity_summary_for_export", "_seed_stability_records",
             "_seed_stability_by_model"],
            {"st": st, "np": np, "pd": pd, "Dict": Dict, "Any": Any,
             "List": List, "Tuple": Tuple, "_report_ledger": _NoLedger()},
        )
        st.session_state["sensitivity_seed_results"] = pd.DataFrame(
            {"seed": [0, 1], "rmse": [1.0, 1.2]})

        stability = ns["_build_sensitivity_summary_for_export"]()["seed_stability"]
        assert "model" not in stability, (
            "with no record of whose sweep it was, the export names no model "
            "rather than guessing one")

    def test_the_latex_seed_sentence_and_table_name_their_models(self):
        from ml.latex_report import generate_latex_report

        tex = generate_latex_report(sensitivity_summary={"seed_stability": {
            "cv_percent": 3.2, "range": "0.80 to 0.86", "metric": "roc_auc",
            "model": "rf", "n_seeds": 10, "by_model": [
                {"model": "rf", "metric": "roc_auc", "n_seeds": 10, "mean": 0.83,
                 "sd": 0.02, "min": 0.80, "max": 0.86, "cv_percent": 3.2},
                {"model": "ridge", "metric": "roc_auc", "n_seeds": 10,
                 "mean": 0.79, "sd": 0.03, "min": 0.74, "max": 0.83,
                 "cv_percent": 4.1}]}})

        assert "Random seed sensitivity analysis of" in tex
        assert "Random Forest" in tex and "across 10 seeds" in tex
        assert "tab:seed-stability" in tex and "Ridge Regression" in tex

    def test_table1_footnote_markers_point_at_a_note(self):
        from ml.latex_report import generate_latex_report

        table1 = pd.DataFrame({"Overall (N=100)": ["50 (10)", "20 (40%)"]},
                              index=["age^1", "sex"])
        note = "^1 Welch t-test: t=2.1, p=0.0400 (unequal variance)"

        tex = generate_latex_report(table1_df=table1, table1_footnotes=[note])
        assert "Welch t-test" in tex, (
            "the marker on the row label referred to a note the manuscript "
            "did not contain")
        assert tex.index("Welch t-test") > tex.index("bottomrule")

        # And nothing is invented when there are no custom tests.
        assert "Welch t-test" not in generate_latex_report(table1_df=table1)

    def test_page10_passes_the_footnotes_to_the_exporter(self):
        text = source(PAGE_10)
        assert "table1_footnotes=st.session_state.get('table1_custom_test_footnotes')" in text


class _NoLedger:
    """Stands in for the page's module-level insight ledger."""

    @staticmethod
    def get_methodology_log():
        return []


# ── MISC-105 · the copy describes what the code does ──────────────────────

class TestMisc105CopyMatchesShippedBehavior:

    def test_the_dropout_slider_does_not_promise_importance_order(self):
        kwargs = call_args(PAGE_08, "slider", "Max features to test")
        assert kwargs is not None, "the dropout slider is gone"
        help_text = kwargs["help"].value if hasattr(kwargs["help"], "value") else ""
        if not help_text:  # a joined implicit concatenation lands as a BinOp
            help_text = source(PAGE_08).split("Max features to test", 1)[1][:400]
        assert "column order" in help_text
        assert "top N by importance" not in help_text, (
            "the run takes feature_names[:max_features]")

    def test_the_dropout_copy_says_neutralize_not_remove(self):
        text = source(PAGE_08)
        assert "Neutralize one feature at a time" in text
        assert "(neutralizing them hurts performance)" in text
        assert "(removing them hurts performance)" not in text
        assert "removing it improved" not in text

    def test_the_plausibility_copy_uses_the_improbability_band_vocabulary(self):
        from ml.eda_actions import plausibility_check

        class _Signals:
            numeric_cols = []
            physio_plausibility_flags = ["glucose: unit mismatch suspected"]

        result = plausibility_check(
            pd.DataFrame({"x": [1.0, 2.0, 3.0]}), None, ["x"], _Signals(), {})
        manuscript = result["insights"][0].manuscript_text
        assert "improbability band" in manuscript
        assert "reference range" not in manuscript.lower(), (
            "MISC-018: p01-p99 is not a reference interval, and the word is "
            "not used for it")

        page = source(REPO / "pages" / "02_EDA.py")
        assert "NHANES reference ranges" not in page
        assert "improbability band" in page

    def test_the_no_consensus_advice_is_something_a_user_can_do(self):
        text = source(REPO / "pages" / "04_Feature_Selection.py")
        assert "Try lowering the threshold" not in text, (
            "the threshold has a floor of 2 and no control lowers it")
        assert "Manual feature selection** below" in text

    def test_a_large_unprofiled_run_gets_the_large_data_scheduler(self):
        from ml.dataset_profile import DataSufficiencyLevel
        from ml.nn_recommender import recommend_nn_config

        unprofiled = recommend_nn_config(
            n_samples=50_000, n_features=20, data_sufficiency=None,
            p_n_ratio=0.0004, task_type="regression")
        profiled = recommend_nn_config(
            n_samples=50_000, n_features=20,
            data_sufficiency=DataSufficiencyLevel.ABUNDANT,
            p_n_ratio=0.0004, task_type="regression")

        assert unprofiled.params["lr_scheduler"] == "cosine_warm_restarts", (
            "None means 'not profiled', not 'scarce' — a 50k-row study fell to "
            "the small-data schedule against this module's own rule")
        assert unprofiled.params["lr_scheduler"] == profiled.params["lr_scheduler"]

        small = recommend_nn_config(
            n_samples=200, n_features=20, data_sufficiency=None,
            p_n_ratio=0.1, task_type="regression")
        assert small.params["lr_scheduler"] == "reduce_on_plateau"

    def test_page06_computes_the_level_when_no_profile_exists(self):
        text = source(REPO / "pages" / "06_Train_and_Compare.py")
        assert "_data_suff, _ = assess_data_sufficiency(" in text

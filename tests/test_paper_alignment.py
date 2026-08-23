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

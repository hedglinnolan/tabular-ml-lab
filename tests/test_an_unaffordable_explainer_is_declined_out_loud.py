"""The explainability caps, and the record they leave behind.

This page's output is a manuscript. A cap that changes which models were
explained, or whether an importance ranking was computed at all, has to be
visible where it engages AND written to the ledger, or the Methods section
describes an analysis that did not occur.

What is asserted here:
  - a narrow frame (500 x 20) acquires no new friction whatsoever;
  - permutation importance stops defaulting ON above the calibrated width, and
    says what it would cost, without ever losing the option;
  - the model-agnostic kernel estimator is declined above its width, out loud,
    naming projected memory and the alternative that does work, and the refusal
    reaches the ledger with manuscript-register text;
  - TreeExplainer is deliberately left uncapped on the feature axis;
  - the Methods sentence names only the models each analysis actually reached.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests.integration.conftest import inject_data_state, inject_trained_state  # noqa: E402

PAGE_07 = os.path.join(PROJECT_ROOT, "pages", "07_Explainability.py")


def _frame(n, p, seed=11):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, p)),
                      columns=[f"feat_{i:05d}" for i in range(p)])
    df["target"] = df["feat_00001"] * 2.0 + rng.normal(0, 0.5, n)
    return df


def _page(df):
    at = AppTest.from_file("pages/07_Explainability.py", default_timeout=600)
    inject_data_state(at, df, target_col="target")
    inject_trained_state(at, df, target_col="target")
    return at


def _add_kernel_model(at, df, key="svr"):
    """A fitted estimator the registry routes to KernelExplainer."""
    from sklearn.svm import SVR
    from ml.eval import calculate_regression_metrics

    X_train, y_train = at.session_state["X_train"], at.session_state["y_train"]
    X_test, y_test = at.session_state["X_test"], at.session_state["y_test"]
    model = SVR(kernel="rbf").fit(X_train, y_train)
    pred = model.predict(X_test)
    at.session_state["trained_models"][key] = model
    at.session_state["fitted_estimators"][key] = model
    at.session_state["model_results"][key] = {
        "metrics": calculate_regression_metrics(y_test.values, pred),
        "y_test": y_test.values,
        "y_test_pred": pred,
    }
    return at


def _all_text(at):
    return " | ".join(
        [e.value for e in at.info if e.value]
        + [e.value for e in at.warning if e.value]
        + [e.value for e in at.caption if e.value]
        + [e.value for e in at.markdown if e.value]
    )


def _click(at, fragment):
    for b in at.button:
        if fragment in (b.label or ""):
            b.click()
            return at.run()
    raise AssertionError(f"No button matching {fragment!r}")


# ── the narrow case: nothing may change ─────────────────────────────────────

class TestANarrowFrameIsUntouched:
    """500 x 20. Every cap here is calibrated in the thousands of columns; a
    panel dataset must not pay one word of friction for guardrails aimed at
    omics."""

    def test_no_cap_speaks_and_permutation_importance_still_defaults_on(self):
        at = _page(_frame(n=500, p=20))
        at.run()

        assert not at.exception
        assert at.session_state["run_perm"] is True
        text = _all_text(at)
        for phrase in ("off by default at", "KernelExplainer on",
                       "SHAP was not computed", "Run KernelExplainer anyway"):
            assert phrase not in text, f"{phrase!r} surfaced on a 500x20 frame"
        # No confirmation control exists to be clicked.
        assert not [c for c in at.checkbox if c.key == "run_kernel_shap_confirm"]

    def test_the_run_still_produces_everything_and_records_no_caveat(self):
        at = _page(_frame(n=500, p=20))
        at.run()
        _click(at, "Run Selected Analyses")

        assert not at.exception
        assert at.session_state["permutation_importance"], "permutation importance did not run"
        assert at.session_state["shap_results"], "SHAP did not run"
        ledger = at.session_state["insight_ledger"]
        assert ledger.get("xai_perm_importance_not_run") is None
        assert not [i for i in ledger.get_unresolved()
                    if i.id.startswith("xai_kernel_shap_skipped")]
        # And the per-analysis model lists exist, so the Methods sentence has
        # something better than the flat OR to read.
        entry = [e for e in ledger.get_methodology_log()
                 if e.get("step") == "Explainability"][-1]
        assert entry["details"]["models_by_analysis"]["permutation_importance"] == ["ridge"]


# ── permutation importance: default off, never unavailable ──────────────────

class TestPermutationImportanceStopsBeingAutomatic:
    def test_it_defaults_off_above_the_width_and_quotes_the_price(self):
        at = _page(_frame(n=120, p=1200))
        at.run()

        assert not at.exception
        assert at.session_state["run_perm"] is False
        text = _all_text(at)
        assert "off by default at 1,200 features" in text, text
        assert "Tick to run it" in text, text

    def test_the_control_survives_and_the_user_can_still_turn_it_on(self):
        at = _page(_frame(n=120, p=1200))
        at.run()
        box = [c for c in at.checkbox if c.key == "run_perm"]
        assert box, "the permutation-importance checkbox was removed"
        box[0].set_value(True).run()
        assert at.session_state["run_perm"] is True
        # ...and a deliberate tick is not undone by the next rerun.
        at.run()
        assert at.session_state["run_perm"] is True

    def test_leaving_it_off_is_recorded_for_the_manuscript(self):
        at = _page(_frame(n=120, p=1200))
        at.run()
        assert at.session_state["run_perm"] is False
        _click(at, "Run Selected Analyses")

        assert not at.exception
        ins = at.session_state["insight_ledger"].get("xai_perm_importance_not_run")
        assert ins is not None, "a skipped importance analysis reached no record"
        assert ins.resolved is False
        assert ins.manuscript_text == "permutation feature importance was not computed"
        limitations = at.session_state["insight_ledger"].discussion_points_for_manuscript()
        assert any("permutation feature importance was not computed" in t
                   for t in limitations["limitations"])


# ── the kernel estimator: refused, out loud, and recorded ───────────────────

class TestTheKernelEstimatorIsDeclinedNotAttempted:
    def test_it_refuses_names_the_cost_and_names_the_alternative(self):
        at = _page(_frame(n=120, p=1200))
        _add_kernel_model(at, None)
        at.run()
        _click(at, "Run Selected Analyses")

        assert not at.exception
        warnings = " | ".join(w.value for w in at.warning if w.value)
        assert "SHAP was not computed for SVR" in warnings, warnings
        assert "1,200 features" in warnings, warnings
        assert "GB" in warnings, warnings
        # A refusal that does not say what still works is a dead end.
        assert "permutation importance remains available" in warnings, warnings
        assert "TreeExplainer models were explained normally" in warnings, warnings

    def test_the_refusal_reaches_the_ledger_and_the_methods_lists(self):
        at = _page(_frame(n=120, p=1200))
        _add_kernel_model(at, None)
        at.run()
        _click(at, "Run Selected Analyses")

        assert not at.exception
        ledger = at.session_state["insight_ledger"]
        ins = ledger.get("xai_kernel_shap_skipped_svr")
        assert ins is not None, "the kernel refusal never reached the ledger"
        assert ins.resolved is False
        assert "SHAP values were not computed for the SVR model" in ins.manuscript_text
        assert ins.metadata["n_features"] == 1200
        assert ins.metadata["policy"] == "refuse"
        # The model that WAS explained is still explained.
        assert "ridge" in at.session_state["shap_results"]
        assert "svr" not in at.session_state["shap_results"]
        entry = [e for e in ledger.get_methodology_log()
                 if e.get("step") == "Explainability"][-1]
        assert entry["details"]["models_by_analysis"]["shap"] == ["ridge"], entry["details"]

    def test_the_confirm_band_quotes_a_price_and_waits_to_be_told(self):
        """200 < p <= 1,000: it still finishes, so it is offered rather than
        refused — but it is no longer started on the user's behalf."""
        at = _page(_frame(n=120, p=800))
        _add_kernel_model(at, None)
        at.run()

        assert not at.exception
        text = _all_text(at)
        assert "KernelExplainer on 800 features" in text, text
        confirm = [c for c in at.checkbox if c.key == "run_kernel_shap_confirm"]
        assert confirm, "no way to say yes to a quoted job"
        assert confirm[0].value is False

        _click(at, "Run Selected Analyses")
        assert not at.exception
        assert "svr" not in at.session_state["shap_results"]
        ins = at.session_state["insight_ledger"].get("xai_kernel_shap_skipped_svr")
        assert ins is not None and ins.metadata["policy"] == "confirm"
        # A deferral, like a refusal, has to name the way back in and the
        # alternative that works — not just restate the price.
        warnings = " | ".join(w.value for w in at.warning if w.value)
        assert "SHAP was not computed for SVR" in warnings, warnings
        assert "Run KernelExplainer anyway" in warnings, warnings
        assert "permutation importance" in warnings, warnings

    def test_a_refusal_is_not_a_crash_and_does_not_stop_the_page(self):
        at = _page(_frame(n=120, p=1200))
        _add_kernel_model(at, None)
        at.run()
        _click(at, "Run Selected Analyses")
        assert not at.exception
        # SHAP still succeeded for one of the two models, so the banner must
        # not claim a clean sweep and must not claim total failure.
        assert not at.error


# ── TreeExplainer: uncapped on purpose ──────────────────────────────────────

def test_tree_explainer_carries_no_feature_cap_and_says_why():
    with open(PAGE_07, encoding="utf-8") as fh:
        src = fh.read()
    head, _, tail = src.partition("if shap_support == 'tree':")
    assert tail, "the TreeExplainer branch moved"
    tree_branch = tail.split("elif shap_support == 'linear':")[0]
    # No threshold of any kind between the branch and its shap_values call.
    for forbidden in ("kernel_shap", "MAX_FEATURES", "n_features >"):
        assert forbidden not in tree_branch, tree_branch
    # And the reason the asymmetry with the kernel branch is deliberate is
    # stated where a future reader will find it.
    assert "deliberately NOT capped on the feature" in head
    assert "fitted exponent of 0.00" in head


# ── the Methods sentence ────────────────────────────────────────────────────

def test_the_methods_sentence_names_only_the_models_each_analysis_reached(monkeypatch):
    """A model whose SHAP was refused but whose permutation importance ran must
    not be named in the SHAP sentence."""
    import ml.publication as publication

    monkeypatch.setattr(publication, "generate_methods_from_log", lambda: {
        "Explainability": [{
            "step": "Explainability",
            "action": "Ran permutation_importance, shap on 2 models",
            "details": {
                "analyses": ["permutation_importance", "shap"],
                "models": ["ridge", "svr"],
                "models_by_analysis": {
                    "permutation_importance": ["ridge", "svr"],
                    "shap": ["ridge"],
                },
                "shap_n_eval_rows": [200],
            },
        }]
    })

    text = publication.generate_methods_section(
        data_config={"feature_cols": [f"p{i}" for i in range(8)]},
        preprocessing_config={},
        model_configs={"ridge": {}, "svr": {}},
        split_config={},
        n_total=1000, n_train=700, n_val=150, n_test=150,
        feature_names=[f"p{i}" for i in range(8)],
        target_name="y", task_type="regression",
        metrics_used=["RMSE"],
        explainability_methods=["permutation_importance", "shap"],
    )

    shap_sentence = [s for s in text.split(". ") if "SHapley" in s]
    assert shap_sentence, text
    assert "svr" not in shap_sentence[0], shap_sentence[0]
    assert "ridge" in shap_sentence[0], shap_sentence[0]
    # Permutation importance reached both, and still says so.
    perm_sentence = [s for s in text.split(". ") if "Permutation importance" in s]
    assert perm_sentence and "svr" in perm_sentence[0], perm_sentence
    # SHAP quotes the rows it explained, not the whole held-out set.
    assert "using 200 test observations" in text, text
    assert "using 150 test observations" not in text.split("SHapley")[-1]


def test_a_width_measured_by_a_real_run_beats_the_pre_run_estimate():
    """`max(estimate, observed)` self-heals in one direction only.

    `_post_transform_width` falls back to the RAW column count whenever
    `feature_names_by_model` has no entry for a model, which OVER-states p
    whenever preprocessing reduces. A 1,200-column upload whose pipeline emits
    800 features then quoted `refuse`, rendered no confirmation checkbox, met
    the real 800 inside the run loop, read `confirm`, found nothing ticked and
    recorded a skip. Taking the max threw the measured 800 away on the next
    render, so the control that would have released the job never appeared and
    the ledger filed "was not affordable" against it in perpetuity.

    A width a real run MEASURED is a fact about this dataset; the estimate is
    only what to say before the pipeline has ever been applied.
    """
    at = _page(_frame(n=120, p=1200))
    _add_kernel_model(at, None)
    at.run()

    # No observed width yet: the raw count is all there is, and it refuses.
    assert not at.exception
    assert not [c for c in at.checkbox if c.key == "run_kernel_shap_confirm"], (
        "a refusal must not offer a confirmation")

    # Now a previous run has measured the real post-preprocessing width.
    at.session_state["kernel_shap_observed_widths"] = {"svr": 800}
    at.run()
    assert not at.exception

    text = _all_text(at)
    assert "KernelExplainer on 800 features" in text, text
    # Only the KERNEL quote is corrected. `kernel_shap_observed_widths` records
    # the width the kernel explainer met, and permutation importance is a
    # different question over every trained model, so its caption still quotes
    # the full width — checked here so a future change cannot quietly widen the
    # correction past what was actually measured.
    kernel_quotes = [i.value for i in at.info if "KernelExplainer on" in (i.value or "")]
    assert kernel_quotes and not any("1,200" in q for q in kernel_quotes), (
        f"the over-stating estimate outlived the measurement: {kernel_quotes}")
    confirm = [c for c in at.checkbox if c.key == "run_kernel_shap_confirm"]
    assert confirm, "the measured width should have restored the way to say yes"
    assert confirm[0].value is False

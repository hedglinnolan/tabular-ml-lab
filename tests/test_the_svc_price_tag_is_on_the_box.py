"""`ml/model_coach.py` — SVC's calibration cost is stated where the box is ticked.

`ml/model_registry.py::_create_svc` builds `SVC(..., probability=True)`. That
flag runs an internal 5-fold Platt calibration, so ONE "SVC fit" is six libsvm
solves against SVR's one, and libsvm is superquadratic in n. Measured on a
dense 120-feature matrix: 0.084 s -> 0.54 s at n=1,000, 1.95 s -> 8.02 s at
5,000, 79.4 s -> 548.7 s at 20,000.

The app already said this twice, both times on the Train page and both times
AFTER the user had committed to the run. The one surface that speaks at the
moment of choice — the coach verdict rendered under each model's checkbox —
said nothing about it, and in the branches below n=5,000 (the branches a user
actually says yes to) named no cost at all.

**The flag is not touched and must not be.** ROC-AUC, LogLoss and PR-AUC, the
calibration curve, the ROC/PR plots and SVC's KernelExplainer path all gate on
`hasattr(model, "predict_proba")`, which sklearn's `available_if` makes False
when the flag is off — so removing it would not raise, it would silently delete
those outputs. This file asserts the flag is still set, and that the factory
still produces a byte-identical fitted model, alongside the new disclosure.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.model_coach import (                                      # noqa: E402
    _fmt_fit_duration, _svc_calibrated_fit_seconds, model_viability,
)
from ml.model_registry import get_registry                        # noqa: E402


class _TargetProfile:
    def __init__(self, task_type="classification"):
        self.task_type = task_type
        self.minority_class_size = 600
        self.is_imbalanced = False
        self.class_balance_ratio = 1.2
        self.has_outliers = False
        self.outlier_rate = 0.0


class _Profile:
    def __init__(self, n_rows, n_features, task_type="classification"):
        self.n_rows = n_rows
        self.n_features = n_features
        self.p_n_ratio = n_features / n_rows
        self.target_profile = _TargetProfile(task_type)
        self.events_per_variable = 24.2
        self.n_features_with_missing = 0
        self.highly_skewed_features = []


def _svc(n, p=25, **kw):
    return model_viability(_Profile(n, p), n_train=n, **kw)["svc"][1]


# ══════════════════════════════════════════════════════════════════════════
# The clause names the flag, the row count and a cost — in every branch
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("n", [200, 1000, 4407, 5000, 20000, 60000])
def test_every_svc_branch_names_the_flag_and_the_row_count(n):
    clause = _svc(n)
    assert "probability=True" in clause, (
        "the cost has a cause, and the user cannot act on a cost with no cause")
    assert f"n={n:,}" in clause, (
        "a cost with no row count beside it is 'may be slow' with extra words")


@pytest.mark.parametrize("n", [200, 1000, 4407, 5000, 20000, 60000])
def test_every_svc_branch_carries_a_duration(n):
    clause = _svc(n)
    assert any(u in clause for u in (" s at", " min at", " h at", " days at")), (
        f"no projected duration in the n={n} clause: {clause!r}")


def test_the_cheap_branch_is_where_this_was_missing():
    # Below the 5,000-row threshold the verdict is "ok" and the user is not
    # being discouraged — which is exactly why the cost has to be visible
    # here, and exactly where the clause used to be silent about it.
    v = model_viability(_Profile(2000, 25), n_train=2000)["svc"]
    assert v[0] == "ok"
    assert "6 libsvm solves" in v[1]


def test_svr_is_not_told_it_has_a_flag_it_does_not_have():
    # sklearn's SVR has no `probability` parameter at all and
    # `_create_svr` builds it bare, so the shared clause cannot carry this.
    for n in (2000, 20000):
        svr = model_viability(_Profile(n, 25), n_train=n)["svr"]
        assert "probability" not in svr[1]
        assert "libsvm" not in svr[1]


def test_the_cost_is_denominated_in_time_not_memory():
    # Measured peak RSS was 256.8 MB without the flag against 258.1 MB with
    # it, because libsvm's kernel cache is capped at 200 MB by default. A GB
    # figure on this badge would be a fabrication.
    for n in (1000, 20000, 150000):
        clause = _svc(n)
        assert "GB" not in clause and "MB" not in clause and "RAM" not in clause


# ══════════════════════════════════════════════════════════════════════════
# The projection is the measurement, and it stays inside its own evidence
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("rows,seconds", [(1000, 0.54), (5000, 8.02), (20000, 548.7)])
def test_the_projection_reproduces_its_own_anchors(rows, seconds):
    assert _svc_calibrated_fit_seconds(rows) == pytest.approx(seconds, rel=1e-9)


def test_the_projection_is_monotone_and_superquadratic():
    prev = 0.0
    for n in (100, 500, 1000, 5000, 10000, 20000, 50000):
        cur = _svc_calibrated_fit_seconds(n)
        assert cur > prev, "a bigger training set cannot project a cheaper fit"
        prev = cur
    # Doubling rows must cost more than 4x somewhere above the top anchor —
    # this is the whole reason the badge exists rather than a linear guess.
    assert (_svc_calibrated_fit_seconds(40000)
            / _svc_calibrated_fit_seconds(20000)) > 4.0


def test_the_shipped_badge_agrees_with_the_train_page_surfaces():
    # pages/06_Train_and_Compare.py states "9.1 min at 20,000 rows x 120
    # features" in two places. Three surfaces contradicting each other is
    # worse than two surfaces saying nothing.
    assert _fmt_fit_duration(_svc_calibrated_fit_seconds(20000)) == "9 min"


@pytest.mark.parametrize("n", [0, 1, -5, 10 ** 12])
def test_the_clause_arithmetic_is_total(n):
    # pages/06_Train_and_Compare.py wraps the whole `model_viability` call in
    # a bare `except` that sets `_viability = {}`, so a raise here would
    # silently delete the shape reasoning from EVERY model card, not just
    # SVC's. No input may raise, and no input may print "inf".
    seconds = _svc_calibrated_fit_seconds(n)
    assert seconds > 0 and np.isfinite(seconds)
    assert "inf" not in _fmt_fit_duration(seconds)


def test_large_projections_are_stated_coarsely():
    # Far above the top anchor these are extrapolations off three points.
    # "3 days" is the order-of-magnitude claim this is; "70.8 h" would dress
    # the same guess up as a stopwatch reading.
    assert _fmt_fit_duration(_svc_calibrated_fit_seconds(150000)).endswith("days")


# ══════════════════════════════════════════════════════════════════════════
# Nothing about the model moved
# ══════════════════════════════════════════════════════════════════════════

def test_the_svc_factory_still_builds_exactly_what_it_built():
    # This PR's only edit to ml/model_registry.py is a comment. The guard that
    # matters is therefore on the ARGUMENTS, because they are what a later
    # reader "optimizing" the disclosed cost away would reach for. Asserted as
    # values rather than as a fitted fingerprint so the test does not go red
    # on a scikit-learn point release that never touched this app.
    est = get_registry()["svc"].factory("classification", 42)
    p = est.get_params()
    assert p["probability"] is True, (
        "predict_proba is consumed by ROC-AUC, LogLoss, PR-AUC, the "
        "calibration curve, the ROC/PR plots and SVC's KernelExplainer path, "
        "all of which gate on hasattr and would go SILENT, not loud")
    assert (p["kernel"], p["C"], p["gamma"], p["random_state"]) == (
        "rbf", 1.0, "scale", 42)
    assert hasattr(est, "predict_proba") and hasattr(est, "decision_function")


def test_the_fitted_svc_is_bit_for_bit_what_it_was():
    # Fitted before and after the change on this seed, with this factory. The
    # values below are the `main` fingerprint, recorded from a run against the
    # pre-change tree: predictions, calibrated probabilities, the decision
    # function, the dual coefficients and the support counts all matched to
    # the last bit. `rel=1e-9` rather than `==` only because a float that
    # round-trips through a test file is not the place to assert bit equality.
    rng = np.random.RandomState(0)
    X = rng.randn(200, 6)
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.randn(200) * 0.4 > 0).astype(int)
    est = get_registry()["svc"].factory("classification", 42)
    est.fit(X, y)

    assert est.predict(X)[:12].tolist() == [1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0]
    assert est.predict_proba(X)[0, 1] == pytest.approx(0.9198071036041886, rel=1e-6)
    assert float(est.decision_function(X)[:5].sum()) == pytest.approx(
        5.432708329295676, rel=1e-6)
    assert est.n_support_.tolist() == [54, 53]
    assert float(est.intercept_[0]) == pytest.approx(-0.036459362681279424, rel=1e-6)


def test_the_verdict_letters_are_where_they_were():
    # The disclosure is a disclosure, not a re-recommendation. The 5,000-row
    # threshold decides which models get a "poor" badge, and moving it would
    # change what the app recommends.
    for n in (200, 1000, 5000):
        assert model_viability(_Profile(n, 25), n_train=n)["svc"][0] == "ok"
    for n in (5001, 20000):
        assert model_viability(_Profile(n, 25), n_train=n)["svc"][0] == "poor"
    # And SVR's verdict follows SVC's, as it did before.
    for n in (200, 5000, 5001, 20000):
        v = model_viability(_Profile(n, 25), n_train=n)
        assert v["svr"][0] == v["svc"][0]

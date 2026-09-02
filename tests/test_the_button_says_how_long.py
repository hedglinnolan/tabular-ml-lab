"""`ml/cost_model.py` — the price of a click, measured on the researcher's own frame.

**The defect (F-10 in the omics audit).** Nothing in the app estimated a
duration from the data. The only durations shown were literals — "30-60
minutes" on the landing page, "~5 minutes" on Explainability, "30-90 seconds"
inside the training loop — identical for 40 columns and 40,000. A confidently
wrong number is worse than none, because a researcher plans around it.

**Why a probe and not a formula.** The audit's rule was "the exponent is
portable, the constant is not". Measured on this repository's own registry
(the PR that added this module carries the table), even the exponent is only
local: a forest is ~linear in rows, an SVM ~quadratic and steepening, and
gradient boosting is overhead-dominated below 8,000 rows before it turns
linear. So the module ships no exponents. It times the estimator the page is
about to fit, on nested subsamples of the actual training rows, doubling
until a small budget is spent, and reads the slope off the last points. What
is asserted here is the machinery: the warm-up is not timed, the budget
bounds the probe, the slope is never used sublinearly past the probed range,
a failed sample fit is `None` and never a number, and the arithmetic that
turns one model's probe into a run's price is additive in the way the page's
own fit-count caption is.

**The property the pages depend on.** No string literal in `pages/` or
`app.py` states a fixed number of minutes, seconds or hours about the user's
run. Cited measurements ("one measured 9.1 min at 20,000 rows") say so in the
same string; everything else is derived from the data or says nothing.

The clock is faked with a monkeypatch so the probe's decisions are
deterministic: `run_at_size` advances the fake clock by the cost it models.
"""
from __future__ import annotations

import ast
import os
import re
import sys

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from ml import cost_model                                        # noqa: E402
from ml.cost_model import (                                      # noqa: E402
    CV_POOL_STARTUP_SECONDS, EXTRAPOLATION_SLOPE_FLOOR, PROBE_BUDGET_SECONDS,
    PROBE_START_ROWS, SLOPE_ASSUMED, ModelCost, ScalingProbe, cv_wall_seconds,
    cv_worker_bytes, humanize_bytes, humanize_seconds, not_estimated,
    probe_fit_seconds, probe_scaling, provenance_clause, training_run_cost,
    widest_extrapolation,
)


class _Clock:
    """A clock that only moves when the work under test says it did."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(cost_model.time, "perf_counter", c)
    return c


def _work(clock, cost_of_size, fail_at=None):
    calls = []

    def run(size):
        calls.append(size)
        if fail_at is not None and size >= fail_at:
            raise RuntimeError("the sample fit raised")
        clock.now += cost_of_size(size)
    run.calls = calls
    return run


# ── the probe's decisions ────────────────────────────────────────────────────

def test_the_warm_up_is_run_but_not_timed(clock):
    """A library's first fit pays one-time costs — LightGBM's first call
    measured 1.36 s against 0.02 s for the second. Timing it would project the
    start-up 60x. So the first call at the start size is made and discarded."""
    run = _work(clock, lambda k: 1e-3 * k)
    probe = probe_scaling(run, size_target=8_000)
    assert run.calls[0] == run.calls[1] == PROBE_START_ROWS, "warm-up, then the first timed point"
    assert probe.points[0] == (PROBE_START_ROWS, pytest.approx(0.128))
    assert len(run.calls) == len(probe.points) + 1


def test_a_linear_fit_is_projected_linearly_and_the_budget_bounds_the_probe(clock):
    run = _work(clock, lambda k: 1e-3 * k)
    probe = probe_scaling(run, size_target=8_000)
    sizes = [s for s, _ in probe.points]
    assert sizes == [128, 256, 512, 1024], "the 2,048-row point would have overspent the budget"
    assert sum(t for _, t in probe.points) <= PROBE_BUDGET_SECONDS
    assert probe.slope == pytest.approx(1.0)
    assert not probe.slope_assumed
    assert probe.seconds_at_target == pytest.approx(8.0)
    assert probe.extrapolation_factor == pytest.approx(8_000 / 1024)


def test_a_quadratic_fit_is_projected_quadratically(clock):
    run = _work(clock, lambda k: 1e-6 * k * k)
    probe = probe_scaling(run, size_target=8_000)
    assert probe.slope == pytest.approx(2.0)
    assert probe.seconds_at_target == pytest.approx(64.0, rel=1e-6)


def test_a_sublinear_local_slope_is_never_used_past_the_probed_range(clock):
    """Overhead still amortizing reads as a slope below 1. Measured: a forest's
    slope between 256 and 512 rows was 0.58, and projecting 16x along it gave
    1.0 s for a fit that took 6.4 s. No fit is sublinear in rows."""
    run = _work(clock, lambda k: 0.1 + 1e-5 * k)
    probe = probe_scaling(run, size_target=100_000)
    assert probe.slope < EXTRAPOLATION_SLOPE_FLOOR, "the measured slope is kept as measured"
    assert probe.extrapolation_slope == EXTRAPOLATION_SLOPE_FLOOR
    linear_from_last = probe.seconds_probed * (100_000 / probe.size_probed)
    assert probe.seconds_at_target == pytest.approx(linear_from_last)
    # ... and inside the probed range the measured slope stands.
    inside = probe.points[0][0] * 3 // 2
    assert probe.seconds_at(inside) < probe.points[0][1] * 1.5


def test_a_probe_that_reaches_the_target_is_a_measurement_not_a_projection(clock):
    run = _work(clock, lambda k: 1e-3 * k)
    probe = probe_scaling(run, size_target=300)
    assert [s for s, _ in probe.points] == [128, 256, 300]
    assert probe.measured_at_target
    assert probe.extrapolation_factor == 1.0
    assert probe.seconds_at_target == pytest.approx(0.3)


def test_a_sample_fit_that_raises_at_once_is_none_not_a_number(clock):
    run = _work(clock, lambda k: 1e-3 * k, fail_at=1)
    assert probe_scaling(run, size_target=8_000) is None


def test_a_sample_fit_that_raises_later_keeps_what_was_measured(clock):
    run = _work(clock, lambda k: 1e-3 * k, fail_at=512)
    probe = probe_scaling(run, size_target=8_000)
    assert [s for s, _ in probe.points] == [128, 256]
    assert not probe.slope_assumed


def test_one_point_assumes_linear_and_says_so(clock):
    """A fit that spends most of the budget at the start size leaves one
    point. The slope is then assumed, flagged, and linear — not silently 0."""
    run = _work(clock, lambda k: 1.5 if k <= 128 else 1e9)
    probe = probe_scaling(run, size_target=8_000)
    assert len(probe.points) == 1
    assert probe.slope_assumed and probe.slope == SLOPE_ASSUMED
    assert probe.seconds_at_target == pytest.approx(1.5 * 8_000 / 128)


def test_the_slope_is_read_over_the_last_three_points_so_one_wobble_does_not_set_it(clock):
    """A forest's timing wobbles with its thread pool — measured 0.58 between
    256 and 512 rows and 1.47 between 512 and 1,024 on one estimator. A line
    through three points reads the trend under the wobble."""
    costs = {128: 0.05, 256: 0.10, 512: 0.20, 1024: 0.60, 2048: 0.80}
    run = _work(clock, lambda k: costs[k])
    probe = probe_scaling(run, size_target=8_000, budget_seconds=10.0)
    two_point = np.log(0.80 / 0.60) / np.log(2)              # 0.42 — the wobble
    three_point = np.polyfit(np.log([512, 1024, 2048]), np.log([0.20, 0.60, 0.80]), 1)[0]
    assert probe.slope == pytest.approx(three_point)
    assert probe.slope != pytest.approx(two_point)


def test_the_row_probe_uses_nested_subsamples_of_the_real_rows(clock):
    seen = []

    def fit(X_sub, y_sub):
        seen.append((len(X_sub), tuple(X_sub[:3, 0])))
        clock.now += 1e-3 * len(X_sub)

    X = np.arange(4_000 * 2, dtype=float).reshape(4_000, 2)
    y = np.arange(4_000)
    probe = probe_fit_seconds(fit, X, y)
    assert probe is not None
    firsts = {first for _, first in seen}
    assert len(firsts) == 1, "every sample is a prefix of one permutation"
    assert seen[0][1][0] != 0.0, "and the permutation is not the identity"


def test_an_empty_frame_has_no_probe(clock):
    assert probe_fit_seconds(lambda X, y: None, np.zeros((0, 3)), np.zeros(0)) is None


# ── the arithmetic that turns probes into a run ──────────────────────────────

def _probe(points, target, slope=1.0):
    last_size, last_seconds = points[-1]
    at = last_seconds if last_size >= target else last_seconds * (target / last_size) ** max(slope, 1.0)
    return ScalingProbe(points=tuple(points), slope=slope, slope_assumed=False,
                        size_target=target, seconds_at_target=at)


def test_cross_validation_costs_k_fold_fits_plus_the_pool_start_up():
    """Measured through the app's own CV path at 8,000 x 60: five parallel
    folds cost 4.96 fold fits for RandomForest, 3.94 for XGBoost, 3.08 for
    SVR — the workers are thread-pinned and the forests oversubscribe, so
    what parallelism gives back the pinning takes. Charging k fits is exact
    for the forest and never 5x too low."""
    probe = _probe([(1000, 1.0), (2000, 2.0)], target=10_000)
    per_fold = probe.seconds_at(8_000)
    assert cv_wall_seconds(probe, 10_000, 5) == pytest.approx(CV_POOL_STARTUP_SECONDS + 5 * per_fold)


def test_fold_workers_each_hold_a_copy_and_the_arithmetic_is_exact():
    assert cv_worker_bytes(40 * 2**20, folds=5, cores=8) == 40 * 2**20 * 6
    assert cv_worker_bytes(40 * 2**20, folds=5, cores=2) == 40 * 2**20 * 3, "two workers, two copies"
    assert cv_worker_bytes(40 * 2**20, folds=5, cores=None) == 40 * 2**20 * 6, "unknown cores: one per fold"


def test_a_runs_cost_is_the_three_additive_terms_the_caption_already_counts():
    """pages/06 states fits as `final + folds` and `trials + final + folds`,
    additive and not nested. The seconds follow the same shape."""
    probes = {"rf": _probe([(1000, 1.0), (2000, 2.0)], target=2_000),
              "ridge": _probe([(2000, 0.01)], target=2_000),
              "nn": None}
    costs = training_run_cost(probes, n_train=2_000, cv_folds=5, cv_models=["rf", "ridge"],
                              optuna_trials={"rf": 30})
    by_key = {c.key: c for c in costs}
    assert by_key["rf"].final_fit_seconds == pytest.approx(2.0)
    assert by_key["rf"].cv_seconds == pytest.approx(cv_wall_seconds(probes["rf"], 2_000, 5))
    assert by_key["rf"].optuna_seconds == pytest.approx(60.0)
    assert by_key["ridge"].optuna_seconds == 0.0, "no trials, no Optuna term"
    assert not by_key["nn"].estimated and by_key["nn"].seconds == 0.0
    assert not_estimated(costs) == ("nn",)
    assert widest_extrapolation(costs) == 1.0


def test_the_provenance_clause_says_measured_only_when_nothing_was_extrapolated():
    measured = ModelCost("a", _probe([(500, 0.5)], target=500), 0.5, 0.0, 0.0)
    projected = ModelCost("b", _probe([(512, 0.5)], target=8_192), 8.0, 0.0, 0.0)
    assert provenance_clause([measured]) == "measured just now on your full training set"
    clause = provenance_clause([measured, projected])
    assert "projected from sample fits of 500+" in clause and "16x" in clause
    assert provenance_clause([ModelCost("c", None, 0, 0, 0)]) == ""


@pytest.mark.parametrize("seconds, words", [
    (0.4, "under a second"),
    (1.2, "about a second"),
    (7.4, "about 7 seconds"),
    (47, "about 45 seconds"),
    (200, "about 3 minutes"),
    (61, "about 60 seconds"),
    (100, "about 2 minutes"),
    (5000, "about 83 minutes"),
    (20000, "about 5.6 hours"),
])
def test_seconds_are_said_coarsely_because_the_method_is_coarse(seconds, words):
    assert humanize_seconds(seconds) == words


def test_the_memory_line_is_exact_and_only_shown_when_it_is_worth_a_line():
    """A 262 KB note under a 600-row study is friction; the audit's case is
    160 MB per fold worker. The arithmetic is the same either way."""
    from utils.fit_cost import MEMORY_LINE_MIN_BYTES, TrainingPrice

    small = TrainingPrice(costs=(), not_probed=(), n_train=600, cv_folds=5,
                          train_matrix_bytes=600 * 8 * 8)
    assert small.memory_sentence(cores=8) is None
    assert small.as_state()["train_matrix_bytes"] == 600 * 8 * 8
    big = TrainingPrice(costs=(), not_probed=(), n_train=200, cv_folds=5,
                        train_matrix_bytes=200 * 100_000 * 8)
    line = big.memory_sentence(cores=8)
    assert line is not None and "153 MB training matrix" in line and "916 MB in all" in line
    assert cv_worker_bytes(200 * 100_000 * 8, 5, 8) >= MEMORY_LINE_MIN_BYTES


def test_bytes_are_said_in_the_unit_that_fits():
    assert humanize_bytes(512) == "512 bytes"
    assert humanize_bytes(1536) == "2 KB"
    assert humanize_bytes(40 * 2**20) == "40 MB"
    assert humanize_bytes(3 * 2**30) == "3.0 GB"


# ── against the benchmark table this module was calibrated on ────────────────

def test_the_forest_measured_on_this_box_is_projected_within_forty_percent():
    """RandomForest at p=60, 100 trees, measured Sep 2 2026 on the target
    laptop: 0.476 s at 1,000 rows, 1.255 s at 2,000, 2.724 s at 4,000, 5.533 s
    at 8,000. A probe that stopped at 4,000 must land near 5.5 s."""
    points = [(1000, 0.476), (2000, 1.255), (4000, 2.724)]
    slope = cost_model._slope(points)
    probe = ScalingProbe(points=tuple(points), slope=slope, slope_assumed=False,
                         size_target=8_000,
                         seconds_at_target=2.724 * 2 ** max(slope, EXTRAPOLATION_SLOPE_FLOOR))
    assert 0.6 * 5.533 <= probe.seconds_at_target <= 1.4 * 5.533


def test_the_support_vector_machine_measured_on_this_box_is_projected_within_forty_percent():
    """SVC with probability=True at p=60: 0.204 s at 1,000, 0.768 s at 2,000,
    2.930 s at 4,000, 10.516 s at 8,000 — the quadratic family."""
    points = [(1000, 0.204), (2000, 0.768), (4000, 2.930)]
    slope = cost_model._slope(points)
    assert 1.8 <= slope <= 2.1
    projected = 2.930 * 2 ** slope
    assert 0.6 * 10.516 <= projected <= 1.4 * 10.516


# ── the pages: no literal durations, and the price sits above the button ─────

_DURATION = re.compile(
    r"\b\d[\d.,]*\s*(?:-|–|to)?\s*\d*\s*(?:minutes?|mins?\b|seconds?|secs?\b|hours?)\b",
    re.IGNORECASE,
)
_SCANNED = ["app.py"] + sorted(
    os.path.join("pages", f) for f in os.listdir(os.path.join(PROJECT_ROOT, "pages"))
    if f.endswith(".py")
)


def _literal_durations(path):
    """Every string literal in `path` that states a duration in digits and does
    not, in the same literal, call it a measurement."""
    src = open(os.path.join(PROJECT_ROOT, path), encoding="utf-8").read()
    tree = ast.parse(src)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value
        else:
            continue
        if _DURATION.search(text) and "measured" not in text.lower():
            hits.append(f"{path}:{node.lineno}: {text.strip()[:90]!r}")
    return hits


def test_no_page_states_a_fixed_number_of_minutes_about_the_users_run():
    """The pin the handoff asked for. A literal "30-60 minutes" is the same
    sentence for 40 columns and 40,000; a cited measurement says "measured"
    in the same string and is allowed, because it is about a number that was
    taken, not one that is promised."""
    offences = [h for path in _SCANNED for h in _literal_durations(path)]
    assert not offences, "\n".join(offences)


def test_the_pin_would_catch_the_literals_this_pr_removed():
    """The control: the sentences that used to be in the pages are offences.
    Parsed from a string rather than a written file, so the repo write guard
    has nothing to count."""
    src = (
        'a = "This model may take 30-90 seconds to train..."\n'
        'b = "**Time:** ~5 minutes"\n'
        'c = "Expect 30 seconds to several minutes depending on dataset size."\n'
        'd = "one measured 9.1 min at 20,000 rows"\n'
        'e = f"about {n} minutes"\n'
    )
    hits = [n.value for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and _DURATION.search(n.value) and "measured" not in n.value.lower()]
    assert len(hits) == 3, hits


def test_the_price_sits_above_the_train_buttons_and_the_loop_quotes_it_back():
    src = open(os.path.join(PROJECT_ROOT, "pages", "06_Train_and_Compare.py"), encoding="utf-8").read()
    priced = src.index("_price_training_run(")
    button = src.index('st.button("Train Models", type="primary", key="train_models_button"')
    assert priced < button, "the price is stated before the click"
    assert src.index("_probe_estimator_for(") < button
    loop = src.index("def _train_models(")
    literals = [n.value for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert not any("30-90 seconds" in lit for lit in literals), "the literal is gone from what the page says"
    assert src.index("st.session_state.get('_train_cost')", loop) > loop, \
        "the loop reads the quote back rather than restating a literal"


def test_the_sensitivity_page_prices_both_loops_above_their_buttons():
    src = open(os.path.join(PROJECT_ROOT, "pages", "08_Sensitivity_Analysis.py"), encoding="utf-8").read()
    seed_button = src.index('key="run_seed"')
    drop_button = src.index('key="run_dropout"')
    first_price = src.index("_price_refits(")
    second_price = src.index("_price_refits(", seed_button)
    assert first_price < seed_button < second_price < drop_button


def test_the_cost_model_and_its_page_helper_import_without_streamlit_side_effects():
    """`ml/cost_model.py` must stay importable in the lean environment the
    commit gates use: no streamlit, no sklearn at import time."""
    src = open(os.path.join(PROJECT_ROOT, "ml", "cost_model.py"), encoding="utf-8").read()
    imported = {n.names[0].name.split(".")[0] for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.Import)}
    imported |= {(n.module or "").split(".")[0] for n in ast.walk(ast.parse(src))
                 if isinstance(n, ast.ImportFrom)}
    assert "streamlit" not in imported and "sklearn" not in imported

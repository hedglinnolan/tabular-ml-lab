"""A price for the click, measured on the researcher's own frame.

**What this replaces.** The audit's F-10: nothing in the app estimated a
duration from the shape of the data. The only durations shown were literals —
"30-60 minutes" on the landing page, "~5 minutes" on Explainability, "30-90
seconds" on Train & Compare — the same words whether the user brought 40
columns or 40,000. A confidently wrong number is worse than none, because the
researcher plans around it.

**Two rules, from the audit's section 07.**

1. *Memory estimates can be exact.* A fold worker holds a copy of the training
   matrix; five of them hold five. That is arithmetic on `nbytes` and needs
   no calibration — `cv_worker_bytes` below.

2. *Time estimates cannot be exact, and the constant is not portable.* The
   same fit takes 7 s on one box and 10 s on another. What IS portable is
   the shape of the curve, and even that only locally: measured on this
   repository's own registry at p=60, a forest's fit time grows about
   linearly in rows (exponent 1.05–1.07), a support-vector machine about
   quadratically (1.9–2.1, and steeper once the kernel cache is exceeded —
   the audit's 5,000→20,000 measurement gives 2.7–3.1), and gradient boosting
   is overhead-dominated below 8,000 rows (0.25–0.42) before it turns linear.
   A fixed per-family exponent would therefore be wrong somewhere for every
   family. So this module does not ship exponents. It ships a **probe**: fit
   the estimator the page is about to fit, on nested subsamples of the actual
   training rows, doubling until a small time budget is spent, and read the
   local slope off the last two points. The constant comes from the
   researcher's machine, the dtype and width from their frame, the slope from
   the estimator's own behavior at the largest size that could be afforded.
   The extrapolation factor is reported alongside, because an estimate
   extrapolated 32x from a 250-row sample is a different claim from one
   measured at three quarters of the data.

**What it is not.** A promise. `humanize_seconds` says "about", the page says
what the number was measured on, and a floor is called a floor: Optuna
explores larger settings than the defaults the probe fitted, and an SVM's
slope keeps rising past the probe's largest sample. Cancellation is not here
either — a cancel that stops the work needs something that owns the work,
which is the job discipline in `turbotab/jobs.py`, not a flag checked between
fits.

Pure numpy; no Streamlit, so it is testable without a script run and
importable in the lean environment the commit gates use.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

#: Seconds a probe may spend per estimator, warm-up excluded. Spent once per
#: configuration — the page caches the result by data and model settings —
#: so this is the cost of changing a pick, not of every rerun, and it is only
#: spent in full on a frame large enough that the run itself is long: a
#: probe that reaches the whole training set stops there. Two seconds is
#: what a 100-tree RandomForest needs to reach 2,000 rows x 60 columns, the
#: size at which its curve has stopped being per-tree overhead; at 0.75 s it
#: stopped at 512 rows and read a slope of 0.58 off the overhead, and the
#: projection came out 6x low.
PROBE_BUDGET_SECONDS = 2.0

#: Rows the first probe fits. Small enough that anything fits it in
#: milliseconds, large enough that a classifier sees every class of an
#: ordinary outcome.
PROBE_START_ROWS = 128

#: The local slope is clamped to this band before extrapolating. Below the
#: floor is timer noise on a fit that took microseconds; above the ceiling is
#: not a fit-time curve any estimator in the registry has.
SLOPE_FLOOR = 0.25
SLOPE_CEILING = 3.0

#: The slope assumed when the probe could only afford one point — a fit that
#: took most of the budget at the first size. Linear is the middle of the
#: measured band and is flagged in the result, never silently used.
SLOPE_ASSUMED = 1.0

#: No fit is sublinear in rows — every estimator reads every row at least
#: once — so a measured slope below this is overhead still amortizing, not a
#: curve to extrapolate along. Beyond the probed range the slope used is at
#: least this; inside it the measured slope stands. Measured: a forest's
#: local slope between 256 and 512 rows was 0.58, and projecting 16x along
#: it gave 1.0 s for a fit that took 6.4 s.
EXTRAPOLATION_SLOPE_FLOOR = 1.0

#: Below this, a timing is at the resolution of the clock and says nothing
#: about scaling; the point is kept for the record but not for the slope.
_TIMER_FLOOR_SECONDS = 1e-4


@dataclass(frozen=True)
class ScalingProbe:
    """What was measured and what was extrapolated, kept apart.

    `points` are (size, seconds) in increasing size, nested — each sample is a
    prefix of one fixed permutation, so the curve is a curve and not noise
    between different draws. `slope` is the log-log slope of the last two
    points, clamped; `slope_assumed` says it was not measured at all.
    """

    points: Tuple[Tuple[int, float], ...]
    slope: float
    slope_assumed: bool
    size_target: int
    seconds_at_target: float

    @property
    def size_probed(self) -> int:
        return self.points[-1][0]

    @property
    def seconds_probed(self) -> float:
        return self.points[-1][1]

    @property
    def extrapolation_factor(self) -> float:
        return self.size_target / max(self.size_probed, 1)

    @property
    def measured_at_target(self) -> bool:
        """The probe reached the target size itself; nothing was extrapolated."""
        return self.size_probed >= self.size_target

    @property
    def extrapolation_slope(self) -> float:
        """The slope used past the probed range: measured, but never sublinear."""
        return max(self.slope, EXTRAPOLATION_SLOPE_FLOOR)

    def seconds_at(self, size: int) -> float:
        """The curve, read at another size — a fold's 80% of rows, say."""
        size = max(int(size), 1)
        if size <= self.size_probed:
            # Inside the measured range: read from the nearest measured point
            # below along the measured slope. Below the first point — a fold
            # of a frame the probe finished in one step — read down from it.
            below = [(s, t) for s, t in self.points if s <= size]
            s0, t0 = below[-1] if below else self.points[0]
            if s0 == size:
                return t0
            return t0 * (size / s0) ** self.slope
        return self.seconds_probed * (size / self.size_probed) ** self.extrapolation_slope


def probe_scaling(run_at_size: Callable[[int], None], size_target: int,
                  budget_seconds: float = PROBE_BUDGET_SECONDS,
                  start_size: int = PROBE_START_ROWS) -> Optional[ScalingProbe]:
    """Time `run_at_size(k)` at doubling `k` until the budget is spent.

    `run_at_size` does the work at size `k` — fits on the first `k` rows of a
    fixed permutation, correlates the first `k` columns — and returns nothing.
    A size that raises ends the probe with what was measured so far; if the
    very first size raises, there is no measurement and the answer is `None`,
    which callers must show as "not estimated" rather than as a number.

    The next size is attempted only when the time it is predicted to take,
    at the slope measured so far, still fits in the budget. So the budget
    bounds the total, and the last point is the largest size that could be
    afforded rather than one past it.

    The first call at the start size is a WARM-UP and is not timed. A
    library's first fit in a process pays one-time costs — LightGBM's first
    call measured 1.36 s against 0.02 s for the second, a forest's first call
    spins up its thread pool — and a probe that timed it would read the
    start-up as the per-fit cost and project it 60x. The run the page is
    about to make pays that cost once too, so it is not hidden, merely not
    multiplied.
    """
    target = max(int(size_target), 1)
    size = min(max(int(start_size), 1), target)
    points = []
    spent = 0.0
    try:
        run_at_size(size)
    except Exception:
        return None
    while True:
        started = time.perf_counter()
        try:
            run_at_size(size)
        except Exception:
            break
        took = time.perf_counter() - started
        points.append((size, took))
        spent += took
        if size >= target:
            break
        next_size = min(size * 2, target)
        slope_so_far = _slope(points)
        predicted = took * (next_size / size) ** (slope_so_far if slope_so_far is not None else SLOPE_ASSUMED)
        if spent + predicted > budget_seconds:
            break
        size = next_size
    if not points:
        return None
    slope = _slope(points)
    assumed = slope is None
    if assumed:
        slope = SLOPE_ASSUMED
    last_size, last_seconds = points[-1]
    if last_size >= target:
        at_target = last_seconds
    else:
        at_target = last_seconds * (target / last_size) ** max(slope, EXTRAPOLATION_SLOPE_FLOOR)
    return ScalingProbe(points=tuple(points), slope=slope, slope_assumed=assumed,
                        size_target=target, seconds_at_target=at_target)


def _slope(points: Sequence[Tuple[int, float]]) -> Optional[float]:
    """Log-log slope over the last three points above the timer floor, clamped.

    Three rather than two because a forest's timing between two adjacent
    sizes wobbles with its thread pool — measured 0.58 between 256 and 512
    rows and 1.47 between 512 and 1,024 on the same estimator — and a
    least-squares line through three points reads the trend under the wobble.
    Two points when only two exist; none when fewer.
    """
    usable = [(s, t) for s, t in points if t >= _TIMER_FLOOR_SECONDS]
    if len(usable) < 2:
        return None
    tail = usable[-3:]
    xs = np.log([s for s, _ in tail])
    ys = np.log([t for _, t in tail])
    if xs[-1] <= xs[0]:
        return None
    raw = float(np.polyfit(xs, ys, 1)[0])
    return min(max(raw, SLOPE_FLOOR), SLOPE_CEILING)


def probe_fit_seconds(fit: Callable[[np.ndarray, np.ndarray], None],
                      X, y, budget_seconds: float = PROBE_BUDGET_SECONDS,
                      seed: int = 0) -> Optional[ScalingProbe]:
    """Time `fit(X_sub, y_sub)` on nested row subsamples of the real frame.

    `fit` is whatever the page is about to do once: clone the configured
    estimator, push the rows through the model's own preprocessing, fit. The
    rows are a fixed seeded permutation so each larger sample contains the
    smaller one. `X` may be a DataFrame or an array; it is indexed
    positionally either way.
    """
    n = int(len(X))
    if n == 0:
        return None
    order = np.random.RandomState(seed).permutation(n)
    take = (lambda k: X.iloc[order[:k]]) if hasattr(X, "iloc") else (lambda k: X[order[:k]])
    y_arr = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)

    def _run(k: int) -> None:
        fit(take(k), y_arr[order[:k]])

    return probe_scaling(_run, n, budget_seconds=budget_seconds)


# ── the training run, in seconds ─────────────────────────────────────────────

#: Seconds the fold worker pool costs before any fold fits: process start-up,
#: the imports each worker repeats, the matrix pickled to each. Measured
#: through `ml.eval.perform_cross_validation` at 8,000 x 60 on 14 cores: a
#: Ridge whose fit takes 0.01 s takes 0.27 s to cross-validate; LightGBM,
#: whose first call in a fresh process costs over a second, 5.4 s for five
#: folds of a 0.36 s fit. One second is the middle of that range.
CV_POOL_STARTUP_SECONDS = 1.0

#: Five parallel fold fits cost about FIVE fold fits, not one. Measured
#: through the app's own CV path at 8,000 x 60 on 14 cores, wall over one
#: fold fit: RandomForest 4.96 (its pool is deliberately not pinned, so five
#: workers oversubscribe the cores), XGBoost 3.94, HistGradientBoosting 3.14,
#: SVR 3.08, ExtraTrees 1.77. joblib pins each worker to one thread, so a
#: fit that used every core alone gets one core in a worker, and what the
#: parallelism gives back the pinning takes. Charging k fold fits is exact
#: for the forest, generous for ExtraTrees, and within 2x for the rest —
#: and never the 5x-too-low that assuming perfect parallelism would be.
CV_FOLD_FITS_PER_FOLD = 1.0


def cv_wall_seconds(probe: ScalingProbe, n_train: int, folds: int) -> float:
    """Wall-clock seconds for `folds` parallel fold fits on `n_train` rows.

    Each fold fits on (folds-1)/folds of the rows, read off the probe's
    curve; the folds together cost about `folds` such fits (see
    `CV_FOLD_FITS_PER_FOLD`) plus the pool's start-up.
    """
    k = max(int(folds), 1)
    per_fold = probe.seconds_at(int(n_train * (k - 1) / k))
    return CV_POOL_STARTUP_SECONDS + k * per_fold * CV_FOLD_FITS_PER_FOLD


def cv_worker_bytes(train_matrix_bytes: int, folds: int, cores: Optional[int]) -> int:
    """Exact: every fold worker holds its own copy of the training matrix.

    joblib's loky backend pickles the matrix to each worker process, so the
    peak is one copy per concurrent worker plus the parent's. This is the
    audit's "five folds each take the whole box" as bytes rather than
    threads, and it is arithmetic — the one estimate here that needs no probe.
    """
    k = max(int(folds), 1)
    workers = max(min(k, int(cores) if cores else k), 1)
    return int(max(int(train_matrix_bytes), 0)) * (workers + 1)


@dataclass(frozen=True)
class ModelCost:
    """One model's share of a training run, in seconds, with its provenance."""

    key: str
    probe: Optional[ScalingProbe]
    final_fit_seconds: float
    cv_seconds: float
    optuna_seconds: float

    @property
    def seconds(self) -> float:
        return self.final_fit_seconds + self.cv_seconds + self.optuna_seconds

    @property
    def estimated(self) -> bool:
        return self.probe is not None


def training_run_cost(probes: Dict[str, Optional[ScalingProbe]], n_train: int,
                      cv_folds: int, cv_models: Sequence[str],
                      optuna_trials: Dict[str, int]) -> Tuple[ModelCost, ...]:
    """The three additive terms per model, from that model's probe.

    `cv_folds` is 0 when cross-validation is off; `cv_models` are the models
    the fold loop will actually run (pages/06 skips the neural network);
    `optuna_trials` maps each tunable model to its trial count for the
    optimized button, empty for the standard one. A model with no probe
    (its sample fit raised) costs 0 here and must be reported as not
    estimated by the caller — the count of such models is what
    `not_estimated` is for.
    """
    costs = []
    for key, probe in probes.items():
        if probe is None:
            costs.append(ModelCost(key, None, 0.0, 0.0, 0.0))
            continue
        final = probe.seconds_at_target
        cv = cv_wall_seconds(probe, n_train, cv_folds) if cv_folds and key in cv_models else 0.0
        trials = int(optuna_trials.get(key, 0))
        optuna = final * trials
        costs.append(ModelCost(key, probe, final, cv, optuna))
    return tuple(costs)


def not_estimated(costs: Sequence[ModelCost]) -> Tuple[str, ...]:
    return tuple(c.key for c in costs if not c.estimated)


def widest_extrapolation(costs: Sequence[ModelCost]) -> float:
    """The largest factor any model's number was extrapolated by — the
    honesty figure the page states beside the total."""
    return max((c.probe.extrapolation_factor for c in costs if c.probe), default=1.0)


# ── words ────────────────────────────────────────────────────────────────────

def humanize_seconds(seconds: float) -> str:
    """'under a second', 'about 40 seconds', 'about 4 minutes', 'about 1.5 hours'.

    Rounded coarsely on purpose: the number is an estimate extrapolated from
    a sample, and "about 3 minutes 40 seconds" would claim a precision the
    method does not have.
    """
    s = max(float(seconds), 0.0)
    if s < 1:
        return "under a second"
    if s < 10:
        whole = int(round(s))
        return "about a second" if whole == 1 else f"about {whole} seconds"
    if s < 90:
        return f"about {int(round(s / 5.0) * 5)} seconds"
    if s < 3 * 3600:
        minutes = s / 60.0
        return f"about {int(round(minutes))} minute{'' if int(round(minutes)) == 1 else 's'}"
    return f"about {s / 3600.0:.1f} hours"


def humanize_bytes(n: int) -> str:
    n = max(int(n), 0)
    if n < 1024:
        return f"{n} bytes"
    if n < 1024 ** 2:
        return f"{n / 1024:.0f} KB"
    if n < 1024 ** 3:
        return f"{n / 1024 ** 2:.0f} MB"
    return f"{n / 1024 ** 3:.1f} GB"


def provenance_clause(costs: Sequence[ModelCost]) -> str:
    """How the number was obtained, in one clause the page appends to it.

    Says "measured" only when every model's probe reached the full training
    set; otherwise names the sample and the factor, because that is the
    difference between a measurement and a projection.
    """
    probed = [c.probe for c in costs if c.probe]
    if not probed:
        return ""
    if all(p.measured_at_target for p in probed):
        return "measured just now on your full training set"
    smallest = min(p.size_probed for p in probed)
    factor = max(p.extrapolation_factor for p in probed)
    return (f"projected from sample fits of {smallest:,}+ of your training rows, "
            f"extrapolated up to {factor:,.0f}x")

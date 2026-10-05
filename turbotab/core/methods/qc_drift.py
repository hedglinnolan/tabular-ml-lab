"""QC-based drift and inter-batch correction, the QC filters, the pooled-QC reference for PQN,
and the D-ratio filter (MODELING_SEQUENCE §1.1 and MS7; BLUEPRINT §13's worked example).

The run order, before the seal and on reference rows only (the pooled QCs, technical replicates of
one pooled sample; never a participant and never the outcome):

1. **QC detection rate** (Broadhurst et al. 2018: "The acceptance criterion for detection rate is
   typically set to > 70%"): a feature detected in fewer than 70% of the QC injections leaves.
2. **QC-RLSC** (Dunn et al. 2011, Nat Protoc 6:1060: "quality control-based robust LOESS signal
   correction to provide signal correction and integration of data from multiple analytical
   batches"): for each feature and each batch, a LOESS of the feature's QC responses over
   injection order; every injection of that batch is divided by the curve at its injection order
   and multiplied by the feature's median QC response over all batches. Dividing by the batch's
   own curve removes both the drift within the batch and the batch's level, so the batches are
   aligned by the same step. The LOESS is R's ``stats::loess`` (Cleveland, Grosse & Shyu 1992),
   degree 2, ``family = "gaussian"``, evaluated exactly (``surface = "direct"``); its span is
   chosen per feature and batch by leave-one-out cross-validation over the QCs (Dunn et al.
   optimized the smoothing parameter by cross-validation to avoid overfitting). The curve is
   never extrapolated: a batch whose study injections fall before its first QC or after its last
   is refused, as is a batch with fewer than five QCs (Broadhurst et al. 2018: "at least five
   pooled QCs distributed evenly across a single batch").
3. **QC RSD** (Broadhurst et al. 2018: "acceptance criterion for RSD is typically set to < 20%
   … or < 30%"): on the corrected QC responses, a feature whose relative standard deviation is at
   least 20% (LC-MS) or 30% (GC-MS) leaves.
4. **The QC rows leave** (an eligibility step): the working table drops them as WP18's reference
   rows (``turbotab/core/reference_rows.py``), so the outcome is read on participants only and no
   QC can be drawn into the held-out rows, and the participant flow counts them on a line of
   their own.
5. **PQN against the pooled-QC reference** (when that normalization is chosen): the reference
   spectrum is the feature-wise median of the integral-normalized QC injections, as pmp's PQN
   function (Bioconductor) uses the QC samples by default; each injection is divided by the median of
   its quotients to it. A study row's result then depends on no other participant.

The **D-ratio** (Broadhurst et al. 2018: the QC standard deviation over the study samples'; "the
acceptance criterion for D-ratio is set to, at most, < 50%") reads the study samples' spread, so
it is training-fold scope: it runs in each training fold (:class:`DRatioFilter`), with the QC
standard deviations frozen from the reference rows, and the pre-seal executor refuses it
(:func:`run_reference_rows`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

DEGREE = 2
SPANS = tuple(float(s) for s in np.round(np.arange(0.30, 1.0001, 0.05), 2))
MIN_QC = 5  # QCs per batch the curve is fit to (Broadhurst et al. 2018)
MAX_GAP = 10  # a pooled QC at least every 10th injection (Broadhurst et al. 2018)
RSD_MAX = {"lc_ms": 20.0, "gc_ms": 30.0}
PLATFORM_WORDS = {"lc_ms": "LC-MS", "gc_ms": "GC-MS"}
DETECTION_MIN = 0.70
D_RATIO_MAX = 0.50
FINDING = "pack::metabolomics::pooled_qc"
QC_FILE = "qc_reference.parquet"
# The corrected pooled-QC injections, kept beside the working table for quality assessment once
# they have left it (WP18's reference rows).
QC_ROWS_FILE = "qc_rows.parquet"

DUNN = "Dunn et al. 2011"
BROADHURST = "Broadhurst et al. 2018"


# ── LOESS, as R's stats::loess computes it ───────────────────────────────────


def _neighbors(n: int, span: float) -> int:
    """R's ``loess_raw``: ``nf = min(n, floor(n * span + 1e-5))``."""
    return min(n, int(math.floor(n * span + 1e-5)))


def smoother_rows(x: np.ndarray, at: np.ndarray, span: float, degree: int = DEGREE,
                  robustness: np.ndarray | None = None) -> np.ndarray:
    """The rows of the local-regression smoother: ``fit(at) = rows @ y`` for data at ``x``.

    Cleveland's local regression as ``stats::loess`` evaluates it exactly (``surface =
    "direct"``): at each point, the ``q = floor(n · span)`` nearest data points get tricube weights
    ``(1 − (d / d_q)³)³`` (``d_q`` the distance to the q-th nearest, which itself gets weight 0),
    times any robustness weights, and a weighted polynomial of ``degree`` in ``x − at`` is fit by
    least squares; its intercept is the value. Spans above 1 are not used here.
    """
    x = np.asarray(x, dtype=float)
    at = np.asarray(at, dtype=float)
    n = len(x)
    if not 0 < span <= 1:
        raise ValueError("a span is in (0, 1]")
    q = _neighbors(n, span)
    if q < degree + 1:
        raise ValueError(f"span {span:g} keeps {q} of {n} points: too few for degree {degree}")
    rob = np.ones(n) if robustness is None else np.asarray(robustness, dtype=float)
    out = np.empty((len(at), n))
    for i, a in enumerate(at):
        d = np.abs(x - a)
        rho = np.partition(d, q - 1)[q - 1]
        u = np.where(d < rho, d / rho if rho > 0 else 0.0, 1.0)
        w = (1.0 - u ** 3) ** 3 * rob
        sw = np.sqrt(w)
        X = np.vander(x - a, degree + 1, increasing=True)
        # least squares of sqrt(w)·y on sqrt(w)·X: the intercept's row of pinv(√W X) √W
        out[i] = np.linalg.pinv(X * sw[:, None])[0] * sw
    return out


def _lowesw(residuals: np.ndarray) -> np.ndarray:
    """R's ``lowesw``: bisquare robustness weights on residuals scaled by six median absolute
    residuals; a residual within 0.1% of the scale keeps weight 1, one beyond 99.9% gets 0."""
    r = np.abs(residuals)
    cmad = 6.0 * float(np.median(r))
    if cmad < np.finfo(float).tiny:
        return np.ones_like(r)
    return np.where(r > 0.999 * cmad, 0.0,
                    np.where(r > 0.001 * cmad, (1.0 - (r / cmad) ** 2) ** 2, 1.0))


def loess(x: Sequence[float], y: Sequence[float], span: float, degree: int = DEGREE,
          family: str = "gaussian", at: Sequence[float] | None = None
          ) -> tuple[np.ndarray, np.ndarray | None]:
    """``(fitted, predicted)`` as ``loess(y ~ x, span, degree, family, control =
    loess.control(surface = "direct"))`` and ``predict(fit, at)`` give them. ``family =
    "symmetric"`` adds R's four robustness iterations."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    iterations = 1 if family == "gaussian" else 4
    rob = np.ones(len(x))
    used = rob
    fitted = np.empty(len(x))
    for j in range(iterations):
        used = rob
        fitted = smoother_rows(x, x, span, degree, used) @ y
        if j < iterations - 1:
            rob = _lowesw(y - fitted)
    predicted = None if at is None else smoother_rows(x, np.asarray(at, dtype=float), span, degree, used) @ y
    return fitted, predicted


def _loo_rows(x: np.ndarray, span: float, degree: int) -> np.ndarray | None:
    """``L`` with ``(L @ y)[i]`` the fit at ``x[i]`` from the other points: leave-one-out, as a
    linear operator shared by every feature measured at the same points. None when the span keeps
    too few of the other points for the degree (one more than it needs, so no fit interpolates)."""
    n = len(x)
    if _neighbors(n - 1, span) < degree + 2:
        return None
    L = np.zeros((n, n))
    for i in range(n):
        others = np.delete(np.arange(n), i)
        L[i, others] = smoother_rows(x[others], x[i:i + 1], span, degree)[0]
    return L


def choose_spans(x: np.ndarray, Y: np.ndarray, degree: int = DEGREE,
                 spans: Sequence[float] = SPANS) -> tuple[np.ndarray, np.ndarray]:
    """Each column's span by leave-one-out cross-validation over ``spans``: the one with the
    smallest mean squared leave-one-out error (ties to the larger, smoother span). Returns
    ``(spans, errors)``; a column no span can fit gets NaN."""
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    if Y.shape[0] != len(x):
        Y = Y.T
    best = np.full(Y.shape[1], np.nan)
    error = np.full(Y.shape[1], np.inf)
    for s in spans:
        L = _loo_rows(x, s, degree)
        if L is None:
            continue
        mse = np.mean((Y - L @ Y) ** 2, axis=0)
        better = mse <= error * (1 + 1e-12)  # ties go to the larger span, which comes later
        best = np.where(better, s, best)
        error = np.where(better, mse, error)
    return best, np.where(np.isfinite(error), error, np.nan)


# ── the injections: order, batch, and which are pooled QCs ───────────────────


@dataclass(frozen=True)
class Injections:
    order: np.ndarray  # injection order (any increasing numbers)
    batch: np.ndarray  # batch label per injection ("" when there is one batch)
    qc: np.ndarray  # True for a pooled-QC injection

    @property
    def batches(self) -> list[str]:
        return sorted(dict.fromkeys(str(b) for b in self.batch))


def injections(frame: pd.DataFrame, qc_column: str, qc_levels: Sequence[Any],
               order_column: str | None, batch_column: str | None) -> Injections:
    levels = {_label(v) for v in qc_levels}
    qc = frame[qc_column].map(lambda v: _label(v) in levels).to_numpy(dtype=bool)
    order = (pd.to_numeric(frame[order_column], errors="coerce").to_numpy(dtype=float)
             if order_column else np.arange(len(frame), dtype=float))
    batch = (frame[batch_column].map(_label).to_numpy(dtype=object) if batch_column
             else np.full(len(frame), "", dtype=object))
    return Injections(order=order, batch=batch, qc=qc)


def _label(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, (int, float, np.integer, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def sufficiency(inj: Injections) -> list[str]:
    """Why QC-RLSC cannot be fit here, one sentence per batch that fails (empty: it can be).

    A batch needs at least :data:`MIN_QC` QC injections, and its study injections must lie within
    its QCs' span of injection order: the curve is interpolated, never extrapolated."""
    problems: list[str] = []
    if not np.isfinite(inj.order).all():
        problems.append("Some injections have no injection order, so no curve can place them.")
        return problems
    for b in inj.batches:
        at = inj.batch == b
        name = f"Batch `{b}`" if b else "The run"
        qc_orders = np.sort(inj.order[at & inj.qc])
        study = inj.order[at & ~inj.qc]
        if len(qc_orders) < MIN_QC:
            problems.append(f"{name} has {len(qc_orders)} pooled-QC injections; a drift curve needs "
                            f"at least {MIN_QC} ({BROADHURST}).")
            continue
        early = int((study < qc_orders[0]).sum())
        late = int((study > qc_orders[-1]).sum())
        if early or late:
            where = " and ".join(p for p in (f"{early} before its first QC" if early else "",
                                             f"{late} after its last" if late else "") if p)
            problems.append(f"{name} has study injections outside its QCs ({where}): the curve "
                            f"would be extrapolated. A run must begin and end with QC injections "
                            f"({DUNN}).")
    return problems


def largest_gap(inj: Injections) -> int:
    """The most study injections between two consecutive QCs of a batch."""
    worst = 0
    for b in inj.batches:
        at = inj.batch == b
        orders = np.sort(inj.order[at])
        qc_orders = set(inj.order[at & inj.qc].tolist())
        run = 0
        for o in orders:
            run = 0 if o in qc_orders else run + 1
            worst = max(worst, run)
    return worst


# ── QC-RLSC ──────────────────────────────────────────────────────────────────


@dataclass
class DriftFit:
    corrected: np.ndarray  # n × p, every injection
    curves: np.ndarray  # n × p, the fitted QC level at each injection (NaN where not corrected)
    spans: dict[str, np.ndarray]  # batch -> span per feature
    level: np.ndarray  # per feature: the median QC response the corrected values are on
    uncorrectable: list[int]  # features some batch could not fit
    reasons: dict[int, str] = field(default_factory=dict)


def _detected(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values) & (values > 0)


def qc_rlsc(values: np.ndarray, inj: Injections, *, degree: int = DEGREE,
            span: float | None = None) -> DriftFit:
    """QC-RLSC on an ``n × p`` block (rows in any order). ``span`` fixes the span; None chooses
    it per feature and batch by leave-one-out cross-validation (:func:`choose_spans`).

    A feature that a batch cannot fit — fewer than :data:`MIN_QC` detected QCs there, or a curve
    that reaches zero — is not corrected anywhere and is listed in ``uncorrectable``.
    """
    values = np.asarray(values, dtype=float)
    n, p = values.shape
    detected = _detected(values)
    qc_vals = np.where(detected & inj.qc[:, None], values, np.nan)
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            level = np.nanmedian(qc_vals, axis=0)
    curves = np.full((n, p), np.nan)
    spans: dict[str, np.ndarray] = {}
    bad: dict[int, str] = {}
    for b in inj.batches:
        rows = np.flatnonzero(inj.batch == b)
        qc_rows = rows[inj.qc[rows]]
        qc_rows = qc_rows[np.argsort(inj.order[qc_rows], kind="stable")]
        x_all = inj.order[rows]
        chosen = np.full(p, np.nan)
        patterns: dict[bytes, list[int]] = {}
        for j in range(p):
            patterns.setdefault(detected[qc_rows, j].tobytes(), []).append(j)
        for key, cols in patterns.items():
            mask = np.frombuffer(key, dtype=bool)
            use = qc_rows[mask]
            if len(use) < MIN_QC:
                for j in cols:
                    bad.setdefault(j, f"fewer than {MIN_QC} detected QCs in batch `{b}`" if b
                                   else f"fewer than {MIN_QC} detected QCs")
                continue
            x = inj.order[use]
            Y = values[np.ix_(use, cols)]
            if span is None:
                s, _ = choose_spans(x, Y, degree)
            else:
                s = np.full(len(cols), float(span))
            for value in np.unique(s[np.isfinite(s)]):
                these = [c for c, sv in zip(cols, s) if sv == value]
                rows_op = smoother_rows(x, x_all, float(value), degree)
                curves[np.ix_(rows, these)] = rows_op @ values[np.ix_(use, these)]
                chosen[these] = value
            for c, sv in zip(cols, s):
                if not np.isfinite(sv):
                    bad.setdefault(c, "no span fits its QCs")
        spans[b] = chosen
    with np.errstate(all="ignore"):
        nonpositive = np.where(np.isfinite(curves), curves <= 0, False).any(axis=0)
    for j in np.flatnonzero(nonpositive):
        bad.setdefault(int(j), "its drift curve reaches zero")
    corrected = values.copy()
    good = np.array([j not in bad for j in range(p)], dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        corrected[:, good] = values[:, good] / curves[:, good] * level[good]
    curves[:, ~good] = np.nan
    return DriftFit(corrected=corrected, curves=curves, spans=spans, level=level,
                    uncorrectable=sorted(bad), reasons=bad)


# ── the QC filters ──────────────────────────────────────────────────────────


def detection_rate(values: np.ndarray, qc: np.ndarray) -> np.ndarray:
    """Per feature: the share of QC injections with a detected (positive) value."""
    block = np.asarray(values, dtype=float)[np.asarray(qc, dtype=bool)]
    if not len(block):
        return np.full(np.asarray(values).shape[1], np.nan)
    return _detected(block).mean(axis=0)


def qc_rsd(values: np.ndarray, qc: np.ndarray) -> np.ndarray:
    """Per feature: 100 × SD / mean of its detected QC responses (the sample SD, n − 1)."""
    block = np.asarray(values, dtype=float)[np.asarray(qc, dtype=bool)]
    block = np.where(_detected(block), block, np.nan)
    import warnings

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return 100.0 * np.nanstd(block, axis=0, ddof=1) / np.nanmean(block, axis=0)


def qc_sd(values: np.ndarray, qc: np.ndarray) -> np.ndarray:
    block = np.asarray(values, dtype=float)[np.asarray(qc, dtype=bool)]
    block = np.where(_detected(block), block, np.nan)
    import warnings

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanstd(block, axis=0, ddof=1)


# ── PQN against the pooled-QC reference ──────────────────────────────────────


def qc_reference(values: np.ndarray, qc: np.ndarray) -> np.ndarray:
    """The feature-wise median of the QC injections, each first scaled to the QCs' median total
    (the integral normalization; positive values only), as :class:`omics.QuotientLog` builds its
    reference from training rows."""
    block = np.asarray(values, dtype=float)[np.asarray(qc, dtype=bool)]
    positive = np.where(_detected(block), block, np.nan)
    import warnings

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        totals = np.nansum(positive, axis=1)
        ok = np.isfinite(totals) & (totals > 0)
        integral = float(np.median(totals[ok])) if ok.any() else 1.0
        scaled = positive[ok] * (integral / totals[ok])[:, None]
        return np.nanmedian(scaled, axis=0)


def dilution(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Each row's most probable dilution: the median of its quotients to ``reference`` (positive
    values only; 1 when there are none)."""
    values = np.asarray(values, dtype=float)
    import warnings

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        q = np.where(_detected(values) & np.isfinite(reference) & (reference > 0),
                     values / reference, np.nan)
        factor = np.nanmedian(q, axis=1)
    return np.where(np.isfinite(factor) & (factor > 0), factor, 1.0)


# ── the D-ratio: training fold ───────────────────────────────────────────────


def d_ratio(qc_sd_values: np.ndarray, study: np.ndarray) -> np.ndarray:
    """The QC standard deviation over the study samples' (both n − 1), per feature."""
    block = np.asarray(study, dtype=float)
    block = np.where(_detected(block), block, np.nan)
    import warnings

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.asarray(qc_sd_values, dtype=float) / np.nanstd(block, axis=0, ddof=1)


class DRatioFilter(TransformerMixin, BaseEstimator):
    """Keeps the features whose D-ratio on the fitting rows is below ``threshold``.

    ``qc_sd`` (feature -> the QC standard deviation, frozen from the reference rows before the
    seal) is the numerator; the denominator is the standard deviation of the rows it is fit on, so
    in cross-validation it is each training fold's, and a held-out row never moves it. A feature
    without a QC standard deviation is kept. Every other column passes through."""

    def __init__(self, columns: Sequence[str] = (), qc_sd: Mapping[str, float] | None = None,
                 threshold: float = D_RATIO_MAX):
        self.columns = columns
        self.qc_sd = qc_sd
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: Any = None) -> "DRatioFilter":
        self.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)
        self.n_features_in_ = X.shape[1]
        sds = dict(self.qc_sd or {})
        cols = [c for c in self.columns if c in X.columns and c in sds]
        ratio = d_ratio(np.array([sds[c] for c in cols], dtype=float),
                        X[cols].to_numpy(dtype=float, na_value=np.nan)) if cols else np.array([])
        self.ratio_ = {c: float(r) for c, r in zip(cols, ratio)}
        self.dropped_ = [c for c, r in self.ratio_.items() if not (r < self.threshold)]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "dropped_"):
            raise ValueError("DRatioFilter is not fitted yet.")
        return X.drop(columns=[c for c in self.dropped_ if c in X.columns])

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        gone = set(self.dropped_)
        return np.asarray([c for c in self.feature_names_in_ if c not in gone], dtype=object)

    def lineage(self) -> list[dict[str, Any]]:
        gone = set(self.dropped_)
        return [{"output": c, "inputs": [c], "operation": "kept (D-ratio)" if c in set(self.columns)
                 else "kept"} for c in self.feature_names_in_ if c not in gone]


# ── the pre-seal executor ────────────────────────────────────────────────────


@dataclass(frozen=True)
class QCPlan:
    """What the pooled-QC answer runs before the seal."""

    qc_column: str
    qc_levels: tuple[str, ...]
    features: tuple[str, ...]
    order_column: str | None = None
    batch_column: str | None = None
    correct: bool = True  # QC-RLSC and the QC filters; False: the QC rows only leave
    platform: str = "lc_ms"
    pqn: bool = False  # PQN against the pooled-QC reference (the omics_scale answer)
    d_ratio_max: float | None = None  # the in-fold D-ratio filter's threshold
    span: float | None = None  # None: leave-one-out cross-validation
    degree: int = DEGREE

    def steps(self) -> list[str]:
        out = []
        if self.correct:
            out += ["qc_detection_filter", "qc_rlsc", "qc_rsd_filter"]
        if self.pqn:
            out.append("qc_pqn")
        return out


@dataclass
class QCResult:
    values: np.ndarray  # n × p, the block after the steps
    record: dict[str, Any]
    qc_sd: np.ndarray  # per feature, after the steps (the D-ratio's numerator)
    kept: list[str]
    rsd: np.ndarray | None = None  # per feature, the corrected QCs' RSD (when the filter ran)


def run_reference_rows(frame: pd.DataFrame, plan: QCPlan,
                       steps: Sequence[str] | None = None) -> QCResult:
    """The pre-seal chain on ``frame`` (every injection, QC and study): ``steps`` in order, each a
    registered contract whose scope lets it run before the seal. A step that reads participants
    (the D-ratio) is refused here: it belongs in each training fold."""
    from turbotab.core.decisions import Refusal
    from turbotab.core.methods.contract import PRE_SEAL_SCOPES, contract

    steps = list(plan.steps() if steps is None else steps)
    for key in steps:
        c = contract(key)
        if c.scope not in PRE_SEAL_SCOPES:
            raise Refusal(
                "training_fold_before_seal",
                f"{c.label} reads the study samples ({c.scope.replace('_', ' ')} scope), so it "
                f"cannot run before the seal: it runs in each training fold, where held-out rows "
                f"never move it.",
                exits=[{"label": "Run it in each training fold", "decision": None}])
    features = [c for c in plan.features if c in frame.columns]
    values = frame[features].to_numpy(dtype=float, na_value=np.nan)
    inj = injections(frame, plan.qc_column, plan.qc_levels, plan.order_column, plan.batch_column)
    keep = np.ones(len(features), dtype=bool)
    rsd_values = None
    record: dict[str, Any] = {"n_qc": int(inj.qc.sum()), "n_features": len(features),
                              "batches": inj.batches if plan.batch_column else [],
                              "platform": plan.platform, "steps": steps,
                              "dropped": {"detection": [], "uncorrectable": [], "rsd": []}}
    for key in steps:
        if key == "qc_detection_filter":
            rate = detection_rate(values, inj.qc)
            low = keep & ~(rate >= DETECTION_MIN)
            record["dropped"]["detection"] = [features[j] for j in np.flatnonzero(low)]
            keep &= ~low
        elif key == "qc_rlsc":
            problems = sufficiency(inj)
            if problems:
                raise Refusal("qc_too_sparse", " ".join(problems),
                              exits=[{"label": "Set the QC rows aside without correcting drift",
                                      "decision": None}])
            idx = np.flatnonzero(keep)
            fit = qc_rlsc(values[:, idx], inj, degree=plan.degree, span=plan.span)
            values[:, idx] = fit.corrected
            bad = [int(idx[j]) for j in fit.uncorrectable]
            record["dropped"]["uncorrectable"] = [features[j] for j in bad]
            keep[bad] = False
            record["spans"] = {b: _span_summary(s[np.isfinite(s)]) for b, s in fit.spans.items()}
            record["largest_gap"] = largest_gap(inj)
            record["qc_per_batch"] = {b: int((inj.qc & (inj.batch == b)).sum()) for b in inj.batches}
        elif key == "qc_rsd_filter":
            rsd = rsd_values = qc_rsd(values, inj.qc)
            high = keep & ~(rsd < RSD_MAX[plan.platform])
            record["dropped"]["rsd"] = [features[j] for j in np.flatnonzero(high)]
            keep &= ~high
        elif key == "qc_pqn":
            idx = np.flatnonzero(keep)
            reference = qc_reference(values[:, idx], inj.qc)
            factor = dilution(values[:, idx], reference)
            values[:, idx] = values[:, idx] / factor[:, None]
            record["pqn_factor_range"] = [float(np.min(factor)), float(np.max(factor))]
    kept = [features[j] for j in np.flatnonzero(keep)]
    record["kept"] = len(kept)
    return QCResult(values=values, record=record, qc_sd=qc_sd(values, inj.qc), kept=kept,
                    rsd=rsd_values)


def _span_summary(spans: np.ndarray) -> dict[str, float] | None:
    if not spans.size:
        return None
    return {"min": float(spans.min()), "median": float(np.median(spans)), "max": float(spans.max())}


# ── the pooled-QC answer: a repair on the metabolomics pack's pooled-QC finding ──

RLSC_OPTIONS = ("qc_rlsc_lc", "qc_rlsc_lc_dratio", "qc_rlsc_gc", "qc_rlsc_gc_dratio")
# Setting the QCs aside without correcting drift is WP18's exclusion of the reference rows, one
# answer offered once (``turbotab/core/reference_rows.py`` owns it and the family both share).
ASIDE = "exclude_rows"
OPTIONS = (*RLSC_OPTIONS, ASIDE)


def _order_column(frame: pd.DataFrame, exclude: Sequence[str]) -> str | None:
    """The injection-order column: an acquisition column read as a run order whose values order
    the injections (numbers, no blanks), else a column that is a permutation of the rows."""
    from turbotab.core.recognizers import acquisition_kind

    for c in frame.columns:
        if str(c) in exclude or acquisition_kind(str(c)) != "run_order":
            continue
        values = pd.to_numeric(frame[c], errors="coerce")
        if values.notna().all() and values.nunique() == len(values):
            return str(c)
    try:
        from turbotab import packs

        found = packs._permutation_column(frame)
    except Exception:  # noqa: BLE001 - no pack, no permutation reading
        found = None
    return found if found not in exclude else None


def _batch_column(frame: pd.DataFrame, exclude: Sequence[str]) -> str | None:
    from turbotab.core.recognizers import acquisition_kind

    for c in frame.columns:
        if str(c) not in exclude and acquisition_kind(str(c)) == "batch" \
                and frame[c].nunique(dropna=True) >= 2:
            return str(c)
    return None


def _features(frame: pd.DataFrame, target: str | None, exclude: Sequence[str]) -> list[str]:
    from turbotab.core.methods.omics import candidate_columns

    gone = set(exclude)
    return [c for c in candidate_columns(frame, target) if c not in gone]


def rlsc_offer(finding: dict[str, Any], p: dict[str, Any], oc: Any) -> list[Any]:
    """The QC-RLSC options for the pooled-QC finding (an injection order is needed; none without
    it). The exclusion without correction is offered beside them by the reference-rows family
    (``reference_rows._offer``), which answers this finding."""
    from turbotab.core.decisions import ApplyRepair
    from turbotab.core.repairs import RepairOption
    from turbotab.core.voice import finish

    column = p.get("column")
    if not oc.has(column) or p.get("qc_value") is None:
        return []
    frame = oc.frame
    levels = [str(p["qc_value"])]
    order = _order_column(frame, [column])
    batch = _batch_column(frame, [column, *([order] if order else [])])
    features = _features(frame, oc.target, [column, *([order] if order else []),
                                            *([batch] if batch else [])])
    if len(features) < 2:
        return []
    inj = injections(frame, column, levels, order, batch)
    n_qc = int(inj.qc.sum())
    base = {"column": column, "qc_levels": levels, "order_column": order, "batch_column": batch,
            "features": features, "n_qc": n_qc}

    def option(key: str, label: str, consequence: str, sentence: str, params: dict[str, Any]) -> Any:
        return RepairOption(key=key, label=finish(label, terminal=False),
                            consequence=finish(consequence),
                            row_local=key == ASIDE, effect="rows" if key == ASIDE else "values",
                            sentence=finish(sentence),
                            decision=ApplyRepair(finding_id=str(finding["id"]), option=key,
                                                 params=params))

    out: list[Any] = []
    if order is not None:
        problems = sufficiency(inj)
        fitted: dict[str, Any] = {}
        if not problems:
            result = run_reference_rows(frame, QCPlan(column, tuple(levels), tuple(features), order,
                                                      batch, platform="lc_ms"))
            fitted = result.record
            rsd = dict(zip(features, result.rsd)) if result.rsd is not None else {}
        for platform in ("lc_ms", "gc_ms"):
            for d_ratio in (None, D_RATIO_MAX):
                key = f"qc_rlsc_{'lc' if platform == 'lc_ms' else 'gc'}{'_dratio' if d_ratio else ''}"
                # A recorded answer's params hold no empty list (``repairs.admits`` reads a list as
                # the values chosen from it): an absent key is an empty one.
                params = {**base, "platform": platform, "d_ratio_max": d_ratio}
                if problems:
                    params["problems"] = problems
                if fitted:
                    gone = fitted["dropped"]
                    before = set(gone["detection"]) | set(gone["uncorrectable"])
                    high = [c for c in features if c not in before
                            and not (rsd.get(c, np.inf) < RSD_MAX[platform])]
                    dropped = {"detection": list(gone["detection"]),
                               "uncorrectable": list(gone["uncorrectable"]), "rsd": high}
                    dropped = {k: v for k, v in dropped.items() if v}
                    if dropped:
                        params["dropped"] = dropped
                    params["kept"] = len(features) - len(before) - len(high)
                    params["batches"] = len(inj.batches) if batch else 1
                words = PLATFORM_WORDS[platform]
                label = f"Drift; {words}{'; D-ratio' if d_ratio else ' filters'}"
                if problems:
                    consequence = "Cannot run: too few QCs, or study injections outside them."
                else:
                    lost = len(features) - params["kept"]
                    consequence = (f"Corrects drift from `{n_qc}` QCs; drops `{lost}` features; QCs "
                                   f"leave{'; D-ratio in-fold' if d_ratio else ''}.")
                out.append(option(key, label, consequence, rlsc_sentence(params), params))
    return out


def rlsc_sentence(params: Mapping[str, Any]) -> str:
    """The methods sentence a QC-RLSC answer earns, with its own counts."""
    n_qc = int(params.get("n_qc") or 0)
    platform = str(params.get("platform") or "lc_ms")
    gone = params.get("dropped") or {}
    n = len(params.get("features") or [])
    batches = int(params.get("batches") or 1)
    per = f"each of the {batches} batches" if batches > 1 else "the run"
    text = (f"Drift was corrected per batch by QC-RLSC ({DUNN}) fitted to the {n_qc} pooled QC "
            f"injections only, which were then removed: for each feature and {per}, a LOESS of degree "
            f"2 over injection order, its span chosen by leave-one-out cross-validation, divided "
            f"each injection's value, rescaled to the feature's median QC value.")
    if params.get("kept") is not None:
        text += (f" Features detected in fewer than 70% of QC injections ({len(gone.get('detection') or [])}), "
                 f"that could not be corrected ({len(gone.get('uncorrectable') or [])}), or with a QC "
                 f"RSD of {RSD_MAX[platform]:g}% or more after correction (the "
                 f"{PLATFORM_WORDS[platform]} criterion; {len(gone.get('rsd') or [])}) were removed "
                 f"({BROADHURST}); {int(params.get('kept') or 0)} of {n} remain.")
    if params.get("d_ratio_max"):
        text += (f" Within each training fold, features whose D-ratio (QC over study-sample standard "
                 f"deviation) was {float(params['d_ratio_max']):.0%} or more were removed.")
    return text


def rlsc_details(params: Mapping[str, Any], record: Mapping[str, Any] | None = None) -> str:
    """The QC answer's specifics for the methods paragraph, after its first clause."""
    n_qc = int(params.get("n_qc") or 0)
    platform = str(params.get("platform") or "lc_ms")
    gone = (record or {}).get("dropped") or params.get("dropped") or {}
    n = len(params.get("features") or [])
    kept = (record or {}).get("kept", params.get("kept"))
    batches = int(params.get("batches") or 1)
    per = f"each of the {batches} batches" if batches > 1 else "the run"
    text = (f"QC-RLSC ({DUNN}) fitted, for each feature and {per}, a LOESS of degree 2 over "
            f"injection order to the {n_qc} pooled QC injections, its span chosen by leave-one-out "
            f"cross-validation; each injection was divided by the curve and rescaled to the "
            f"feature's median QC value.")
    if kept is not None:
        text += (f" Features detected in fewer than 70% of QC injections "
                 f"({len(gone.get('detection') or [])}), that could not be corrected "
                 f"({len(gone.get('uncorrectable') or [])}), or with a QC RSD of "
                 f"{RSD_MAX[platform]:g}% or more after correction ({len(gone.get('rsd') or [])}; the "
                 f"{PLATFORM_WORDS[platform]} criterion) were removed ({BROADHURST}); {int(kept)} of "
                 f"{n} remain.")
    return text


def _columns_out(option: str, params: Mapping[str, Any]) -> list[str]:
    """Columns the answer takes out of the predictors: the QC label (a constant once the QCs
    leave), the injection order once drift is corrected from it, and the features the QC filters
    removed."""
    if option not in RLSC_OPTIONS:
        return []
    out = [str(params.get("column"))] if params.get("column") else []
    if option in RLSC_OPTIONS:
        if params.get("order_column"):
            out.append(str(params["order_column"]))
        gone = params.get("dropped") or {}
        for key in ("detection", "uncorrectable", "rsd"):
            out.extend(str(c) for c in gone.get(key) or [])
    return list(dict.fromkeys(out))


def _marks(option: str, params: Mapping[str, Any]) -> set[tuple[str, str]]:
    """What a QC-RLSC answer does: corrects each feature, and (as the exclusion does) makes the QC
    rows leave, so the naming census's finding about the same rows is answered by it too."""
    if option not in RLSC_OPTIONS:
        return set()
    levels = params.get("qc_levels") or params.get("levels") or []
    return ({(str(c), f"qc:{option}") for c in params.get("features") or []}
            | {(str(params.get("column")), f"reference:{level}") for level in levels})


def _applied(state: Any) -> tuple[str, str, dict[str, Any]] | None:
    """``(finding id, option, params)`` of the applied pooled-QC answer, or None."""
    from turbotab.core.stages.finding_words import family

    for fid, d in dict(getattr(state, "findings", None) or {}).items():
        if family(str(fid)) != FINDING:
            continue
        get = d.get if isinstance(d, Mapping) else (lambda k, _d=d: getattr(_d, k, None))
        if get("action") == "applied" and get("option") in OPTIONS:
            return str(fid), str(get("option")), dict(get("params") or {})
    return None


def _qc_answer_fits(decision: Any, ctx: Any) -> None:
    """QC-RLSC is refused, with its reason, where it cannot be fit: fewer than five QCs in a batch,
    or study injections outside a batch's QCs (the curve would be extrapolated)."""
    from turbotab.core.decisions import Refusal
    from turbotab.core.stages.finding_words import family

    if family(str(decision.finding_id)) != FINDING or decision.option not in RLSC_OPTIONS:
        return
    params = dict(decision.params or {})
    if not params:
        reader = ctx.get("artifact") if isinstance(ctx, Mapping) else getattr(ctx, "artifact", None)
        found = None
        if callable(reader):
            try:
                data = getattr(reader("findings"), "data", None) or reader("findings")
                found = next((f for f in (data or {}).get("findings") or []
                              if f.get("id") == decision.finding_id), None)
            except Exception:  # noqa: BLE001 - no findings: nothing to check against
                found = None
        same = [o for o in (found or {}).get("repairs") or [] if o.get("key") == decision.option]
        params = dict(same[0]["decision"]["params"]) if same else {}
    problems = list(params.get("problems") or [])
    if problems:
        aside = {"column": params.get("column"), "levels": list(params.get("qc_levels") or [])}
        raise Refusal("qc_too_sparse", " ".join(problems),
                      exits=[{"label": "Set the QC rows aside without correcting drift",
                              "decision": {"kind": "apply_repair", "finding_id": decision.finding_id,
                                           "option": ASIDE, "params": aside}}])


def _qc_pqn_needs_the_qcs(decision: Any, ctx: Any) -> None:
    """PQN against the pooled QCs needs the QC rows settled first (which rows are QCs)."""
    from turbotab.core.decisions import Refusal, _state
    from turbotab.core.stages.finding_words import family

    if family(str(decision.finding_id)) != "omics_scale" or decision.option != "qc_pqn_log2":
        return
    if _applied(_state(ctx)) is None:
        raise Refusal("qc_rows_unsettled",
                      "PQN against the pooled QCs needs the QC injections named first: answer the "
                      "finding about pooled QC rows, then choose this normalization.",
                      exits=[{"label": "Answer the pooled-QC finding first", "decision": None}])


# ── what the working table, the cohort and the design read ───────────────────


def reference_plan(state: Any) -> QCPlan | None:
    """The pre-seal chain the working table runs (QC-RLSC and its filters, and PQN against the
    pooled QCs when that normalization is chosen), or None when nothing changes a value."""
    found = _applied(state)
    if found is None:
        return None
    _, option, p = found
    from turbotab.core.methods.omics import normalization_of

    norm = normalization_of(state)
    pqn = norm is not None and norm["method"] == "qc_pqn_log2"
    correct = option in RLSC_OPTIONS and not p.get("problems")
    if not correct and not pqn:
        return None
    # The exclusion records only the rows (WP18's ``{column, levels}``): PQN against the QCs then
    # normalizes the columns the normalization answer names.
    features = p.get("features") or (norm or {}).get("columns") or ()
    return QCPlan(qc_column=str(p["column"]),
                  qc_levels=tuple(str(v) for v in p.get("qc_levels") or p.get("levels") or ()),
                  features=tuple(str(c) for c in features),
                  order_column=p.get("order_column"), batch_column=p.get("batch_column"),
                  correct=correct, platform=str(p.get("platform") or "lc_ms"), pqn=pqn,
                  d_ratio_max=p.get("d_ratio_max"))


def correct_table(path: Any, plan: QCPlan, qc_out: Any) -> dict[str, Any]:
    """Run the plan on the parquet table at ``path`` (every injection), write the corrected feature
    columns back in place, and the QCs' per-feature statistics to ``qc_out``. Returns the record."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    names = [plan.qc_column, *(c for c in (plan.order_column, plan.batch_column) if c)]
    features = [c for c in plan.features if c in table.column_names]
    frame = table.select(list(dict.fromkeys([*names, *features]))).to_pandas()
    result = run_reference_rows(frame, plan)
    for j, c in enumerate(features):
        i = table.column_names.index(c)
        table = table.set_column(i, pa.field(c, pa.float64()), pa.array(result.values[:, j],
                                                                        type=pa.float64()))
    pq.write_table(table, path, compression="zstd")
    kept = set(result.kept)
    pd.DataFrame({"feature": features, "qc_sd": result.qc_sd,
                  "kept": [c in kept for c in features]}).to_parquet(qc_out, index=False)
    return result.record


def working_qc_sd(bundle: Any) -> dict[str, float] | None:
    """The pooled QCs' per-feature standard deviation the working table recorded, or None."""
    files = getattr(bundle, "files", None) or {}
    path = files.get(QC_FILE)
    if path is None:
        return None
    frame = pd.read_parquet(path)
    frame = frame[frame["kept"]]
    return {str(f): float(s) for f, s in zip(frame["feature"], frame["qc_sd"]) if np.isfinite(s)}


def design_d_ratio(state: Any, inputs: Sequence[str], qc: Mapping[str, float] | None
                   ) -> dict[str, Any] | None:
    """``{columns, qc_sd, threshold}`` for the in-fold D-ratio filter, or None."""
    found = _applied(state)
    if found is None or not qc:
        return None
    _, option, p = found
    threshold = p.get("d_ratio_max")
    if option not in RLSC_OPTIONS or not threshold:
        return None
    present = set(inputs)
    columns = [c for c in qc if c in present]
    if not columns:
        return None
    return {"columns": columns, "qc_sd": {c: float(qc[c]) for c in columns},
            "threshold": float(threshold)}


def describe_d_ratio(spec: Mapping[str, Any]) -> tuple[str, str]:
    n = len(spec.get("columns") or [])
    return ("Filter by D-ratio", f"Removes, of {n:,} features, those whose pooled-QC standard "
            f"deviation is {float(spec.get('threshold') or D_RATIO_MAX):.0%} or more of the training "
            f"fold's own ({BROADHURST}); the QC side was fixed before the seal.")


def _register() -> None:
    from turbotab.core.decisions import register_validator
    from turbotab.core.methods.contract import (CONTRACTS, ContractOption, MethodContract, Relation,
                                                register_contract)
    from turbotab.core.voice import register_repair_sentence

    # The finding is answered by the reference-rows family (WP18), which offers these options
    # beside its exclusion (``reference_rows._offer``).
    register_validator("apply_repair", _qc_answer_fits)
    register_validator("apply_repair", _qc_pqn_needs_the_qcs)

    def sentence(d: Any, state: Any, ctx: Any) -> str:
        params = dict(d.params or {})
        if d.option not in RLSC_OPTIONS:
            from turbotab.core.reference_rows import exclusion_sentence

            return exclusion_sentence(d, ctx)
        return rlsc_sentence(params).rstrip(".")

    register_repair_sentence(FINDING, sentence)
    if "qc_rlsc" in CONTRACTS:
        return
    both = ("prediction", "inference")

    def opt(key: str, label: str, customary: str, sound: str, rung: str, order: int) -> ContractOption:
        return ContractOption(key, label, customary, dict.fromkeys(both, sound),
                              dict.fromkeys(both, rung), dict.fromkeys(both, order))

    pre = ("Before the seal, on the pooled QCs only: technical replicates, never a participant or "
           "the outcome.")
    register_contract(MethodContract(
        key="qc_detection_filter", label="QC detection-rate filter", slot="repairs",
        scope="reference_rows", run_order=1.0, needs=("pooled-QC injections",),
        question="(stated with QC-RLSC)",
        options=(opt("detection_70", "Detected in ≥ 70% of QCs", f"Customary ({BROADHURST})", pre,
                     "recommended", 0),),
        storyboard=("Count each feature's detections among the QC injections",
                    "Remove features below 70%"),
        relations=(), sources=(BROADHURST,)))
    register_contract(MethodContract(
        key="qc_rlsc", label="QC-RLSC drift and batch correction", slot="repairs",
        scope="reference_rows", run_order=2.0,
        needs=("pooled-QC injections", "an injection order", "at least five QCs per batch"),
        question="Pooled QCs and an injection order are here: correct drift from them?",
        options=(
            opt("qc_rlsc_lc", "Drift; LC-MS filters", f"Customary for large LC/GC-MS studies ({DUNN})",
                pre, "recommended", 0),
            opt("qc_rlsc_gc", "Drift; GC-MS filters", f"Customary for large LC/GC-MS studies ({DUNN})",
                pre, "recommended", 1),
            opt(ASIDE, "Exclude the QC rows", "Customary when drift is negligible",
                "Leaves drift in the values.", "available", 2),
        ),
        storyboard=("Fit a LOESS to each feature's QCs over injection order, batch by batch",
                    "Divide every injection by the curve", "Rescale to the median QC value",
                    "Set the QC rows aside"),
        relations=(
            Relation("implies", "qc_rows_leave",
                     "QC drift correction implies that the QC rows leave the cohort."),
            Relation("precedes", "omics_normalization",
                     "QC drift correction precedes every in-fold normalization."),
            Relation("precedes", "detection_limit",
                     "QC drift correction precedes every in-fold step."),
        ),
        sources=(DUNN, BROADHURST),
        clause=lambda run: ("drift was corrected per batch by QC-RLSC fitted to pooled QCs only, "
                            "which were then removed") if run.get("qc_rlsc") else None,
    ))
    register_contract(MethodContract(
        key="qc_rsd_filter", label="QC-RSD filter", slot="repairs", scope="reference_rows",
        run_order=3.0, needs=("pooled-QC injections",), question="(stated with QC-RLSC)",
        options=(opt("rsd", "RSD < 20% (LC-MS) or 30% (GC-MS)", f"Customary ({BROADHURST})", pre,
                     "recommended", 0),),
        storyboard=("Compute each feature's RSD over the corrected QCs", "Remove features above it"),
        relations=(), sources=(BROADHURST,)))
    register_contract(MethodContract(
        key="qc_pqn", label="PQN against the pooled-QC reference", slot="repairs",
        scope="reference_rows", run_order=4.0, needs=("pooled-QC injections",),
        question="(an omics normalization option)",
        options=(opt("qc_reference", "Pooled-QC reference", "pmp's default (the QC samples)", pre,
                     "available", 0),),
        storyboard=("Median of the integral-normalized QC injections", "Divide each injection by its "
                    "median quotient to it"),
        relations=(Relation("precedes", "log_transform", "The quotient normalization precedes the log."),),
        sources=("Dieterle et al. 2006", "pmp (Bioconductor)")))
    register_contract(MethodContract(
        key="qc_rows_leave", label="The QC rows leave", slot="eligibility", scope="row_local",
        run_order=1.0, needs=("the pooled-QC label",), question="(implied by the pooled-QC answer)",
        options=(opt("leave", "Set aside", "Always: QCs are not participants",
                     "Sound: they are technical replicates.", "recommended", 0),),
        storyboard=("Mark each pooled-QC injection", "Set them aside before the seal"),
        relations=(), sources=(DUNN,)))
    register_contract(MethodContract(
        key="d_ratio_filter", label="D-ratio filter", slot="in_fold", scope="training_fold",
        run_order=0.5, needs=("pooled-QC standard deviations",),
        question="(an option of the pooled-QC answer)",
        options=(opt("d_ratio_50", "D-ratio < 50%", f"Customary ({BROADHURST})",
                     "Sound in each training fold: it reads the study samples' spread.",
                     "available", 0),),
        storyboard=("Divide each feature's QC SD by the training fold's", "Remove features at 50% or more"),
        relations=(Relation("conflicts", "before_the_seal",
                            "The D-ratio reads participants, so it cannot run before the seal.",
                            rung="refused"),),
        sources=(BROADHURST,),
        short="the D-ratio filter"))


_register()


__all__ = [
    "BROADHURST", "D_RATIO_MAX", "DEGREE", "DETECTION_MIN", "DRatioFilter", "DUNN", "DriftFit",
    "FINDING", "Injections", "MAX_GAP", "MIN_QC", "QCPlan", "QCResult", "QC_FILE", "RSD_MAX",
    "SPANS", "choose_spans", "d_ratio", "detection_rate", "dilution", "injections", "largest_gap",
    "loess", "qc_reference", "qc_rlsc", "qc_rsd", "qc_sd", "run_reference_rows", "smoother_rows",
    "sufficiency", "ASIDE", "OPTIONS", "QC_ROWS_FILE", "RLSC_OPTIONS", "correct_table",
    "describe_d_ratio", "design_d_ratio", "reference_plan", "rlsc_offer", "rlsc_sentence",
    "working_qc_sd",
]

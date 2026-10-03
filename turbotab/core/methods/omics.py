"""Omics values: what scale they are on, and the normalizations a linear model needs, fit in-fold
(docs/turbotab-next/audit/AUDIT_REPORT.md §5 WP11; closes ME-09).

Sixty samples, 300 genes, no gene differing between cases and controls, and the cases sequenced
1.46× deeper: an elastic net on the raw counts "predicted" the outcome with a cross-validated AUC
of 0.83 to 1.0, because a deeper library has larger values in every gene. Urine 25% more dilute in
cases did the same to raw metabolite intensities. Normalizing removes it (AUDIT_REPORT ME-09).

The reading — label-free, before the split
-----------------------------------------
:func:`scale_reading` reads a block of assay columns under the genomics or metabolomics lens from
the values alone; it never looks at the outcome:

* **counts** — whole numbers, none negative, reaching at least :data:`COUNT_MAX` (genotype dosages
  0/1/2 and scores do not), under the genomics lens. A row total is the sample's library size.
* **intensities** — none negative, spanning :data:`RAW_RANGE`-fold, and either reaching
  :data:`RAW_MAX` (log-scale values sit below ~40) or right-skewed feature by feature (median
  skewness at least :data:`RAW_SKEW`; a log scale is near-symmetric), under the metabolomics lens.
  A row total is the sample's total signal, which dilution scales.

The reading raises the app's own finding ``omics_scale`` with its repair options (the ``omics_scale``
repair family below): a normalization that runs inside every training fold, or the researcher's
statement that the values were already normalized. Until one is recorded, a family whose model is a
weighted sum of the values (``linear_in_values``) is refused (:func:`_linear_families_need_a_scale`)
and the design stage refuses as a backstop.

The normalizations — fit on training rows, applied to every row
--------------------------------------------------------------
* :class:`LogCPM` — edgeR's ``cpm(y, log = TRUE, prior.count = 2)`` on library sizes scaled by
  edgeR's TMM factors (``calcNormFactors(method = "TMM")``, Robinson & Oshlack 2010), or by none.
  Fit chooses the TMM reference sample among the training rows, exactly as ``calcNormFactors``
  does, and keeps it; a held-out row's factor is computed against that same reference and divided
  by the training factors' geometric mean, so on the training rows the factors are edgeR's own.
* :class:`QuotientLog` — probabilistic quotient normalization (Dieterle et al. 2006) and/or log2:
  the reference spectrum is the feature-wise median of the training rows after integral
  normalization; each sample is divided by the median of its quotients to that reference (its
  most probable dilution); then log2. A zero cannot be logged: it becomes missing, and the option
  that does this says so.

Both are pipeline steps, so cross-validation refits them on each training fold and nothing learned
from data sees a held-out row. Hornung et al. (2015) found that normalizing the whole table before
cross-validation "did not result in a noteworthy optimistic bias", so the gain here is from
normalizing at all; fitting in-fold costs nothing and keeps the lockbox rule without exception.

The library-size check — training rows only
------------------------------------------
:func:`library_size_check` asks whether the per-sample totals — computed from the assay values
alone, without the outcome — differ with the outcome, on the training rows (never the held-out
ones): Mann–Whitney for two classes, Kruskal–Wallis for more, Spearman for a number. The shelf
states it as a concern on every family while the values are not normalized.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

OMICS_LENSES = ("genomics", "metabolomics")
FINDING = "omics_scale"
MIN_FEATURES = 20  # fewer assay-like columns than this is not a feature block
COUNT_MAX = 100.0  # whole numbers whose largest value is below this are not read as counts
RAW_MAX = 1_000.0  # raw intensities reach thousands; log2 intensities stay below ~40
RAW_RANGE = 100.0  # ... and span two orders of magnitude (largest over smallest positive value)
RAW_SKEW = 0.5  # or, below RAW_MAX (a targeted panel in µM), are right-skewed feature by feature
CHECK_ALPHA = 0.01  # the library-size check speaks below this p-value
PRIOR_COUNT = 2.0  # edgeR cpm(log = TRUE)'s default prior.count
LOGRATIO_TRIM, SUM_TRIM, A_CUTOFF = 0.3, 0.05, -1e10  # edgeR calcNormFactors' TMM defaults

COUNT_METHODS = ("log_cpm_tmm", "log_cpm", "declared_normalized")
INTENSITY_METHODS = ("pqn_log2", "log2", "declared_normalized")
DECLARED = "declared_normalized"
TRANSFORMS = ("log_cpm_tmm", "log_cpm", "pqn_log2", "log2")

# What each normalization is called in a design step and on a lineage link.
STEP_LABELS = {
    "log_cpm_tmm": ("Log-CPM with TMM factors", "log-CPM (TMM)"),
    "log_cpm": ("Log-CPM", "log-CPM"),
    "pqn_log2": ("Quotient normalization, then log2", "PQN, log2"),
    "log2": ("Log2", "log2"),
}


# ── the reading: label-free, from the values alone ──────────────────────────


@dataclass(frozen=True)
class ScaleReading:
    """A block of assay columns read as raw ``counts`` or raw ``intensities``."""

    kind: str
    columns: tuple[str, ...]
    totals: np.ndarray  # per row: the library size (counts) or total signal (intensities)
    maximum: float
    minimum_positive: float
    n_zero: int  # cells equal to zero (intensities: they cannot be logged)
    n_missing: int

    @property
    def spread(self) -> float:
        """Largest over smallest positive row total."""
        t = self.totals[np.isfinite(self.totals) & (self.totals > 0)]
        return float(t.max() / t.min()) if t.size else float("nan")


def _excluded_by_name(name: str) -> bool:
    """Identifiers and a person's characteristics are never assay features (the roles stage's own
    name rules, so the block and the roles agree)."""
    try:
        from turbotab.core.stages.rows import _COVARIATE_TOKENS, _TIME_TOKENS, _id_like, _tokens
    except ImportError:  # pragma: no cover - the roles module always exists in this app
        return False
    tokens = set(_tokens(str(name)))
    return _id_like(str(name)) or bool(tokens & _COVARIATE_TOKENS) or bool(tokens & _TIME_TOKENS)


def candidate_columns(frame: pd.DataFrame, target: str | None = None,
                      only: Sequence[str] | None = None) -> list[str]:
    """Numeric columns that could be assay features: not the outcome, not an identifier or a
    person's characteristic by name, not two-valued. ``only`` limits them (the exposures)."""
    allowed = None if only is None else set(only)
    out = []
    for name in frame.columns:
        c = str(name)
        if c == target or c.startswith("__") or (allowed is not None and c not in allowed):
            continue
        s = frame[name]
        if pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
            continue
        if allowed is None and _excluded_by_name(c):
            continue
        out.append(c)
    return out


def _block(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    return frame[list(columns)].to_numpy(dtype=float, na_value=np.nan)


def scale_reading(frame: pd.DataFrame, lenses: Sequence[str] | None, target: str | None = None,
                  only: Sequence[str] | None = None) -> ScaleReading | None:
    """The block of raw counts or raw intensities in ``frame``, or None.

    Reads values only: the outcome column is skipped, never compared. ``only`` restricts the
    candidates (the design stage passes the exposures).
    """
    lenses = [k for k in (lenses or []) if k in OMICS_LENSES]
    if not lenses:
        return None
    columns = candidate_columns(frame, target, only)
    if len(columns) < MIN_FEATURES:
        return None
    values = _block(frame, columns)
    finite = np.isfinite(values)
    with np.errstate(invalid="ignore"):
        nonneg = np.where(finite, values >= 0, True).all(axis=0)
        whole = np.where(finite, np.equal(np.mod(values, 1), 0), True).all(axis=0)
        distinct = np.array([np.unique(values[finite[:, j], j]).size > 2 for j in range(len(columns))])
    keep = nonneg & distinct
    if "genomics" in lenses:
        counts = keep & whole
        if counts.sum() >= MIN_FEATURES:
            block = values[:, counts]
            if np.nanmax(block) >= COUNT_MAX:
                return _reading("counts", [c for c, k in zip(columns, counts) if k], block)
    if "metabolomics" in lenses and keep.sum() >= MIN_FEATURES:
        block = values[:, keep]
        positive = block[np.isfinite(block) & (block > 0)]
        if positive.size and float(positive.max()) / float(positive.min()) >= RAW_RANGE and (
                float(positive.max()) >= RAW_MAX or _median_skew(block) >= RAW_SKEW):
            return _reading("intensities", [c for c, k in zip(columns, keep) if k], block)
    return None


def _median_skew(block: np.ndarray) -> float:
    """The median over features of each one's sample skewness: unlogged concentrations are
    right-skewed (a log-normal feature with a 30% CV has skewness near 1); log-scale ones are not."""
    from scipy.stats import skew

    with np.errstate(all="ignore"):
        s = skew(block, axis=0, nan_policy="omit")
    s = np.asarray(s, dtype=float)
    s = s[np.isfinite(s)]
    return float(np.median(s)) if s.size else float("nan")


def _reading(kind: str, columns: list[str], block: np.ndarray) -> ScaleReading:
    finite = np.isfinite(block)
    positive = block[finite & (block > 0)]
    return ScaleReading(
        kind=kind, columns=tuple(columns), totals=np.nansum(block, axis=1),
        maximum=float(np.nanmax(block)), minimum_positive=float(positive.min()) if positive.size else 0.0,
        n_zero=int(np.sum(finite & (block == 0))), n_missing=int(np.sum(~finite)))


# ── edgeR's TMM and log-CPM, by their definitions ───────────────────────────


def _rank(values: np.ndarray) -> np.ndarray:
    """R's ``rank()``: ties share their average rank."""
    from scipy.stats import rankdata

    return rankdata(values, method="average")


def tmm_factor(obs: np.ndarray, ref: np.ndarray, lib_obs: float, lib_ref: float, *,
               logratio_trim: float = LOGRATIO_TRIM, sum_trim: float = SUM_TRIM,
               a_cutoff: float = A_CUTOFF) -> float:
    """edgeR's ``.calcFactorTMM`` (Robinson & Oshlack 2010) for one sample against the reference:
    the precision-weighted mean of the log ratios left after trimming 30% of them by M and 5% by A.
    """
    obs = np.asarray(obs, dtype=float)
    ref = np.asarray(ref, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_r = np.log2((obs / lib_obs) / (ref / lib_ref))
        abs_e = (np.log2(obs / lib_obs) + np.log2(ref / lib_ref)) / 2
        v = (lib_obs - obs) / lib_obs / obs + (lib_ref - ref) / lib_ref / ref
    fin = np.isfinite(log_r) & np.isfinite(abs_e) & (abs_e > a_cutoff)
    log_r, abs_e, v = log_r[fin], abs_e[fin], v[fin]
    if log_r.size == 0 or np.max(np.abs(log_r)) < 1e-6:
        return 1.0
    n = log_r.size
    lo_l = math.floor(n * logratio_trim) + 1
    hi_l = n + 1 - lo_l
    lo_s = math.floor(n * sum_trim) + 1
    hi_s = n + 1 - lo_s
    rl, ra = _rank(log_r), _rank(abs_e)
    keep = (rl >= lo_l) & (rl <= hi_l) & (ra >= lo_s) & (ra <= hi_s)
    # sum(logR/v, na.rm = TRUE) / sum(1/v, na.rm = TRUE); a missing result is 0, so a factor of 1
    with np.errstate(divide="ignore", invalid="ignore"):
        f = float(np.nansum(log_r[keep] / v[keep]) / np.nansum(1.0 / v[keep]))
    return float(2.0 ** f) if math.isfinite(f) else 1.0


def tmm_reference(counts: np.ndarray, lib: np.ndarray) -> int:
    """edgeR's reference sample: the one whose upper-quartile factor (75th percentile of its counts
    over its library size, genes that are zero everywhere left out) is nearest their mean; when
    the median of those factors is below 1e-20, the one with the largest sum of root counts."""
    present = (counts > 0).any(axis=0)
    x = counts[:, present]
    if x.shape[1] == 0:
        return 0
    f75 = np.quantile(x, 0.75, axis=1) / lib  # R's quantile type 7 is numpy's default
    if np.median(f75) < 1e-20:
        return int(np.argmax(np.sqrt(x).sum(axis=1)))
    return int(np.argmin(np.abs(f75 - f75.mean())))


def tmm_factors(counts: np.ndarray, lib: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """edgeR ``calcNormFactors(method = "TMM")`` for samples in rows: (factors, reference row).
    The factors multiply to one."""
    raw, ref, _ = _tmm_raw(counts, lib)
    return raw / math.exp(float(np.mean(np.log(raw)))), ref


def _tmm_raw(counts: np.ndarray, lib: np.ndarray | None = None) -> tuple[np.ndarray, int, np.ndarray]:
    """(each row's TMM factor against the reference before they are scaled to multiply to one,
    the reference row, the genes that are positive somewhere). One row, or no positive gene, is
    edgeR's degenerate case: every factor is 1."""
    counts = np.asarray(counts, dtype=float)
    lib = counts.sum(axis=1) if lib is None else np.asarray(lib, dtype=float)
    present = (counts > 0).any(axis=0)
    x = counts[:, present]
    if x.shape[1] == 0 or x.shape[0] == 1:
        return np.ones(counts.shape[0]), 0, present
    ref = tmm_reference(x, lib)
    raw = np.array([tmm_factor(x[i], x[ref], lib[i], lib[ref]) for i in range(x.shape[0])])
    return raw, ref, present


def log_cpm(counts: np.ndarray, lib: np.ndarray, *, prior_count: float = PRIOR_COUNT,
            mean_lib: float | None = None) -> np.ndarray:
    """edgeR ``cpm(y, lib.size = lib, log = TRUE, prior.count)`` for samples in rows.

    edgeR's C code (``compute_offsets``, ``calc_cpm_log``): each sample's prior is
    ``prior.count × lib / mean(lib)``, it is added to every count and twice to the library size,
    and the value is ``log2((y + prior) / (lib + 2·prior) × 10⁶)``. ``mean_lib`` fixes the mean
    (the training rows'), so a held-out row gets the prior the training rows' mean implies.
    """
    lib = np.asarray(lib, dtype=float)
    mean = float(np.mean(lib)) if mean_lib is None else float(mean_lib)
    prior = prior_count * lib / mean
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log2((counts + prior[:, None]) / (lib + 2.0 * prior)[:, None] * 1e6)


# ── the in-fold transformers ─────────────────────────────────────────────────


class _BlockStep(TransformerMixin, BaseEstimator):
    """A step that rewrites ``columns`` (the assay block) and passes every other column through,
    keeping names and the row-id index."""

    columns: Sequence[str]

    def _split(self, X: pd.DataFrame) -> tuple[list[str], np.ndarray]:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"{type(self).__name__} needs a pandas DataFrame with named columns.")
        cols = [c for c in self.columns if c in X.columns]
        return cols, X[cols].to_numpy(dtype=float, na_value=np.nan)

    def _out(self, X: pd.DataFrame, cols: list[str], values: np.ndarray) -> pd.DataFrame:
        out = X.copy()
        if cols:
            out[cols] = values
        return out

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray([str(c) for c in self.feature_names_in_], dtype=object)

    def _fit_names(self, X: pd.DataFrame) -> None:
        self.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)
        self.n_features_in_ = X.shape[1]

    def lineage(self) -> list[dict[str, Any]]:
        """What the lineage draws: each block column rewritten by this step, the rest kept."""
        block = set(self.columns)
        op = self.operation
        return [{"output": c, "inputs": [c], "operation": op if c in block else "kept"}
                for c in self.feature_names_in_]


class LogCPM(_BlockStep):
    """log2 counts per million (edgeR ``cpm(log = TRUE, prior.count = 2)``), on library sizes
    scaled by TMM factors when ``tmm``; the reference sample and the mean library size come from
    the rows it is fit on. A count that is missing stays missing and adds nothing to its row's
    library size."""

    def __init__(self, columns: Sequence[str] = (), tmm: bool = True, prior_count: float = PRIOR_COUNT):
        self.columns = columns
        self.tmm = tmm
        self.prior_count = prior_count

    @property
    def operation(self) -> str:
        return STEP_LABELS["log_cpm_tmm" if self.tmm else "log_cpm"][1]

    def fit(self, X: pd.DataFrame, y: Any = None) -> "LogCPM":
        self._fit_names(X)
        cols, counts = self._split(X)
        counts = np.where(np.isfinite(counts), counts, 0.0)
        lib = counts.sum(axis=1)
        self.n_samples_fit_ = int(len(lib))
        if self.tmm and counts.shape[1]:
            raw, ref, present = _tmm_raw(counts, lib)
            self.ref_counts_ = counts[ref].copy()
            self.ref_lib_ = float(lib[ref])
            self.present_ = present
            self.degenerate_ = bool(present.sum() == 0 or len(lib) == 1)
            self.factor_scale_ = math.exp(float(np.mean(np.log(raw))))  # factors multiply to one
            effective = lib * raw / self.factor_scale_
        else:
            effective = lib
        self.mean_lib_ = float(np.mean(effective)) if len(effective) else 1.0
        return self

    def factors(self, counts: np.ndarray) -> np.ndarray:
        """Each row's TMM factor against the fitted reference, on the training rows' scale."""
        lib = counts.sum(axis=1)
        if not self.tmm or getattr(self, "degenerate_", False) or not hasattr(self, "present_"):
            return np.ones(len(lib))
        present = self.present_
        raw = np.array([tmm_factor(counts[i][present], self.ref_counts_[present], lib[i], self.ref_lib_)
                        if lib[i] > 0 else 1.0 for i in range(len(lib))])
        return raw / self.factor_scale_

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "mean_lib_"):
            raise ValueError("LogCPM is not fitted yet.")
        cols, counts = self._split(X)
        if not cols:
            return X.copy()
        missing = ~np.isfinite(counts)
        filled = np.where(missing, 0.0, counts)
        effective = filled.sum(axis=1) * self.factors(filled)
        out = log_cpm(filled, effective, prior_count=self.prior_count, mean_lib=self.mean_lib_)
        out[missing] = np.nan
        out[effective <= 0] = np.nan
        return self._out(X, cols, out)


class QuotientLog(_BlockStep):
    """Probabilistic quotient normalization (``quotient``) and/or log2 (``log``).

    Dieterle et al. (2006), in Kohl et al.'s (2011) summary: PQN "starts, with an integral
    normalization of each spectrum, followed by the calculation of a reference spectrum such as a
    median spectrum. Next, for each variable of interest the quotient of a given test spectrum and
    reference spectrum is calculated and the median of all quotients is estimated. Finally, all
    variables of the test spectrum are divided by the median quotient." Here the reference is the
    feature-wise median of the training rows, each first scaled to the training rows' median total
    (the integral normalization's constant: Dieterle used 100; any constant only shifts every log2
    value alike, and this one keeps the values in their own units). Only positive values enter a
    total, the reference or a quotient. The integral step cancels from a sample's own result, which
    is its values divided by the median of their quotients to the reference; it shapes the
    reference. After that, log2: a zero cannot be logged and becomes missing.
    """

    def __init__(self, columns: Sequence[str] = (), quotient: bool = True, log: bool = True):
        self.columns = columns
        self.quotient = quotient
        self.log = log

    @property
    def operation(self) -> str:
        return STEP_LABELS["pqn_log2" if self.quotient else "log2"][1]

    def fit(self, X: pd.DataFrame, y: Any = None) -> "QuotientLog":
        import warnings

        self._fit_names(X)
        _, values = self._split(X)
        positive = np.where(np.isfinite(values) & (values > 0), values, np.nan)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            totals = np.nansum(positive, axis=1)
            ok = np.isfinite(totals) & (totals > 0)
            self.integral_ = float(np.median(totals[ok])) if ok.any() else 1.0
            scaled = positive[ok] * (self.integral_ / totals[ok])[:, None]
            self.reference_ = (np.nanmedian(scaled, axis=0) if scaled.shape[0]
                               else np.full(values.shape[1], np.nan))
        return self

    def dilution(self, values: np.ndarray) -> np.ndarray:
        """Each row's most probable dilution factor against the fitted reference (1 when unknown)."""
        if not self.quotient:
            return np.ones(values.shape[0])
        ref = self.reference_
        with np.errstate(all="ignore"):
            import warnings

            q = np.where(np.isfinite(values) & (values > 0) & np.isfinite(ref) & (ref > 0),
                         values / ref, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                factor = np.nanmedian(q, axis=1) if q.shape[1] else np.full(q.shape[0], np.nan)
        return np.where(np.isfinite(factor) & (factor > 0), factor, 1.0)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "reference_"):
            raise ValueError("QuotientLog is not fitted yet.")
        cols, values = self._split(X)
        if not cols:
            return X.copy()
        out = values / self.dilution(values)[:, None]
        if self.log:
            with np.errstate(divide="ignore", invalid="ignore"):
                out = np.where(out > 0, np.log2(np.where(out > 0, out, 1.0)), np.nan)
        return self._out(X, cols, out)


def normalizer(spec: Mapping[str, Any]) -> Any:
    """The pipeline step for a normalization spec ``{method, columns}``; None when it changes nothing."""
    method, columns = str(spec.get("method")), list(spec.get("columns") or [])
    if not columns or method not in TRANSFORMS:
        return None
    if method in ("log_cpm_tmm", "log_cpm"):
        return LogCPM(columns, tmm=method == "log_cpm_tmm")
    return QuotientLog(columns, quotient=method == "pqn_log2", log=True)


def describe(spec: Mapping[str, Any]) -> tuple[str, str] | None:
    """(label, detail) of the normalization step in the app's voice."""
    method, n = str(spec.get("method")), len(spec.get("columns") or [])
    if method not in TRANSFORMS or not n:
        return None
    what = f"{n:,} assay column{'s' if n != 1 else ''}"
    detail = {
        "log_cpm_tmm": (f"Turns {what} into log2 counts per million, each sample's library size "
                        f"(its total over these columns) scaled by TMM factors against a reference "
                        f"sample chosen within each training fold (edgeR, prior count 2)."),
        "log_cpm": (f"Turns {what} into log2 counts per million on each sample's own library size, "
                    f"its total over these columns (edgeR, prior count 2, the mean library size "
                    f"from each training fold)."),
        "pqn_log2": (f"Divides each sample's {what} by its most probable dilution, the median "
                     f"quotient to a reference spectrum (the median of the training fold's "
                     f"integral-normalized samples), then takes log2; a zero becomes missing."),
        "log2": f"Takes log2 of {what}; a zero becomes missing.",
    }[method]
    return STEP_LABELS[method][0], detail


# ── the finding and its repair family ───────────────────────────────────────


def _tick(value: Any) -> str:
    return f"`{value}`"


def _count(n: int) -> str:
    return _tick(f"{int(n):,}")


def _num(x: float) -> str:
    if not math.isfinite(x):
        return "?"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:,.3g}"


def scale_finding(frame: pd.DataFrame, lenses: Sequence[str] | None,
                  target: str | None = None) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """The app's own finding for raw counts or raw intensities, as ``(raw, finding)``."""
    reading = scale_reading(frame, lenses, target)
    if reading is None:
        return None
    n = len(reading.columns)
    totals = reading.totals[np.isfinite(reading.totals) & (reading.totals > 0)]
    lo, hi = (float(totals.min()), float(totals.max())) if totals.size else (float("nan"),) * 2
    spread = reading.spread
    if reading.kind == "counts":
        title = f"{_count(n)} columns hold raw counts, and library sizes differ by sample."
        detail = (f"{_count(n)} columns are whole numbers from 0 to {_tick(_num(reading.maximum))}. "
                  f"Each sample's total over them, its library size, runs from {_tick(_num(lo))} to "
                  f"{_tick(_num(hi))}, a {spread:.1f}-fold spread. On raw counts a sample sequenced "
                  f"deeper has larger values in every gene.")
        why = ("A model fit to raw counts can read sequencing depth as biology: in a test with no "
               "gene differing, cases sequenced 1.46 times deeper gave a cross-validated AUC near "
               "1. Linear models wait until a normalization is chosen or you state the values are "
               "already normalized.")
        summary = (f"{_count(n)} columns are raw counts; library sizes vary {spread:.1f}-fold, so "
                   f"depth would read as biology.")
    else:
        title = f"{_count(n)} columns hold raw intensities, and total signal differs by sample."
        zeros = (f" {_count(reading.n_zero)} values are zero, which a log turns into missing."
                 if reading.n_zero else "")
        detail = (f"{_count(n)} columns run from {_tick(_num(reading.minimum_positive))} to "
                  f"{_tick(_num(reading.maximum))}, unlogged. Each sample's total signal runs from "
                  f"{_tick(_num(lo))} to {_tick(_num(hi))}, a {spread:.1f}-fold spread that dilution "
                  f"alone can make.{zeros}")
        why = ("A more dilute sample has smaller values in every feature, so a model fit to raw "
               "intensities can read dilution as biology, and the raw scale lets a few abundant "
               "features dominate. Linear models wait until a normalization is chosen or you state "
               "the values are already normalized.")
        summary = (f"{_count(n)} columns are raw intensities; total signal varies {spread:.1f}-fold, "
                   f"so dilution would read as biology.")
    finding = {
        "id": FINDING, "severity": "warning", "title": title, "detail": detail,
        "why_it_matters": why, "affected_columns": list(reading.columns), "source": "structural",
        "lens": "genomics" if reading.kind == "counts" else "metabolomics", "evidence": None,
        "summary": summary, "routes_to": "models", "lever_label": "Normalize before modeling",
        "group": None,
    }
    params = {"kind": reading.kind, "columns": list(reading.columns), "n_zero": reading.n_zero,
              "spread": round(spread, 4) if math.isfinite(spread) else None}
    return {"params": params, "confidence": "high"}, finding


def _offer(finding: dict[str, Any], p: dict[str, Any], oc: Any) -> list[Any]:
    from turbotab.core.repairs import RepairOption
    from turbotab.core.decisions import ApplyRepair
    from turbotab.core.voice import finish

    columns = [c for c in p.get("columns") or [] if oc.has(c)]
    if len(columns) < MIN_FEATURES:
        return []
    kind = str(p.get("kind"))
    n = len(columns)
    zeros = int(p.get("n_zero") or 0)
    params = {"kind": kind, "columns": columns, "n_zero": zeros}
    exposures = "those of them the models take as exposures"

    def option(key: str, label: str, consequence: str, sentence: str, in_fold: bool) -> Any:
        return RepairOption(
            key=key, label=finish(label, terminal=False), consequence=finish(consequence),
            row_local=not in_fold, effect="values", sentence=finish(sentence),
            decision=ApplyRepair(finding_id=str(finding["id"]), option=key, params=params))

    declared = option(
        DECLARED, "Already normalized",
        "Values enter as they are; you state they were normalized before this file.",
        f"The researcher stated that the {n:,} assay columns were already normalized; they enter "
        f"the models as they are.", in_fold=False)
    if kind == "counts":
        return [
            option("log_cpm_tmm", "Log-CPM with TMM",
                   "TMM factors from each training fold scale library sizes; values become log2 counts per million.",
                   f"Of the {n:,} count columns, {exposures} were transformed to log2 counts per "
                   f"million (edgeR cpm, prior count 2) on library sizes scaled by TMM factors "
                   f"(Robinson and Oshlack 2010), fit within each training fold.", in_fold=True),
            option("log_cpm", "Log-CPM, library size only",
                   "Each sample's own library size divides its counts, then log2; no TMM composition factors.",
                   f"Of the {n:,} count columns, {exposures} were transformed to log2 counts per "
                   f"million (edgeR cpm, prior count 2) on each sample's library size, the mean "
                   f"library size taken within each training fold.", in_fold=True),
            declared,
        ]
    zero_words = f"; {zeros:,} zeros become missing" if zeros else ""
    zero_sentence = (f" {zeros:,} zero values could not be logged and were set to missing."
                     if zeros else "")
    return [
        option("pqn_log2", "PQN, then log2",
               f"Each sample is divided by its dilution against a fold's median reference, then logged{zero_words}.",
               f"Of the {n:,} intensity columns, {exposures} were normalized by probabilistic "
               f"quotient normalization (Dieterle et al. 2006), the reference spectrum the median of "
               f"the integral-normalized training rows within each fold, then log2-transformed."
               f"{zero_sentence}",
               in_fold=True),
        option("log2", "Log2 only",
               f"Values become their log2; samples are not rescaled for dilution{zero_words}.",
               f"Of the {n:,} intensity columns, {exposures} were log2-transformed, with no "
               f"normalization for dilution.{zero_sentence}", in_fold=True),
        declared,
    ]


def _marks(option: str, params: Mapping[str, Any]) -> set[tuple[str, str]]:
    return {(str(c), f"scale:{option}") for c in params.get("columns") or []}


def _register_repairs() -> None:
    from turbotab.core.repairs import Family, register_family

    effects = {key: "values" for key in (*COUNT_METHODS, *INTENSITY_METHODS)}
    # No SQL: the transforms run inside the pipeline, fit on each training fold (``normalizer``).
    register_family(Family("omics_scale", 9, _offer, effects, marks=_marks), [FINDING])


def normalization_of(state: Any) -> dict[str, Any] | None:
    """The normalization the state records: ``{method, kind, columns, finding_id}`` from the applied
    ``omics_scale`` repair, else None."""
    from turbotab.core.stages.finding_words import family

    found = getattr(state, "findings", None) or {}
    for fid, d in dict(found).items():
        if family(str(fid)) != FINDING:
            continue
        action = getattr(d, "action", None) if not isinstance(d, Mapping) else d.get("action")
        option = getattr(d, "option", None) if not isinstance(d, Mapping) else d.get("option")
        params = (getattr(d, "params", None) if not isinstance(d, Mapping) else d.get("params")) or {}
        if action == "applied" and option in (*COUNT_METHODS, *INTENSITY_METHODS):
            return {"method": str(option), "kind": params.get("kind"),
                    "columns": [str(c) for c in params.get("columns") or []], "finding_id": str(fid),
                    "n_zero": int(params.get("n_zero") or 0)}
    return None


def design_normalization(state: Any, inputs: Sequence[str]) -> dict[str, Any] | None:
    """The normalization the pipeline runs: the recorded method on the recorded columns that are
    exposures among ``inputs`` (a covariate such as age is never normalized as a gene)."""
    found = normalization_of(state)
    if found is None:
        return None
    # The settled exposures only (BLUEPRINT §14.1): an exposure role that rode along unconfirmed
    # normalizes nothing.
    from turbotab.core.readings import settled_roles

    roles = settled_roles(state)
    present = set(inputs)
    columns = [c for c in found["columns"] if c in present and roles.get(c) == "exposure"]
    return {"method": found["method"], "kind": found["kind"], "columns": columns}


# ── the library-size check: training rows only ─────────────────────────────


def library_size_check(totals: np.ndarray, y: Any, task: str, kind: str = "counts",
                       target: str = "the outcome", event: Any = None,
                       rows: str = "training rows") -> dict[str, Any] | None:
    """Whether the per-sample totals differ with the outcome; a dict with ``test``, ``statistic``,
    ``p``, ``flagged`` and the ``sentence`` to state, or None when it cannot be computed.

    ``totals`` come from the assay values alone. Callers pass training rows only under
    prediction; under inference every analyzed row (BLUEPRINT §12 ruling 3), and ``rows`` names
    them in the sentence.
    """
    from scipy import stats

    t = np.asarray(totals, dtype=float)
    y = np.asarray(y)
    ok = np.isfinite(t) & pd.notna(pd.Series(y)).to_numpy()
    t, y = t[ok], y[ok]
    if len(t) < 6:
        return None
    what = "Library size" if kind == "counts" else "Total signal"
    reason = ("depth alone could predict the outcome on raw counts" if kind == "counts"
              else "dilution alone could predict the outcome on raw intensities")
    out: dict[str, Any] = {"kind": kind, "n": int(len(t))}
    if task == "regression":
        rho, p = stats.spearmanr(t, y.astype(float))
        out.update(test="spearman", statistic=float(rho), p=float(p))
        said = f"Spearman ρ = {rho:+.2f}"
    else:
        levels = list(pd.unique(pd.Series(y)))
        groups = [t[y == level] for level in levels]
        if len(groups) < 2 or min(len(g) for g in groups) < 3:
            return None
        if len(groups) == 2:
            # The fit stage codes the named event 1 (``coded_outcome``); otherwise the named level,
            # else the last one. ``event`` names it in the sentence.
            coded = {str(v) for v in levels} <= {"0", "1", "0.0", "1.0", "True", "False"}
            if coded:
                first = next(v for v in levels if str(v) in ("1", "1.0", "True"))
            else:
                first = next((v for v in levels if str(v) == str(event)), levels[-1])
            a = t[y == first]
            b = t[y != first]
            res = stats.mannwhitneyu(a, b, alternative="two-sided", method="auto")
            auc = float(res.statistic) / (len(a) * len(b))  # P(event's total > the other's)
            ratio = float(np.median(a) / np.median(b)) if np.median(b) > 0 else float("nan")
            name = str(event) if coded and event is not None else str(first)
            out.update(test="mann_whitney", statistic=float(res.statistic), p=float(res.pvalue),
                       auc=auc, ratio=ratio, event=name)
            said = (f"median {ratio:.2f}× as large in `{name}`, AUC {auc:.2f}"
                    if math.isfinite(ratio) else f"AUC {auc:.2f}")
        else:
            h, p = stats.kruskal(*groups)
            out.update(test="kruskal_wallis", statistic=float(h), p=float(p))
            said = f"Kruskal–Wallis H = {h:.1f}"
    from turbotab.core.models.inference import format_p

    out["flagged"] = bool(out["p"] < CHECK_ALPHA)
    out["sentence"] = (f"{what} tracks `{target}` on the {rows} ({said}, p = "
                       f"{format_p(out['p'])}): {reason}.")
    return out


def check_on(frame: pd.DataFrame, state: Any, y: Any, task: str,
             rows: str = "training rows") -> dict[str, Any] | None:
    """:func:`library_size_check` over the exposures that read as raw counts or intensities in
    ``frame`` (``rows``: the training rows, or every analyzed row under inference); None when none
    do, or when a normalization is recorded."""
    roles = getattr(state, "roles", None) or {}
    exposures = [c for c in frame.columns if roles.get(c) == "exposure"]
    reading = scale_reading(frame, getattr(state, "lens", None), getattr(state, "target", None),
                            only=exposures)
    if reading is None:
        return None
    found = normalization_of(state)
    if found is not None and found["method"] in TRANSFORMS:
        return None  # the totals are divided out of the values the models see
    return library_size_check(reading.totals, y, task, reading.kind,
                              target=str(getattr(state, "target", None) or "the outcome"),
                              event=getattr(state, "event", None), rows=rows)


# ── the refusal: linear families wait for a scale ────────────────────────────


def _ctx(ctx: Any, name: str) -> Any:
    if ctx is None:
        return None
    return ctx.get(name) if isinstance(ctx, Mapping) else getattr(ctx, name, None)


def _scale_finding_of(ctx: Any) -> dict[str, Any] | None:
    reader = _ctx(ctx, "artifact")
    if not callable(reader):
        return None
    try:
        artifact = reader("findings")
    except Exception:  # noqa: BLE001 - an unreadable artifact checks nothing
        return None
    data = getattr(artifact, "data", artifact)
    if not isinstance(data, Mapping):
        return None
    from turbotab.core.stages.finding_words import family

    return next((dict(f) for f in data.get("findings") or []
                 if isinstance(f, Mapping) and family(str(f.get("id"))) == FINDING), None)


def linear_in_values(key: str) -> bool:
    from turbotab.core.models import get_family

    try:
        return bool(getattr(get_family(key), "linear_in_values", False))
    except KeyError:
        return False


def refusal_for(models: Sequence[str], state: Any, finding: Mapping[str, Any]) -> Any:
    """The Refusal a selection of ``models`` meets while ``finding`` stands unanswered, or None."""
    from turbotab.core.decisions import Refusal, SelectModels

    if normalization_of(state) is not None:
        return None
    linear = [m for m in models if linear_in_values(m)]
    if not linear:
        return None
    from turbotab.core.models import get_family

    names = [get_family(m).label for m in linear]
    listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    options = list(finding.get("repairs") or [])
    kind = next((str(((o.get("decision") or {}).get("params") or {}).get("kind")) for o in options),
                "counts")
    what = "raw counts" if kind == "counts" else "raw intensities"
    exits = [{"label": o["label"], "decision": o["decision"]} for o in options]
    others = [m for m in models if m not in linear]
    if others:
        exits.append({"label": "Keep only the families that are not linear",
                      "decision": SelectModels(models=others)})
    exits.append({"label": "Choose from the shelf", "decision": None})
    verb = "fits" if len(names) == 1 else "fit"
    return Refusal(
        "raw_assay_values",
        f"{listed} {verb} a weighted sum of the values as they are, and these are {what}: "
        f"sequencing depth or dilution would read as biology. Choose a normalization, or state "
        f"that the values are already normalized.",
        exits=exits)


def zeros_refusal(models: Sequence[str], state: Any) -> Any:
    """The Refusal a selection meets when the recorded normalization logs values that include
    zeros, nothing fills missing values, and some of ``models`` cannot fit a missing value."""
    from turbotab.core.decisions import Refusal, SelectModels
    from turbotab.core.models import get_family

    found = normalization_of(state)
    if found is None or found["method"] not in LOGGED or not found["n_zero"] or _fills_missing(state):
        return None
    known = []
    for m in models:
        try:
            known.append(get_family(m))
        except KeyError:
            continue
    unable = [f for f in known if not getattr(f, "handles_missing", False)]
    if not unable:
        return None
    able = [f.key for f in known if getattr(f, "handles_missing", False)]
    exits: list[dict[str, Any]] = [
        {"label": "Settle the zeros first, on the finding about values below detection",
         "decision": None}]
    if able:
        exits.append({"label": "Keep only the families that handle missing values",
                      "decision": SelectModels(models=able)})
    exits.append({"label": "Choose from the shelf", "decision": None})
    return Refusal("zeros_cannot_be_logged",
                   zeros_message(found["n_zero"], [f.label for f in unable]), exits=exits)


def _linear_families_need_a_scale(decision: Any, ctx: Any) -> None:
    state = _ctx(ctx, "state")
    if state is None:
        return
    finding = _scale_finding_of(ctx)
    if finding is None:
        return
    refusal = refusal_for(decision.models, state, finding) or zeros_refusal(decision.models, state)
    if refusal is not None:
        raise refusal


LOGGED = ("pqn_log2", "log2")  # the normalizations that take a plain log, so a zero cannot pass


def _fills_missing(state: Any) -> bool:
    from turbotab.core.decisions import missing_strategy

    return missing_strategy(state) in ("impute", "multiple_imputation")


def zeros_message(n_zero: int, names: Sequence[str]) -> str:
    listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    return (f"{n_zero:,} zero values in the assay columns cannot be logged, so log2 makes them "
            f"missing inside the pipeline, after complete cases were counted; {listed} cannot fit "
            f"rows with a missing value. Settle the zeros first, as values below detection or as "
            f"true zeros, or keep only families that handle missing values.")


def design_refusal(state: Any, frame: pd.DataFrame, families: Sequence[Any]) -> str | None:
    """The design stage's backstop: a message when a linear family would see raw assay values, or
    when a log would turn zeros into missing values that nothing fills and a family cannot take.
    ``frame`` holds the training rows."""
    found = normalization_of(state)
    if found is not None:
        if found["method"] not in LOGGED or _fills_missing(state):
            return None
        unable = [f for f in families if not getattr(f, "handles_missing", False)]
        columns = [c for c in found["columns"] if c in frame.columns]
        if not unable or not columns:
            return None
        n_zero = int((frame[columns].to_numpy(dtype=float, na_value=np.nan) == 0).sum())
        return zeros_message(n_zero, [f.label for f in unable]) if n_zero else None
    linear = [f for f in families if getattr(f, "linear_in_values", False)]
    if not linear:
        return None
    roles = getattr(state, "roles", None) or {}
    exposures = [c for c in frame.columns if roles.get(c) == "exposure"]
    reading = scale_reading(frame, getattr(state, "lens", None), getattr(state, "target", None),
                            only=exposures)
    if reading is None:
        return None
    names = ", ".join(f.label for f in linear)
    return (f"{len(reading.columns):,} exposures are raw {reading.kind} and no normalization was "
            f"chosen: {names} would read depth or dilution as biology. Choose one on the finding "
            f"about them, or state that the values are already normalized.")


# ── purpose-specific coaching for raw counts (replaces "Do not pre-normalize") ──

RAW_COUNTS_COACHING = (
    "Raw counts keep what a count model needs: DESeq2 and edgeR estimate measurement precision "
    "from them and correct for library size themselves, so a count model takes the counts as they "
    "are. The models in this app read the values directly, so here the counts need a "
    "normalization first. For inference, the feature-wise tests run on log counts per million "
    "with TMM factors, one linear model per gene as limma fits, without its variance moderation. "
    "For prediction, the same normalization is refit within each training fold. Whether to "
    "normalize matters far more than where: for microarray data, Hornung and colleagues (2015) "
    "found that normalizing the whole table before cross-validation \"did not result in a "
    "noteworthy optimistic bias\".")


def restate_raw_counts(finding: dict[str, Any]) -> dict[str, Any]:
    """The genomics data-type finding with its raw-counts coaching made purpose-specific, in place.

    The pack's text ends "Do not pre-normalize these.": DESeq2's advice for its own input, which
    this app's models are not (AUDIT_REPORT ME-09). The pack module is shared with Classic, so the
    Next app restates it here rather than editing it.
    """
    try:
        from turbotab import packs

        original = packs.COACHING[packs.RAW_COUNTS]
    except Exception:  # noqa: BLE001 - nothing to restate without the pack
        return finding
    for key in ("detail", "why_it_matters"):
        text = finding.get(key)
        if not isinstance(text, str):
            continue
        if original in text:
            finding[key] = text.replace(original, RAW_COUNTS_COACHING)
        elif "Do not pre-normalize these." in text:
            head, tail = text.split("Do not pre-normalize these.", 1)
            finding[key] = (head.rstrip() + " " + RAW_COUNTS_COACHING + tail).strip()
    return finding


def _register() -> None:
    from turbotab.core.decisions import register_validator

    _register_repairs()
    register_validator("select_models", _linear_families_need_a_scale)


_register()

__all__ = [
    "CHECK_ALPHA", "COUNT_METHODS", "DECLARED", "FINDING", "INTENSITY_METHODS", "LogCPM",
    "OMICS_LENSES", "QuotientLog", "RAW_COUNTS_COACHING", "ScaleReading", "TRANSFORMS",
    "candidate_columns", "check_on", "describe", "design_normalization", "design_refusal",
    "library_size_check", "log_cpm", "normalization_of", "normalizer", "refusal_for",
    "restate_raw_counts", "scale_finding", "scale_reading", "tmm_factor", "tmm_factors",
    "tmm_reference", "zeros_message", "zeros_refusal",
]

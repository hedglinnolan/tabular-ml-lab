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
# ``qc_pqn_log2``: PQN against the pooled-QC reference before the seal (reference rows,
# ``methods.qc_drift``), then log2 in-fold (MODELING_SEQUENCE §1.1, MS7).
INTENSITY_METHODS = ("pqn_log2", "qc_pqn_log2", "log2", "declared_normalized")
DECLARED = "declared_normalized"
TRANSFORMS = ("log_cpm_tmm", "log_cpm", "pqn_log2", "qc_pqn_log2", "log2")

# What each normalization is called in a design step and on a lineage link.
STEP_LABELS = {
    "log_cpm_tmm": ("Log-CPM with TMM factors", "log-CPM (TMM)"),
    "log_cpm": ("Log-CPM", "log-CPM"),
    "pqn_log2": ("Quotient normalization, then log2", "PQN, log2"),
    "qc_pqn_log2": ("Log2, after PQN to the pooled QCs", "log2"),
    "log2": ("Log2", "log2"),
}
# When values below detection are filled between the normalization and the log (MS7), the two
# halves are their own steps: the quotient normalization, then (after the fill) the log.
PQN_LABELS = ("Quotient normalization", "PQN")
LOG_LABELS = ("Log2", "log2")


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
    person's characteristic by name, not two-valued, and not how the samples were acquired: a
    column named as a run order, batch, plate or well, or one that numbers the rows once each (a
    run order read name-blind) is never an intensity (MS7 repair: a `run_order` column was counted
    among the intensities). ``only`` limits them (the exposures)."""
    from turbotab.core.recognizers import acquisition_kind

    allowed = None if only is None else set(only)
    out = []
    for name in frame.columns:
        c = str(name)
        if c == target or c.startswith("__") or (allowed is not None and c not in allowed):
            continue
        s = frame[name]
        if pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
            continue
        if allowed is None and (_excluded_by_name(c) or acquisition_kind(c) is not None
                                or _numbers_the_rows(s)):
            continue
        out.append(c)
    return out


def _numbers_the_rows(s: pd.Series) -> bool:
    """Whole numbers that are a permutation of 0…n−1 or 1…n: a row numbering, never a reading."""
    if not len(s) or s.isna().any():
        return False
    v = s.to_numpy(dtype=float)
    first = float(v.min())
    if first not in (0.0, 1.0) or float(v.max()) != first + len(v) - 1:
        return False
    return bool(np.array_equal(np.sort(v), np.arange(first, first + len(v))))


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


def normalizer(spec: Mapping[str, Any], *, log: bool = True) -> Any:
    """The pipeline step for a normalization spec ``{method, columns}``; None when it changes nothing.

    ``log=False`` gives the quotient normalization alone (None for a method with nothing to fit
    before its log), for a design that fills values below detection between the two (MS7)."""
    method, columns = str(spec.get("method")), list(spec.get("columns") or [])
    if not columns or method not in TRANSFORMS:
        return None
    if method in ("log_cpm_tmm", "log_cpm"):
        return LogCPM(columns, tmm=method == "log_cpm_tmm")
    if not log:
        return QuotientLog(columns, quotient=True, log=False) if method == "pqn_log2" else None
    # qc_pqn_log2: the quotient normalization ran before the seal, against the pooled QCs.
    return QuotientLog(columns, quotient=method == "pqn_log2", log=True)


def logger(spec: Mapping[str, Any]) -> Any:
    """The log2 step that follows the fill of values below detection (MS7), or None."""
    method, columns = str(spec.get("method")), list(spec.get("columns") or [])
    if not columns or method not in LOGGED:
        return None
    return QuotientLog(columns, quotient=False, log=True)


def splits_for_detection(spec: Mapping[str, Any] | None, censored: Mapping[str, Any] | None) -> bool:
    """Whether the normalization's log waits for the fill of values below detection: a logged
    intensity normalization whose columns include a left-censored one (MODELING_SEQUENCE §1.1:
    normalization, then detection-limit handling, then the log)."""
    if not spec or str(spec.get("method")) not in LOGGED or not censored:
        return False
    return bool(set(spec.get("columns") or []) & set(censored.get("columns") or []))


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
                     f"integral-normalized samples), then takes log2."),
        "qc_pqn_log2": (f"Takes log2 of {what}, each injection already divided by its dilution "
                        f"against the pooled QCs' reference spectrum before the seal."),
        "log2": f"Takes log2 of {what}.",
    }[method]
    return STEP_LABELS[method][0], detail


def describe_split(spec: Mapping[str, Any], part: str) -> tuple[str, str]:
    """(label, detail) of the quotient normalization or the log, when a fill sits between them."""
    n = len(spec.get("columns") or [])
    what = f"{n:,} assay column{'s' if n != 1 else ''}"
    if part == "normalize":
        return PQN_LABELS[0], (f"Divides each sample's {what} by its most probable dilution, the "
                               f"median quotient to a reference spectrum (the median of the training "
                               f"fold's integral-normalized samples).")
    return LOG_LABELS[0], (f"Takes log2 of {what}, after values below detection are filled, so no "
                           f"zero or blank reaches the log.")


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
    if reading.kind == "intensities" and reading.n_zero:
        # Which columns hold the zeros a log cannot take (MS7: each routes to the detection-limit
        # question, never to a median fill).
        block = _block(frame, reading.columns)
        hits = np.sum(np.isfinite(block) & (block == 0), axis=0)
        params["zero_columns"] = {c: int(k) for c, k in zip(reading.columns, hits) if k}
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
    exposures = "those of them the models take as study factors"

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
    zero_columns = dict(p.get("zero_columns") or {})
    if zero_columns:
        params["zero_columns"] = {c: int(k) for c, k in zero_columns.items() if oc.has(c)}
    # A zero cannot be logged (MS7): it is answered on the finding about zeros first, as a value
    # below detection, and then filled by the detection-limit answer before the log.
    zero_words = (f"; first say what its {zeros:,} zeros are (below detection?)" if zeros else "")
    options = [
        option("pqn_log2", "PQN, then log2",
               f"Each sample is divided by its dilution against a fold's median reference, then logged{zero_words}.",
               f"Of the {n:,} intensity columns, {exposures} were normalized by probabilistic "
               f"quotient normalization (Dieterle et al. 2006), the reference spectrum the median of "
               f"the integral-normalized training rows within each fold, then log2-transformed.",
               in_fold=True),
    ]
    if _has_pooled_qcs(oc.frame):
        options.append(option(
            "qc_pqn_log2", "PQN to pooled QCs",
            f"Each injection is divided by its dilution against the pooled QCs' reference, then "
            f"logged{zero_words}.",
            f"Of the {n:,} intensity columns, {exposures} were normalized by probabilistic quotient "
            f"normalization against the pooled-QC reference spectrum (the median of the "
            f"integral-normalized QC injections) before the seal, then log2-transformed.",
            in_fold=False))
    options += [
        option("log2", "Log2 only",
               f"Values become their log2; samples are not rescaled for dilution{zero_words}.",
               f"Of the {n:,} intensity columns, {exposures} were log2-transformed, with no "
               f"normalization for dilution.", in_fold=True),
        declared,
    ]
    return options


def _has_pooled_qcs(frame: pd.DataFrame) -> bool:
    """Whether the table holds pooled-QC injections (the metabolomics pack's own reading)."""
    try:
        from turbotab import packs

        return packs._pooled_qc(frame) is not None
    except Exception:  # noqa: BLE001 - no pack, no pooled-QC option
        return False


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
                    "n_zero": int(params.get("n_zero") or 0),
                    "zero_columns": {str(c): int(k) for c, k in
                                     dict(params.get("zero_columns") or {}).items()}}
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


def unanswered_zeros(state: Any, frame: pd.DataFrame | None = None) -> dict[str, int]:
    """The logged normalization's columns whose zeros nobody has said the meaning of: ``{column:
    zeros}``. A zero the user recoded as a non-detection (the ``zeros_nondetect`` repair) is a
    blank below the detection limit by then, answered. ``frame`` (the design's rows) counts them
    there; without it the recorded finding's counts are read."""
    found = normalization_of(state)
    if found is None or found["method"] not in LOGGED:
        return {}
    from turbotab.core.repairs import nondetect_columns

    answered = set(nondetect_columns(state))
    if frame is not None:
        cols = [c for c in found["columns"] if c in frame.columns and c not in answered]
        if not cols:
            return {}
        hits = (frame[cols].to_numpy(dtype=float, na_value=np.nan) == 0).sum(axis=0)
        return {c: int(k) for c, k in zip(cols, hits) if k}
    per_column = dict(found.get("zero_columns") or {})
    if not per_column and found["n_zero"]:
        # A record from before the per-column counts: its zeros are unanswered unless every
        # normalized column's zeros were recoded.
        return {} if set(found["columns"]) <= answered else {"": int(found["n_zero"])}
    return {c: int(k) for c, k in per_column.items() if c not in answered and k}


def _nondetect_decision(findings: Any) -> Any:
    """The "zeros mean not detected" repair a finding offers here, or None."""
    data = getattr(findings, "data", findings)
    items = (data or {}).get("findings") if isinstance(data, Mapping) else None
    for f in items or []:
        for o in f.get("repairs") or []:
            if o.get("key") == "nondetect":
                return o.get("decision")
    return None


def _declared_decision(found: Mapping[str, Any]) -> Any:
    from turbotab.core.decisions import ApplyRepair

    return ApplyRepair(finding_id=str(found["finding_id"]), option=DECLARED,
                       params={"kind": found.get("kind"), "columns": list(found["columns"]),
                               "n_zero": int(found.get("n_zero") or 0)})


def zeros_exits(state: Any, findings: Any = None) -> list[dict[str, Any]]:
    """Where a zero a log would turn missing goes: the detection-limit question (its repair, when
    a finding offers it), or no log at all."""
    found = normalization_of(state)
    exits: list[dict[str, Any]] = [
        {"label": "Zeros mean not detected: fill them as values below detection",
         "decision": _nondetect_decision(findings)}]
    if found is not None:
        exits.append({"label": "Keep the values unlogged: they were already normalized",
                      "decision": _declared_decision(found)})
    return exits


def zeros_refusal(models: Sequence[str], state: Any, findings: Any = None) -> Any:
    """The Refusal a selection meets while the recorded normalization would log zeros nobody has
    answered (MS7). Whatever the families and whatever fills missing values: a zero turned missing
    would otherwise be median-filled in the middle of the distribution, or read as missing at
    random, when it is almost always a value below detection. The exit is the detection-limit
    question; ``models`` do not change the answer."""
    from turbotab.core.decisions import Refusal

    zeros = unanswered_zeros(state)
    if not zeros:
        return None
    return Refusal("zeros_cannot_be_logged",
                   zeros_message(sum(zeros.values()), len([c for c in zeros if c])),
                   exits=zeros_exits(state, findings))


def _linear_families_need_a_scale(decision: Any, ctx: Any) -> None:
    state = _ctx(ctx, "state")
    if state is None:
        return
    finding = _scale_finding_of(ctx)
    if finding is None:
        return
    refusal = (refusal_for(decision.models, state, finding)
               or zeros_refusal(decision.models, state, _findings_artifact(ctx)))
    if refusal is not None:
        raise refusal


def _findings_artifact(ctx: Any) -> Any:
    reader = _ctx(ctx, "artifact")
    if not callable(reader):
        return None
    try:
        return reader("findings")
    except Exception:  # noqa: BLE001 - an unreadable artifact offers no repair
        return None


def _logged_options_wait_for_the_zeros(decision: Any, ctx: Any) -> None:
    """Applying a logged normalization while its columns hold zeros nobody has answered is
    refused, with the detection-limit question as the way forward (MS7)."""
    from turbotab.core.decisions import Refusal
    from turbotab.core.stages.finding_words import family

    if family(str(decision.finding_id)) != FINDING or decision.option not in LOGGED:
        return
    state = _ctx(ctx, "state")
    params = dict(decision.params or {})
    if not params:
        found = _scale_finding_of(ctx) or {}
        same = [o for o in found.get("repairs") or [] if o.get("key") == decision.option]
        params = dict(same[0]["decision"]["params"]) if same else {}
    from turbotab.core.repairs import nondetect_columns

    answered = set(nondetect_columns(state)) if state is not None else set()
    zeros = {c: k for c, k in dict(params.get("zero_columns") or {}).items() if c not in answered}
    if not zeros:
        return
    findings = _findings_artifact(ctx)
    exits = [{"label": "Zeros mean not detected: fill them as values below detection",
              "decision": _nondetect_decision(findings)},
             {"label": "Keep the values unlogged: they were already normalized",
              "decision": {"kind": "apply_repair", "finding_id": decision.finding_id,
                           "option": DECLARED,
                           "params": {k: params[k] for k in ("kind", "columns", "n_zero") if k in params}}}]
    raise Refusal("zeros_cannot_be_logged", zeros_message(sum(zeros.values()), len(zeros)),
                  exits=exits)


LOGGED = ("pqn_log2", "qc_pqn_log2", "log2")  # the normalizations that take a plain log


def zeros_message(n_zero: int, n_columns: int = 0) -> str:
    where = (f" in {n_columns:,} assay column{'s' if n_columns != 1 else ''}" if n_columns
             else " in the assay columns")
    return (f"{n_zero:,} zero values{where} cannot be logged: a log would make them missing, and a "
            f"value below detection must never be filled as if it were missing at random (a median "
            f"fill puts it in the middle of the distribution). Say what the zeros are first: values "
            f"below detection, which the detection-limit answer then fills before the log, or keep "
            f"the values unlogged.")


def design_refusal(state: Any, frame: pd.DataFrame, families: Sequence[Any]) -> str | None:
    """The design stage's backstop: a message when a linear family would see raw assay values, or
    when a log would meet zeros nobody has answered (MS7: they route to the detection-limit
    question, never to a fill for missing values). ``frame`` holds the training rows."""
    found = normalization_of(state)
    if found is not None:
        if found["method"] not in LOGGED:
            return None
        zeros = unanswered_zeros(state, frame)
        return zeros_message(sum(zeros.values()), len(zeros)) if zeros else None
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
    return (f"{len(reading.columns):,} study factors are raw {reading.kind} and no normalization was "
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


# ── values below detection: after the normalization, before the log (MS7) ────
#
# MODELING_SEQUENCE §1.1: "PQN with a study-sample reference … detection-limit handling,
# censoring-aware: half-minimum is customary at ≤ 10% censored; above 10%, QRILC or a
# censored-normal draw … log or glog". Lubin et al. (2004, Environ Health Perspect 112:1691):
# assigning one-half the detection limit "can be biased unless the percentage of measurements below
# detection limits is small (5-10%)". Wei et al. (2018, Sci Rep 8:663): "QRILC was the favored one
# for left-censored MNAR".

CENSORED_CUSTOMARY_MAX = 0.10  # half-minimum's customary range (Lubin et al. 2004: 5–10%)
QRILC_UPPER = 0.99  # imputeLCMD's ``upper.q``
LUBIN = "Lubin et al. 2004"
WEI = "Wei et al. 2018"
LAZAR = "Lazar et al. 2016"


def qrilc_parameters(observed_log: np.ndarray, p_missing: float) -> tuple[float, float]:
    """(mean, sd) of the complete-data normal that imputeLCMD's ``impute.QRILC`` fits to one
    sample's log values: the observed quantiles at ``seq(0.001, 0.991, 0.01)`` regressed by least
    squares on the standard normal quantiles at ``seq(p + 0.001, 0.991, (0.99 − p) / 99)``, ``p``
    the sample's share below detection. Only the upper ``1 − p`` of the normal is observed, which is
    what shifting the normal quantiles by ``p`` says."""
    from scipy import stats

    k = np.arange(100)
    probs_normal = p_missing + 0.001 + k * (QRILC_UPPER - p_missing) / (QRILC_UPPER * 100)
    probs_sample = 0.001 + 0.01 * k
    q_normal = stats.norm.ppf(probs_normal)
    q_sample = np.quantile(observed_log, probs_sample)  # R's quantile type 7
    slope, intercept = np.polyfit(q_normal, q_sample, 1)
    return float(intercept), float(slope)


class QRILCFill(TransformerMixin, BaseEstimator):
    """Quantile regression imputation of left-censored data (QRILC; Lazar et al. 2016, the
    ``imputeLCMD`` package), one sample at a time on the log scale.

    For each row: its detected values among ``columns`` are logged; ``p`` is the share of its
    ``censored`` columns' blanks among all of them; :func:`qrilc_parameters` gives the mean and
    standard deviation of the complete-data normal; each blank is drawn from that normal truncated
    above at its ``p + 0.001`` quantile, and returned on the original scale. A row's fill reads only
    the row. imputeLCMD passes the standard deviation where ``rtmvnorm`` takes a variance; this
    draws with the fitted standard deviation, as the method is described. Draws are seeded by the
    row's id, so a row is filled the same way every time.

    A sample QRILC cannot read — fewer than five detected values, 99% or more below detection, or a
    degenerate fit — still has values below detection, never blanks for a median: each becomes half
    its column's smallest detected value on the fitting rows (half-minimum, the customary rung), and
    ``fallback_`` counts them. Those rows alone read the fitting rows (training-fold scope); every
    other row's fill reads only itself.
    """

    MIN_DETECTED = 5

    def __init__(self, columns: Sequence[str] = (), censored: Sequence[str] = (), seed: int = 0):
        self.columns = columns
        self.censored = censored
        self.seed = seed

    def fit(self, X: pd.DataFrame, y: Any = None) -> "QRILCFill":
        self.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)
        self.n_features_in_ = X.shape[1]
        self.half_minimum_: dict[str, float] = {}
        for c in self.censored:
            if c in X.columns:
                x = pd.to_numeric(X[c], errors="coerce").to_numpy(dtype=float)
                detected = x[np.isfinite(x) & (x > 0)]
                if detected.size:
                    self.half_minimum_[str(c)] = float(detected.min()) / 2
        self.fallback_ = 0
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        from scipy import stats

        cols = [c for c in self.columns if c in X.columns]
        censored = [c for c in self.censored if c in X.columns]
        if not cols or not censored:
            return X.copy()
        out = X.copy()
        values = X[cols].to_numpy(dtype=float, na_value=np.nan)
        is_censored = np.array([c in set(censored) for c in cols])
        for i, row_id in enumerate(X.index):
            v = values[i]
            blank = ~np.isfinite(v) & is_censored
            observed = np.isfinite(v) & (v > 0)
            n_used = int(blank.sum() + observed.sum())
            if not blank.any() or observed.sum() < self.MIN_DETECTED or n_used == 0:
                continue
            p = blank.sum() / n_used
            if p >= QRILC_UPPER:
                continue
            mu, sd = qrilc_parameters(np.log(v[observed]), p)
            if not (math.isfinite(mu) and math.isfinite(sd) and sd > 0):
                continue
            upper = stats.norm.ppf(p + 0.001)  # the truncation point, in SDs from the mean
            rng = np.random.default_rng([int(self.seed), _row_seed(row_id)])
            draws = stats.truncnorm.rvs(-np.inf, upper, loc=mu, scale=sd, size=int(blank.sum()),
                                        random_state=rng)
            v = v.copy()
            v[blank] = np.exp(draws)
            values[i] = v
        # What QRILC could not read is still below detection: half the column's minimum.
        half = getattr(self, "half_minimum_", {})
        left = 0
        for j, c in enumerate(cols):
            if is_censored[j] and c in half:
                gap = ~np.isfinite(values[:, j])
                left += int(gap.sum())
                values[gap, j] = half[c]
        self.fallback_ = left
        out[cols] = values
        return out

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.feature_names_in_, dtype=object)

    def lineage(self) -> list[dict[str, Any]]:
        filled = set(self.censored)
        return [{"output": c, "inputs": [c],
                 "operation": ("below detection: QRILC draw (half the minimum for a sample with "
                               "too few detected values)") if c in filled else "kept"}
                for c in self.feature_names_in_]


class LogScaleCensoredFill(TransformerMixin, BaseEstimator):
    """The censoring-aware fill for columns the pipeline logs next (MS7): each blank becomes its
    expected value below the limit (the smallest detected value on the fitting rows) under a
    left-censored normal fitted to the column's logarithm, E[X | X < L] = exp(μ + σ²/2) Φ(a − σ) /
    Φ(a). The log is the scale the column is analyzed on (MODELING_SEQUENCE §1.1: "logged
    quantities are imputed on the log scale"), and a fit on the raw scale can put the expected
    value below zero, which the log would turn back into a blank. Every fill is positive. Fitted on
    the training fold, without the outcome; every other column passes through."""

    def __init__(self, columns: Sequence[str] = ()):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: Any = None) -> "LogScaleCensoredFill":
        from turbotab.core.methods.imputation import tobit_fit
        from turbotab.core.methods.missing import below_limit_mean

        self.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)
        self.n_features_in_ = X.shape[1]
        self.fill_: dict[str, float] = {}
        self.limit_: dict[str, float] = {}
        for c in self.columns:
            if c not in X.columns:
                continue
            x = pd.to_numeric(X[c], errors="coerce").to_numpy(dtype=float)
            detected = x[np.isfinite(x) & (x > 0)]
            if not len(detected):
                continue
            limit = float(detected.min())
            self.limit_[c] = limit
            blank = ~(np.isfinite(x) & (x > 0))
            if not blank.any() or len(detected) < 3 or np.ptp(detected) == 0:
                self.fill_[c] = limit / 2  # nothing to fit: half the limit, as half-minimum would
                continue
            L = math.log(limit)
            z = np.where(blank, L, np.log(np.where(blank, 1.0, x)))
            theta, _ = tobit_fit(np.ones((len(x), 1)), z, blank, L)
            self.fill_[c] = below_limit_mean(float(theta[0]), math.exp(float(theta[1])), L, True)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for c, v in getattr(self, "fill_", {}).items():
            if c in out.columns:
                col = pd.to_numeric(out[c], errors="coerce")
                out[c] = col.where(col.notna(), v)
        return out

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.feature_names_in_, dtype=object)

    def lineage(self) -> list[dict[str, Any]]:
        filled = set(getattr(self, "fill_", {}))
        return [{"output": c, "inputs": [c],
                 "operation": "below detection: expected value (log scale)" if c in filled else "kept"}
                for c in self.feature_names_in_]


def _row_seed(row_id: Any) -> int:
    try:
        return int(row_id) & 0x7FFFFFFF
    except (TypeError, ValueError):
        import zlib

        return zlib.crc32(str(row_id).encode("utf-8"))


def detection_limit_options(purpose: str | None, share: float | None) -> list[dict[str, Any]]:
    """The answers to "how are values below detection filled?", soundest first for ``purpose``
    at the censored ``share`` (the largest of the censored columns'), each with both labels and
    its rung (North star 5; §11.3)."""
    inference = purpose == "inference"
    high = share is not None and share > CENSORED_CUSTOMARY_MAX
    at = f"{share:.0%}" if share is not None else "an unknown share"
    aware = ("a censored-normal (Tobit) draw below the limit within the multiple imputation, given "
             "the outcome" if inference else
             "the expected value below the limit under a censored-normal fit, in each training fold "
             "without the outcome")
    half_rung = ("block_and_record" if inference else "rank_lower") if high else "available"
    half_sound = (f"Biased at {at} below the limit: {LUBIN} found half the limit biased unless 5–10% "
                  f"are below it, and every non-detect becomes one value." if high else
                  f"Sound enough at {at} below the limit ({LUBIN}: the bias is small at 5–10%).")
    options = [
        {"key": "censoring_aware", "label": "Censored-normal",
         "customary": f"Recommended for study factors below detection ({LUBIN})",
         "sound": f"Sound: {aware}.", "rung": "recommended"},
        {"key": "qrilc", "label": "QRILC",
         "customary": f"Customary in metabolomics for left-censored values ({WEI}; {LAZAR})",
         "sound": ("Refused under inference: one draw per value, outside the multiple imputation, "
                   "would drop its uncertainty from the intervals." if inference else
                   "Sound for left-censored values: each sample's blanks are drawn below its "
                   "detection quantile from the normal its detected values imply."),
         "rung": "refused" if inference else ("recommended" if high else "available")},
        {"key": "half_minimum", "label": "Half the smallest detected value",
         "customary": "Customary: MetaboAnalyst's default", "sound": half_sound, "rung": half_rung},
        {"key": "as_missing", "label": "As any other blank",
         "customary": "Not customary for non-detects",
         "sound": ("Refused under inference: values below a limit are not missing at random, and "
                   "filling them as if they were biases every association with them." if inference
                   else "Unsound: the median places non-detections in the middle of the "
                        "distribution; kept only with your reason."),
         "rung": "refused" if inference else "rank_lower"},
    ]
    order = {"censoring_aware": 0, "qrilc": 1 if high else 2, "half_minimum": 2 if high else 1,
             "as_missing": 3}
    return sorted(options, key=lambda o: order[o["key"]])


def censored_shares(column_info: Mapping[str, Any] | None, columns: Sequence[str],
                    n_rows: int | None, zeros: Mapping[str, int] | None = None) -> dict[str, float]:
    """Each censored column's share of non-detections from column summaries alone (no table at
    hand): its blanks, plus ``zeros`` (the zeros the user recoded as non-detections, counted by the
    finding that named them), over ``n_rows``. :func:`censored_shares_on` reads the table itself
    and is preferred: the summaries may predate the recoding, and count the pooled QCs."""
    out: dict[str, float] = {}
    if not column_info or not n_rows:
        return out
    for c in columns:
        info = column_info.get(c)
        if info is None:
            continue
        get = info.get if isinstance(info, Mapping) else (lambda k, i=info: getattr(i, k, None))
        missing = get("n_missing")
        if missing is not None:
            out[c] = min(1.0, (int(missing) + int((zeros or {}).get(c) or 0)) / int(n_rows))
    return out


def participant_rows(frame: pd.DataFrame, state: Any) -> np.ndarray:
    """Which of ``frame``'s rows are participants: every row no recorded reference-row answer takes
    out (the pooled QCs leave before any analysis, by the QC exclusion or QC-RLSC; WP18). A table
    the QC rows already left keeps every row."""
    from turbotab.core.methods.qc_drift import _label
    from turbotab.core.reference_rows import reference_rules

    keep = np.ones(len(frame), dtype=bool)
    for rule in reference_rules(getattr(state, "findings", None) or {}):
        column = str(rule["column"])
        if column not in frame.columns:
            continue
        levels = {_label(v) for v in rule.get("levels") or []}
        keep &= ~frame[column].map(lambda v: _label(v) in levels).to_numpy(dtype=bool)
    return keep


def censored_shares_on(frame: pd.DataFrame, columns: Sequence[str], state: Any) -> dict[str, float]:
    """Each censored column's share of non-detections among the participant rows
    (:func:`participant_rows`): its blanks, and its zeros where the user recoded them as
    non-detections (``zeros_nondetect``; a working table not yet recomputed still holds them as
    zeros). A property of the measurement, read without the outcome."""
    from turbotab.core.repairs import nondetect_columns

    keep = participant_rows(frame, state)
    n = int(keep.sum())
    if not n:
        return {}
    recoded = set(nondetect_columns(getattr(state, "findings", None) or {}))
    out: dict[str, float] = {}
    for c in columns:
        if c not in frame.columns:
            continue
        x = pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype=float)[keep]
        below = ~np.isfinite(x)
        if c in recoded:
            below |= x == 0
        out[c] = float(below.sum()) / n
    return out


def censored_share_of(ctx: Any, columns: Sequence[str]) -> float | None:
    """The largest censored share among ``columns`` (:func:`censored_shares_on` over the table the
    analysis reads, when the context can open it; else the column summaries), or None."""
    state = _ctx(ctx, "state")
    from turbotab.core.decisions import _store_of

    store = _store_of(ctx)
    shares: dict[str, float] = {}
    if store is not None:
        try:
            present = set(store.columns)
            rules = [str(r["column"]) for r in _reference_rules_of(state)]
            wanted = [c for c in dict.fromkeys([*columns, *rules]) if c in present]
            if wanted:
                shares = censored_shares_on(store.materialize(wanted), columns, state)
        except Exception:  # noqa: BLE001 - an unreadable table: the summaries stand in
            shares = {}
    if not shares:
        from turbotab.core.repairs import nondetect_columns

        recoded = set(nondetect_columns(getattr(state, "findings", None) or {}))
        norm = normalization_of(state) or {}
        zeros = {c: k for c, k in dict(norm.get("zero_columns") or {}).items() if c in recoded}
        shares = censored_shares(_ctx(ctx, "column_info"), columns, _n_rows_of(ctx), zeros)
    return max(shares.values()) if shares else None


def _reference_rules_of(state: Any) -> list[dict[str, Any]]:
    from turbotab.core.reference_rows import reference_rules

    return reference_rules(getattr(state, "findings", None) or {})


def _detection_limit_leash(decision: Any, ctx: Any) -> None:
    """The leash rows of MODELING_SEQUENCE §4 for values below detection, under inference:

    * MAR imputation of detection-limit blanks is **refused** (censoring-aware instead), whatever
      reason the answer gives — under prediction a reason keeps it, ranked lower;
    * QRILC is refused (one draw outside the multiple imputation);
    * half the minimum above 10% censored is **blocked and recorded**: kept only with the
      attestation (``acknowledged``) the methods sentence carries.
    """
    from turbotab.core.decisions import Refusal, _censored_named, _missing_base, _state

    state = _state(ctx)
    if state is None or getattr(state, "purpose", None) != "inference":
        return
    if decision.strategy == "complete_case" and not decision.below_detection:
        return
    roles = getattr(state, "roles", None) or {}
    gone = set(decision.drop_columns)
    from turbotab.core.models.pipeline import PREDICTOR_ROLES

    censored = [c for c in dict.fromkeys([*decision.censored_columns, *_censored_named(ctx)])
                if (not roles or roles.get(c) in PREDICTOR_ROLES) and c not in gone]
    if not censored:
        return
    # The share of non-detections among the participants, zeros recoded as non-detections
    # included (a share of blanks alone reads 0 when an export wrote non-detects as zeros).
    share = censored_share_of(ctx, censored)
    named = ", ".join(f"`{c}`" for c in censored[:6]) + (" and others" if len(censored) > 6 else "")
    aware = _missing_base(decision, below_detection="censoring_aware", censored_columns=censored)
    if decision.below_detection in (None, "as_missing") and decision.strategy != "complete_case":
        exits = [{"label": "Censored-normal: a draw below the limit within the multiple imputation",
                  "decision": aware}]
        if share is not None and share <= CENSORED_CUSTOMARY_MAX:
            exits.append({"label": "Half the smallest detected value (customary at this share)",
                          "decision": _missing_base(decision, below_detection="half_minimum",
                                                    censored_columns=censored)})
        raise Refusal(
            "mar_below_detection_under_inference",
            f"{named} hold values below a detection limit. Under inference they are refused as "
            f"missing at random, whatever the reason: a value below a limit is known to be small, "
            f"and imputing it from the observed values biases every association with it "
            f"({LUBIN}). Fill them censoring-aware.", exits=exits)
    if decision.below_detection == "qrilc":
        raise Refusal(
            "qrilc_under_inference",
            "QRILC fills each value below detection once, outside the multiple imputation, so the "
            "intervals would not carry that uncertainty. Under inference the censored-normal draw "
            "within the multiple imputation is the sound answer.",
            exits=[{"label": "Censored-normal: a draw below the limit within the multiple imputation",
                    "decision": aware}])
    if (decision.below_detection == "half_minimum" and share is not None
            and share > CENSORED_CUSTOMARY_MAX and not decision.acknowledged):
        raise Refusal(
            "half_minimum_above_ten_percent",
            f"Up to {share:.0%} of {named} lie below the detection limit. Half the smallest detected "
            f"value is customary only at 10% or less: beyond it the coefficient is biased "
            f"({LUBIN}). Blocked until it is recorded as a limitation, or filled censoring-aware.",
            exits=[{"label": "Censored-normal: a draw below the limit within the multiple imputation",
                    "decision": aware},
                   {"label": "Keep half the minimum, recorded as a limitation",
                    "decision": _missing_base(decision, acknowledged=True,
                                              censored_columns=censored)}])


def _n_rows_of(ctx: Any) -> int | None:
    """The rows the column summaries count: the context's own (the server reads both from the table
    the analysis reads), else the ingest's."""
    n = _ctx(ctx, "n_rows")
    if n is not None:
        return int(n)
    reader = _ctx(ctx, "artifact")
    if callable(reader):
        try:
            ingest = reader("ingest")
        except Exception:  # noqa: BLE001 - no ingest artifact, no share
            ingest = None
        data = getattr(ingest, "data", ingest)
        if isinstance(data, Mapping) and data.get("n_rows") is not None:
            return int(data["n_rows"])
    n = _ctx(ctx, "n_rows")
    return int(n) if n is not None else None


# ── in-fold screening at p ≫ n ───────────────────────────────────────────────


def sis_size(n_rows: int) -> int:
    """Sure independence screening's ``d = [n / log n]`` (Fan & Lv 2008, J R Stat Soc B 70:849)."""
    return max(1, int(math.floor(n_rows / math.log(n_rows)))) if n_rows > 2 else 1


class UnivariateScreen(TransformerMixin, BaseEstimator):
    """Keeps the ``columns`` with the largest absolute marginal correlation with the outcome on the
    rows it is fit on — sure independence screening (Fan & Lv 2008), ``d = [n / log n]`` of them
    unless ``keep`` says how many. Fit inside the pipeline, so cross-validation repeats it in every
    training fold, as Ambroise & McLachlan (2002) require of any selection; every column outside
    ``columns`` passes through."""

    def __init__(self, columns: Sequence[str] = (), keep: int | None = None):
        self.columns = columns
        self.keep = keep

    def fit(self, X: pd.DataFrame, y: Any = None) -> "UnivariateScreen":
        if y is None:
            raise ValueError("Screening reads the outcome of the rows it is fit on.")
        self.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)
        self.n_features_in_ = X.shape[1]
        cols = [c for c in self.columns if c in X.columns]
        target = pd.Series(np.asarray(y, dtype=object))
        yv = pd.to_numeric(target, errors="coerce").to_numpy(dtype=float)
        if np.isnan(yv).any():  # two labels as text: the last in sorted order is 1
            levels = sorted(target.dropna().unique(), key=str)
            yv = (target == levels[-1]).to_numpy(dtype=float)
        V = X[cols].to_numpy(dtype=float, na_value=np.nan)
        with np.errstate(all="ignore"):
            Vc = V - np.nanmean(V, axis=0)
            yc = yv - yv.mean()
            r = (np.nansum(Vc * yc[:, None], axis=0)
                 / np.sqrt(np.nansum(Vc ** 2, axis=0) * np.sum(yc ** 2)))
        r = np.where(np.isfinite(r), np.abs(r), -1.0)
        d = int(self.keep) if self.keep else sis_size(len(X))
        order = np.argsort(-r, kind="stable")[:min(d, len(cols))]
        self.kept_ = [cols[j] for j in sorted(order)]
        self.dropped_ = [c for c in cols if c not in set(self.kept_)]
        self.size_ = len(self.kept_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "dropped_"):
            raise ValueError("UnivariateScreen is not fitted yet.")
        return X.drop(columns=[c for c in self.dropped_ if c in X.columns])

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        gone = set(self.dropped_)
        return np.asarray([c for c in self.feature_names_in_ if c not in gone], dtype=object)

    def lineage(self) -> list[dict[str, Any]]:
        gone = set(self.dropped_)
        screened = set(self.columns)
        return [{"output": c, "inputs": [c], "operation": "kept by screening" if c in screened else "kept"}
                for c in self.feature_names_in_ if c not in gone]


def _register_screened_family() -> None:
    from turbotab.core.models.base import Assessment, register_family
    from turbotab.core.models.elastic_net import ElasticNet

    class ScreenedElasticNet(ElasticNet):
        """An elastic net after sure independence screening, both inside every training fold."""

        key = "screened_elastic_net"
        label = "Screened elastic net"
        tasks = ("regression", "binary")  # the screen correlates with a number or a 0/1 event
        purposes = ("prediction",)
        inductive_bias = ("Only features with a strong marginal association survive; among them, "
                          "straight-line effects shrunk toward zero.")
        strengths = (
            "Cuts tens of thousands of features to n / log n before the penalty, inside each fold.",
            "Tunes its penalty by cross-validation on training rows.",
        )
        cautions = (
            "A feature that matters only jointly with others can be screened out.",
            "Which features survive changes from fold to fold.",
        )
        # MODEL_FAMILY_CONTRACT §1: the elastic net's declarations, but for prediction only, so it
        # gives no table under inference, on its two tasks, and reviewed by the omics lenses.
        inference_decl = None
        raw_scale = {"regression": "value", "binary": "margin"}
        review_lenses = ("metabolomics", "genomics")
        consequence = ("Keeps the features most tied to the outcome in each training fold, then an "
                       "elastic net.")

        def methods_label(self, task: Any) -> str:
            return "screened elastic net"

        def preprocess(self, spec: Any) -> list[tuple[str, Any]]:
            screened = [c for c in spec.predictors if spec.roles.get(c) == "exposure"]
            return [("screen", UnivariateScreen(screened))]

        def describe(self, task: Any, purpose: Any) -> tuple[str, str]:
            label = ("Screened elastic net regression" if task == "regression"
                     else "Screened penalized logistic regression")
            return label, ("Keeps the n / log n study factors most correlated with the outcome on each "
                           "training fold (sure independence screening, Fan & Lv 2008), then chooses "
                           "the penalty by an inner cross-validation within those rows.")

        def describe_step(self, name: str) -> tuple[str, str] | None:
            if name == "screen":
                return ("Screen the features", "Keeps the n / log n study factors with the largest "
                        "absolute correlation with the outcome, recomputed on each training fold; "
                        "every other column passes.")
            return None

        def assess(self, s: Any) -> Assessment:
            if s.purpose == "inference":
                return Assessment(0.0, "poor", ("Screening chooses the reported features from the "
                                                "outcome: under inference that is selection, refused "
                                                "for the reported model.",))
            base = super().assess(s)
            if s.n_features >= s.n_rows:
                return Assessment(min(base.score, 3.8), base.fit, base.concerns)
            return Assessment(1.5, "fair", ("With fewer features than rows the penalty alone can "
                                            "select; screening adds instability.",))

    try:
        from turbotab.core.models import get_family

        get_family("screened_elastic_net")
    except KeyError:
        register_family(ScreenedElasticNet())


# ── an exposure family and its multiplicity control ───────────────────────────


MULTIPLICITY_SMALL = 10  # tests a "few prespecified hypotheses" may number (the app's convention)
ROTHMAN = "Rothman 1990"
BH = "Benjamini & Hochberg 1995"


def multiplicity_policy(state: Any) -> dict[str, Any]:
    """The recorded multiplicity method, or the one an exposure family implies: Benjamini–Hochberg
    (MODELING_SEQUENCE §2: "An exposure family implies multiplicity control: BH q-values for
    feature-wise analyses")."""
    spec = getattr(state, "multiplicity", None)
    if spec is None:
        # ESTIMAND: an exposure family declares its method with the estimand (``set_estimand``),
        # which also writes this slot; a state built without that fold still reads it there.
        from turbotab.core.decisions import ESTIMAND_MULTIPLICITY

        est = getattr(state, "estimand", None)
        get = (est.get if isinstance(est, Mapping) else
               (lambda k, d=None: getattr(est, k, d)))
        if est is not None and get("family") and get("multiplicity"):
            return {"method": ESTIMAND_MULTIPLICITY[str(get("multiplicity"))],
                    "acknowledged": bool(get("multiplicity_acknowledged", False)), "recorded": True}
        return {"method": "bh", "acknowledged": False, "recorded": False}
    get = spec.get if isinstance(spec, Mapping) else (lambda k, d=None: getattr(spec, k, d))
    return {"method": str(get("method")), "acknowledged": bool(get("acknowledged", False)),
            "recorded": True}


def multiplicity_rung(method: str, n_tests: int | None, omics: bool = False) -> str:
    """BH is recommended; the number of tests stated (no adjustment) is customary for a few
    prespecified hypotheses (Rothman 1990) and blocked and recorded beyond them, or for an omics
    family at any size (METABOLOMICS_PACK §08: its absence "is a fatal flaw in review"); no control
    at all is blocked and recorded."""
    if method == "bh":
        return "recommended"
    if (method == "stated_count" and not omics and n_tests is not None
            and n_tests <= MULTIPLICITY_SMALL):
        return "available"
    return "block_and_record"


def omics_family(state: Any) -> bool:
    """Whether an exposure family is an omics one: the study's lens is metabolomics or genomics."""
    lens = getattr(state, "lens", None) if state is not None else None
    return bool(set(lens or []) & set(OMICS_LENSES))


def multiplicity_options(n_tests: int | None) -> list[dict[str, Any]]:
    many = n_tests is None or n_tests > MULTIPLICITY_SMALL
    return [
        {"key": "bh", "label": "Benjamini–Hochberg q-values",
         "customary": "Customary for metabolome- and genome-wide association studies",
         "sound": ("Sound: holds the expected share of false discoveries among the reported ones "
                   f"({BH})."), "rung": "recommended"},
        {"key": "stated_count", "label": "Unadjusted, the number of tests stated",
         "customary": f"Customary for a few prespecified nutrient hypotheses ({ROTHMAN})",
         "sound": ("Not for a feature-wise family: among hundreds of tests many will pass p < 0.05 "
                   "by chance." if many else "Sound for a few prespecified hypotheses, when every "
                   "result is shown and the number of tests is stated."),
         "rung": multiplicity_rung("stated_count", n_tests)},
        {"key": "none", "label": "No multiplicity control",
         "customary": "Not customary for a family of study factors",
         "sound": "Unsound: a table of unadjusted tests reads chance findings as discoveries.",
         "rung": "block_and_record"},
    ]


def apply_multiplicity(table: Any, policy: Mapping[str, Any]) -> Any:
    """The feature-wise table under the recorded multiplicity method: BH q-values (the family's
    own), or, when the method is the number of tests stated or none (recorded with its
    attestation), no q column and the caption saying so. Every member is still shown."""
    method = str(policy.get("method") or "bh")
    if method == "bh":
        return table
    m = sum(1 for r in table.rows if r.get("p") is not None)
    for r in table.rows:
        r["q"] = None
    caption = str(table.info.get("caption") or "")
    import re

    # The family's discovery count, wherever it sits: the fit adds "Estimated from all n analyzed
    # rows." after it (MS7 repair: a regex anchored at the end left it in place).
    if method == "stated_count":
        said = (f"Unadjusted p-values for {m:,} tests, stated as such ({ROTHMAN}); no "
                f"multiplicity adjustment.")
    else:
        said = (f"No multiplicity control, recorded as a limitation: {m:,} tests at p < 0.05 "
                f"would give about {0.05 * m:,.0f} false positives by chance alone.")
    caption, n = re.subn(r"Benjamini–Hochberg q < [0-9.]+: [0-9,]+ of [0-9,]+\.", said, caption)
    if not n:
        caption = f"{caption} {said}".strip()
    table.info["caption"] = caption
    table.concerns[:] = [c for c in table.concerns if "false-discovery threshold" not in c]
    return table


def multiplicity_sentence(method: str, acknowledged: bool = False) -> str:
    if method == "bh":
        return (f"The exposures are tested as one family with Benjamini–Hochberg q-values ({BH}); "
                f"every member is shown")
    if method == "stated_count":
        text = (f"The exposures' p-values are unadjusted, with the number of tests stated "
                f"({ROTHMAN}); every member is shown")
        return text + ("; recorded as a limitation" if acknowledged else "")
    return ("The exposures' tests carry no multiplicity control, recorded as a limitation: chance "
            "findings cannot be told from discoveries; every member is shown")


def _multiplicity_leash(decision: Any, ctx: Any) -> None:
    """An exposure family without multiplicity control is blocked and recorded (MODELING_SEQUENCE
    §2, MS7): kept only with the attestation the methods sentence carries."""
    from turbotab.core.decisions import Refusal, _state

    state = _state(ctx)
    roles = (getattr(state, "roles", None) or {}) if state is not None else {}
    n_tests = sum(1 for r in roles.values() if r == "exposure") or None
    omics = omics_family(state)
    if (multiplicity_rung(decision.method, n_tests, omics) != "block_and_record"
            or decision.acknowledged):
        return
    what = (f"Unadjusted p-values for {'an omics' if omics else 'a'} family of "
            f"{n_tests:,} tests" if decision.method == "stated_count" and n_tests else
            "A family of study factors without multiplicity control")
    raise Refusal(
        "family_without_multiplicity",
        f"{what} reads chance findings as discoveries: at p < 0.05 about one test in twenty passes "
        f"with nothing there. Benjamini–Hochberg q-values hold the false-discovery rate; this is "
        f"blocked until it is recorded as a limitation.",
        exits=[{"label": "Benjamini–Hochberg q-values",
                "decision": {"kind": "set_multiplicity", "method": "bh"}},
               {"label": "Keep it, recorded as a limitation",
                "decision": {"kind": "set_multiplicity", "method": decision.method,
                             "acknowledged": True}}])


# ── zeros a log cannot take, when the pack's own finding is silent ───────────


ZEROS_FINDING = "omics_zeros"


def zeros_finding(frame: pd.DataFrame, lenses: Sequence[str] | None, target: str | None,
                  present: Sequence[str] = ()) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """The app's own finding for zeros among raw intensities when the pack's zeros finding does not
    fire (fewer than 1% of cells): a log cannot take them, so they must be answered — as values
    below detection, by the same repair the pack's finding offers — before any logged
    normalization (MS7)."""
    if any(str(i).split("#")[0] == "pack::metabolomics::zeros_or_missing" for i in present):
        return None
    reading = scale_reading(frame, lenses, target)
    if reading is None or reading.kind != "intensities" or not reading.n_zero:
        return None
    block = _block(frame, reading.columns)
    hits = np.sum(np.isfinite(block) & (block == 0), axis=0)
    cols = [c for c, k in zip(reading.columns, hits) if k]
    n = int(hits.sum())
    finding = {
        "id": ZEROS_FINDING, "severity": "warning",
        "title": f"{_count(n)} zeros in {_count(len(cols))} intensity columns, which a log cannot take.",
        "detail": (f"{_count(n)} cells across {_count(len(cols))} raw-intensity columns are exactly "
                   f"zero. A log turns a zero into a missing value; a zero written by an export "
                   f"usually means not detected."),
        "why_it_matters": ("A value below detection filled as if it were missing at random lands in "
                           "the middle of the distribution. Say whether these zeros are "
                           "non-detections; the missing-values answer then fills them below the "
                           "limit, before the log."),
        "affected_columns": cols[:8], "source": "structural", "lens": "metabolomics",
        "evidence": None, "summary": f"{_count(n)} zeros a log cannot take; are they non-detections?",
        "routes_to": "missing", "lever_label": "Say what the zeros are", "group": None,
    }
    return {"params": {"columns": cols, "n_zero": n}, "confidence": "high"}, finding


# ── the method contracts (BLUEPRINT §13) ─────────────────────────────────────


def _register_contracts() -> None:
    from turbotab.core.contracts import (CONTRACTS, ContractOption, MethodContract, Relation,
                                        register_contract)

    if "omics_normalization" in CONTRACTS:
        return
    both = ("prediction", "inference")

    def opt(key: str, label: str, customary: str, sound: str | tuple[str, str], rung: str | tuple[str, str],
            order: int | tuple[int, int]) -> ContractOption:
        s = sound if isinstance(sound, tuple) else (sound, sound)
        r = rung if isinstance(rung, tuple) else (rung, rung)
        o = order if isinstance(order, tuple) else (order, order)
        return ContractOption(key, label, customary, dict(zip(both, s)), dict(zip(both, r)),
                              dict(zip(both, o)))

    register_contract(MethodContract(
        key="omics_normalization", label="Omics normalization", slot="in_fold", scope="training_fold",
        run_order=2.0, needs=("an assay block read as raw counts or raw intensities",),
        question="These columns are raw counts or intensities: how are they normalized?",
        options=(
            opt("pqn_log2", "PQN, then log2", "Customary in metabolomics (Dieterle et al. 2006)",
                "Sound: dilution leaves the values; fit on the training fold.", "recommended", 0),
            opt("qc_pqn_log2", "PQN to pooled QCs", "pmp's default reference (the QC samples)",
                "Sound: the reference is learned from technical replicates only, before the seal.",
                "available", 1),
            opt("log2", "Log2 only", "Customary when dilution is controlled",
                "Leaves dilution in the values.", "rank_lower", 2),
            opt("log_cpm_tmm", "Log-CPM with TMM", "Customary for RNA-seq (edgeR)",
                "Sound: depth and composition leave the values; fit on the training fold.",
                "recommended", 0),
            opt("log_cpm", "Log-CPM", "Customary for RNA-seq", "Removes depth, not composition.",
                "available", 1),
            opt("declared_normalized", "Already normalized", "The researcher's statement",
                "Sound when true; the app cannot check it.", "available", 3),
        ),
        storyboard=("Read each sample's total signal", "Divide by its dilution or depth",
                    "Take log2"),
        relations=(
            Relation("conflicts", "linear_families",
                     "Raw counts or intensities conflict with linear families until a normalization "
                     "is chosen or the values are declared normalized.", rung="refused"),
            Relation("precedes", "detection_limit",
                     "Normalization comes before values below detection are filled."),
            Relation("invalidates", "screen",
                     "A new normalization re-asks which features a screen keeps."),
        ),
        sources=("Dieterle et al. 2006", "Robinson & Oshlack 2010", "Hornung et al. 2015"),
        short="PQN",
        option_slots={"qc_pqn_log2": "in_fold"},
        option_shorts={"log_cpm_tmm": "log-CPM with TMM factors", "log_cpm": "log-CPM",
                       "qc_pqn_log2": "", "log2": "", "declared_normalized": ""},
    ))
    register_contract(MethodContract(
        key="detection_limit", label="Values below detection", slot="in_fold", scope="training_fold",
        run_order=3.0, needs=("columns whose blanks are non-detections",),
        question="These blanks lie below a detection limit: how are they filled?",
        options=(
            opt("censoring_aware", "Censored-normal", f"Recommended for study factors ({LUBIN})",
                ("Sound: the expected value below the limit, fit on the training fold.",
                 "Sound: a draw below the limit within the multiple imputation."), "recommended", 0),
            opt("qrilc", "QRILC", f"Customary for left-censored metabolomics data ({WEI})",
                ("Sound: drawn below each sample's detection quantile.",
                 "Refused: one draw outside the multiple imputation."), ("available", "refused"), (1, 3)),
            opt("half_minimum", "Half the minimum", "Customary: MetaboAnalyst's default",
                f"Customary at ≤ 10% censored; biased beyond ({LUBIN}).", ("available", "available"),
                (2, 1)),
            opt("as_missing", "As any other blank", "Not customary for non-detects",
                ("Ranked lower: kept only with a reason.", "Refused: not missing at random."),
                ("rank_lower", "refused"), (3, 2)),
        ),
        storyboard=("Find each column's detection limit", "Fill below it", "Hand the log only "
                    "positive values"),
        relations=(
            Relation("implies", "censoring_aware",
                     "A detection-limit reading implies censoring-aware handling."),
            Relation("conflicts", "mar_imputation",
                     "MAR imputation of detection-limit blanks is refused under inference.",
                     purposes=("inference",), rung="refused"),
            Relation("precedes", "log_transform",
                     "Values below detection are filled before the log, so no zero or blank is "
                     "logged."),
        ),
        sources=(LUBIN, WEI, LAZAR, "Di Guida et al. 2016"),
        short="detection-limit-aware imputation",
        # QRILC reads only the sample it fills, except a sample too sparse to read, whose values
        # below detection take half the column's minimum on the training fold: the wider scope.
        option_scopes={"qrilc": "training_fold"},
    ))
    register_contract(MethodContract(
        key="log_transform", label="Log transformation", slot="in_fold", scope="row_local",
        run_order=4.0, needs=("positive values",), question="(stated: part of the normalization)",
        options=(opt("log2", "Log2", "Customary for intensities and counts",
                     "Sound: makes multiplicative effects additive.", "recommended", 0),),
        storyboard=("Take log2 of each value",),
        relations=(
            Relation("conflicts", "zeros",
                     "A zero a log would turn missing routes to the detection-limit question, never "
                     "to a median fill.", rung="refused"),
        ),
        sources=("Di Guida et al. 2016",),
        short="log transformation",
    ))
    register_contract(MethodContract(
        key="autoscaling", label="Autoscaling", slot="in_fold", scope="training_fold",
        run_order=9.0, needs=("a family that standardizes its inputs",),
        question="(stated: the family's scaling)",
        options=(opt("autoscaling", "Autoscaling", "Pareto scaling is customary in metabolomics",
                     "Sound: van den Berg et al. (2006) found autoscaling performed better in their "
                     "explorative analysis.", "recommended", 0),),
        storyboard=("Center each column on the training fold", "Divide by its SD"),
        relations=(),
        sources=("van den Berg et al. 2006",),
        short="autoscaling",
    ))
    register_contract(MethodContract(
        key="screen", label="In-fold screening", slot="in_fold", scope="model",
        run_order=8.0, needs=("an outcome", "many more features than rows"),
        question="Screen the features inside each training fold?",
        options=(
            opt("sis", "Sure independence screening", "Customary in genomic prediction",
                ("Sound inside the resampling (Ambroise & McLachlan 2002).",
                 "Refused for the reported model: selection under inference."),
                ("available", "refused"), (0, 0)),
        ),
        storyboard=("Correlate each feature with the outcome on the training fold",
                    "Keep the n / log n strongest"),
        relations=(
            Relation("conflicts", "selection_outside_resampling",
                     "Selection outside the resampling is refused: a false performance number.",
                     purposes=("prediction",), rung="refused"),
        ),
        sources=("Fan & Lv 2008", "Ambroise & McLachlan 2002", "Varma & Simon 2006"),
        short="sure independence screening",
    ))
    register_contract(MethodContract(
        key="multiplicity", label="Multiplicity control", slot="evaluation", scope="model",
        run_order=1.0, needs=("an exposure family",),
        question="The study factors are tested as a family: how is multiplicity controlled?",
        options=(
            opt("bh", "Benjamini–Hochberg q-values", "Customary for MWAS and EWAS",
                f"Sound: holds the false-discovery rate ({BH}).", "recommended", 0),
            opt("stated_count", "Number of tests stated", f"Customary for a few hypotheses ({ROTHMAN})",
                "Sound for a few prespecified hypotheses only.", "block_and_record", 1),
            opt("none", "No control", "Not customary", "Unsound for a family.", "block_and_record", 2),
        ),
        storyboard=("Test each exposure on its own", "Adjust the p-values across the family"),
        relations=(
            Relation("implies", "bh_q_values",
                     "An exposure family implies multiplicity control: BH q-values for feature-wise "
                     "analyses; every member is shown.", purposes=("inference",)),
            Relation("conflicts", "unadjusted_family",
                     "A feature-wise table without multiplicity control is blocked and recorded.",
                     purposes=("inference",), rung="block_and_record"),
        ),
        sources=(BH, ROTHMAN, "Gelman & Loken 2013"),
    ))


# ── the methods paragraph a chain writes (BLUEPRINT §13's chain test) ─────────


def chain_choices(state: Any, steps: Sequence[str], family: str | None = None) -> dict[str, str | None]:
    """The method contracts a run used, each with the option it chose: the pre-seal answers in the
    state, the in-fold steps the design built (``steps``: the family's step keys, in order), the
    batch answer, and an exposure family's multiplicity."""
    from turbotab.core.methods import qc_drift

    choices: dict[str, str | None] = {}
    qc = qc_drift._applied(state)
    if qc is not None:
        _, option, params = qc
        if option in qc_drift.RLSC_OPTIONS and not params.get("problems"):
            choices.update({"qc_detection_filter": None, "qc_rlsc": option, "qc_rsd_filter": None})
        choices["qc_rows_leave"] = None
    norm = normalization_of(state)
    if norm is not None and norm["method"] == "qc_pqn_log2":
        choices["qc_pqn"] = None
    spec_missing = getattr(state, "missing", None)
    for key in steps:
        if key == "d_ratio":
            choices["d_ratio_filter"] = None
        elif key == "normalize" and norm is not None:
            choices["omics_normalization"] = norm["method"]
            if norm["method"] in LOGGED:
                choices["log_transform"] = "log2"
        elif key == "log":
            choices["log_transform"] = "log2"
        elif key == "detect" and spec_missing is not None:
            choices["detection_limit"] = spec_missing.below_detection
        elif key == "batch":
            choices["batch"] = "reference_combat"
        elif key == "screen":
            choices["screen"] = "sis"
        elif key == "scale" and set(getattr(state, "lens", None) or []) & set(OMICS_LENSES):
            choices["autoscaling"] = "autoscaling"
    if norm is not None and norm["method"] == "qc_pqn_log2":
        choices["log_transform"] = "log2"
    batch = getattr(state, "batch", None)
    if batch is not None and batch.method == "covariate":
        choices["batch"] = "covariate"
    if family == "featurewise":
        choices["multiplicity"] = multiplicity_policy(state)["method"]
    return choices


def model_clause(family: str, validation: str | None) -> str | None:
    """The clause the fitted family writes for the methods paragraph: how its parameters were
    tuned against how the rows were validated."""
    if family in ("elastic_net", "screened_elastic_net") and validation in ("kfold", "repeated_kfold"):
        return "elastic-net parameters were tuned in an inner CV nested in an outer CV"
    if family in ("elastic_net", "screened_elastic_net"):
        return "elastic-net parameters were tuned by an inner CV within the rows each fit was given"
    return None


def methods_paragraph(state: Any, steps: Sequence[str], family: str, split: Mapping[str, Any] | None,
                      working: Mapping[str, Any] | None = None,
                      table: Mapping[str, Any] | None = None, *,
                      censored: Mapping[str, Any] | None = None,
                      missing: Mapping[str, Any] | None = None,
                      inference: Mapping[str, Any] | None = None,
                      figure: bool = False) -> str:
    """The methods paragraph a run writes: its first sentence from the chain's contracts (the
    pre-seal clauses, the in-fold group, the model's clause), then the details: the QC answer's
    columns and counts, the fill of values below detection, how units were kept together, and an
    exposure family's tests and multiplicity. The fit stage writes it into each fitted model
    (``FittedModel.methods``), from what the run did:

    * ``censored``: the design's detect step (``{method, columns}``): the features it fills, after
      the QC filters removed theirs, not the columns the answer named;
    * ``missing``: the inference table's missing-data record: under multiple imputation the values
      below detection are drawn within each imputation, a single fill says so;
    * ``inference``: the table's record, whose clustering an inference run states in place of
      folds it does not have;
    * ``figure``: whether the batch figure ComBat serves exists (:func:`batch.batch_figure`);
      "ComBat was used for visualization only" is said only then."""
    from turbotab.core.methods import qc_drift
    from turbotab.core.contracts import paragraph

    purpose = str(getattr(state, "purpose", None) or "prediction")
    choices = chain_choices(state, steps, family)
    batch = getattr(state, "batch", None)
    qc = qc_drift._applied(state)
    run = {"qc_rlsc": choices.get("qc_rlsc"),
           "batch": ({"method": batch.method, "figures": bool(batch.figures and figure)}
                     if batch is not None else {})}
    if qc is not None:
        run["qc_batch"] = qc[2].get("batch_column")
    details: list[str] = []
    if qc is not None and choices.get("qc_rlsc"):
        details.append(qc_drift.rlsc_details(qc[2], (working or {}).get("qc_correction")))
    norm = normalization_of(state)
    if purpose == "inference" and norm is not None and norm["method"] in TRANSFORMS:
        details.append(normalization_details(norm["method"]))
    if "detection_limit" in choices:
        said = detection_details(state, choices["detection_limit"], censored=censored,
                                 purpose=purpose, missing=missing, steps=steps)
        if said:
            details.append(said)
    validation = (split or {}).get("validation")
    if purpose == "prediction" and split and split.get("grouped_by"):
        details.append(f"Folds kept each `{split['grouped_by']}`'s rows together.")
    elif purpose == "inference" and (inference or {}).get("covariance") == "CR2" \
            and (inference or {}).get("grouped_by"):
        details.append(f"Intervals were cluster-robust (CR2) by `{inference['grouped_by']}`.")
    if family == "featurewise" and table is not None:
        details.append(featurewise_details(table, choices.get("multiplicity") or "bh"))
    clause = model_clause(family, validation) if purpose == "prediction" else None
    return paragraph(choices, run, purpose, model_clause=clause, details=details)


# The contracts whose run makes a methods paragraph worth writing (autoscaling alone is any
# standardizing family's, not the omics chain's).
CHAIN_CONTRACTS = ("qc_detection_filter", "qc_rlsc", "qc_rsd_filter", "qc_pqn", "qc_rows_leave",
                   "d_ratio_filter", "omics_normalization", "detection_limit", "log_transform",
                   "batch", "screen", "multiplicity")


def fit_methods(state: Any, family: str, steps: Sequence[str], split: Mapping[str, Any] | None,
                working: Mapping[str, Any] | None, model: Mapping[str, Any],
                censored: Mapping[str, Any] | None, figure: bool = False) -> str | None:
    """The methods paragraph the fit stage writes for one fitted ``model`` (its artifact entry),
    or None when the run used none of the omics chain's contracts."""
    choices = chain_choices(state, steps, family)
    if not set(choices) & set(CHAIN_CONTRACTS):
        return None
    info = model.get("inference") or {}
    table = {"rows": model.get("coefficients") or []} if family == "featurewise" else None
    return methods_paragraph(state, steps, family, split, working, table, censored=censored,
                             missing=info.get("missing"), inference=info, figure=figure)


def unpooled_refusal(label: str, state: Any,
                     frame: pd.DataFrame | None = None) -> tuple[str, list[dict[str, Any]]]:
    """(reason, exits) of a table that pools no multiple imputations (feature-wise tests), its
    reason naming exactly the ways forward its exits take (:func:`featurewise_missing_exits`;
    ``frame``: the rows and columns the table is estimated from)."""
    from turbotab.core.methods.exposure_form import SPLINE_MIN_VALUES

    exits = featurewise_missing_exits(state, frame)
    ways = [EXIT_WORDS[e["way"]] for e in exits]
    reason = (f"{label} is not pooled over multiple imputations here: its tests run feature by "
              f"feature over more columns than an imputation model holds.")
    complete = complete_rows(frame)
    if complete is not None and not any(e["way"] == "complete_case" for e in exits):
        reason += (f" Complete cases are not offered: {complete:,} of the {len(frame):,} rows "
                   f"hold every value, fewer than the {SPLINE_MIN_VALUES} the design needs.")
    said = _or(ways)
    reason += f" {said[:1].upper()}{said[1:]}."
    return reason, [{k: v for k, v in e.items() if k != "way"} for e in exits]


# The ways forward a feature-wise table offers, as its refusal names them (each a request).
EXIT_WORDS = {
    "complete_case": "choose complete cases",
    "censoring_aware": "fill the values below detection once, censoring-aware, recorded as a "
                       "limitation",
    "single_fill": "fill the missing values once, recorded as a limitation",
}


def _or(ways: Sequence[str]) -> str:
    return ways[0] if len(ways) == 1 else ", ".join(ways[:-1]) + f", or {ways[-1]}"


def complete_rows(frame: pd.DataFrame | None) -> int | None:
    """How many of ``frame``'s rows hold every value (None without a frame)."""
    if frame is None:
        return None
    return int(frame.notna().all(axis=1).sum()) if frame.shape[1] else int(len(frame))


def featurewise_missing_exits(state: Any, frame: pd.DataFrame | None = None
                              ) -> list[dict[str, Any]]:
    """Where a feature-wise table goes when its missing-values answer is multiple imputation, which
    it cannot pool (more columns than an imputation model holds): complete cases, offered only when
    they keep rows enough for the design (:data:`SPLINE_MIN_VALUES`, the fewest a spline's knots
    can be placed on; ``frame``: the table's rows and columns, so a way forward that leaves nothing
    to fit is never offered, DoD §1); when values below detection are named, one censoring-aware
    fill recorded as a limitation; and when neither is open, one fill recorded as a limitation (a
    single fill under inference is block-and-record: its intervals are too narrow). Each exit
    carries ``way``, its key in :data:`EXIT_WORDS`."""
    from turbotab.core.methods.exposure_form import SPLINE_MIN_VALUES

    spec = getattr(state, "missing", None)
    keep = {"drop_columns": list(getattr(spec, "drop_columns", None) or [])}
    exits: list[dict[str, Any]] = []
    complete = complete_rows(frame)
    if complete is None or complete >= SPLINE_MIN_VALUES:
        label = ("Complete cases" if complete is None
                 else f"Complete cases ({complete:,} of the {len(frame):,} rows)")
        exits.append({"label": label, "way": "complete_case",
                      "decision": {"kind": "set_missing", "strategy": "complete_case", **keep}})
    censored = list(getattr(spec, "censored_columns", None) or [])
    if censored:
        exits.append({
            "label": "Fill values below detection once, censoring-aware, recorded as a limitation",
            "way": "censoring_aware",
            "decision": {"kind": "set_missing", "strategy": "impute",
                         "below_detection": "censoring_aware", "censored_columns": censored,
                         "acknowledged": True, **keep}})
    if not exits:
        exits.append({"label": "Fill the missing values once, recorded as a limitation",
                      "way": "single_fill",
                      "decision": {"kind": "set_missing", "strategy": "impute",
                                   "acknowledged": True, **keep}})
    return exits


def normalization_details(method: str) -> str:
    """The normalization's sentence where nothing is grouped by fold (inference: every row)."""
    return {
        "log_cpm_tmm": ("Counts were transformed to log2 counts per million on library sizes scaled "
                        "by TMM factors (edgeR, prior count 2; Robinson and Oshlack 2010)."),
        "log_cpm": "Counts were transformed to log2 counts per million (edgeR, prior count 2).",
        "pqn_log2": ("Intensities were normalized by probabilistic quotient normalization "
                     "(Dieterle et al. 2006) and log2-transformed."),
        "qc_pqn_log2": ("Intensities were normalized by probabilistic quotient normalization against "
                        "the pooled-QC reference spectrum and log2-transformed."),
        "log2": "Intensities were log2-transformed.",
    }[method]


def detection_details(state: Any, method: str | None, *, censored: Mapping[str, Any] | None = None,
                      purpose: str = "prediction", missing: Mapping[str, Any] | None = None,
                      steps: Sequence[str] | None = None) -> str | None:
    """The sentence for values below detection: the features the design's detect step fills
    (``censored``, after the QC filters; the answer's own columns when no design is at hand), and
    how, by purpose. Under inference with multiple imputation they are drawn below the limit within
    each imputation; filled once, the sentence says it is a single fill recorded as a limitation.
    "After normalization and before the log" is said only where the design's steps put the fill
    there (``steps``)."""
    if censored is not None:
        n = len(censored.get("columns") or [])
    else:
        n = len(getattr(getattr(state, "missing", None), "censored_columns", None) or [])
    if not n:
        return None
    omics = bool(set(getattr(state, "lens", None) or []) & set(OMICS_LENSES))
    noun = "feature" if omics else "column"
    what = f"{n:,} {noun}{'s' if n != 1 else ''}"
    keys = list(steps) if steps is not None else ["normalize", "detect", "log"]
    between = ("normalize" in keys and "log" in keys
               and keys.index("normalize") < keys.index("detect") < keys.index("log")
               if "detect" in keys else False)
    where = " after normalization and before the log" if between else ""
    rows = "the training fold" if purpose == "prediction" else "the analyzed rows"
    record = dict(missing or {})
    if getattr(getattr(state, "missing", None), "strategy", None) == "complete_case":
        # Complete cases leave a row with any blank before the design, so nothing is filled.
        return (f"Rows with a value below detection in any of {what} were left out (complete "
                f"cases).")
    if purpose == "inference" and record.get("method") == "multiple_imputation":
        m = int(record.get("m") or 0)
        if method == "half_minimum":  # filled before the chained equations (``imputation_frame``)
            return (f"Values below detection in {what} were set to half the {noun}'s smallest "
                    f"detected value before the {m:,} multiple imputations ({LUBIN}).")
        return (f"Values below detection in {what} were drawn below the limit from a censored-normal "
                f"(Tobit) model within each of the {m:,} multiple imputations, given the outcome "
                f"({LUBIN}).")
    if method == "half_minimum":
        how = f"half the {noun}'s smallest detected value on {rows}"
    elif method == "qrilc":
        how = f"a draw below its sample's detection quantile (QRILC; {LAZAR})"
    else:
        # Before a log the censored normal is fitted on the log scale; elsewhere on the scale that
        # fits better (``imputation.censoring_of``).
        scale = (f"the {noun}'s logarithm" if between
                 else f"the {noun} or its logarithm, whichever fits better,")
        how = (f"its expected value below the limit under a left-censored normal fitted to {scale} "
               f"on {rows} ({LUBIN})")
    said = f"Values below detection in {what} were filled{where}, each by {how}."
    if purpose == "inference":
        said += " A single fill, recorded as a limitation: its intervals are too narrow."
    return said


def featurewise_details(table: Mapping[str, Any], method: str) -> str:
    rows = [r for r in table.get("rows") or [] if r.get("p") is not None]
    m = len(rows)
    found = sum(1 for r in rows if r.get("q") is not None and r["q"] < 0.05)
    if method == "bh":
        return (f"Each of the {m:,} features was tested in its own linear model with the "
                f"covariates; Benjamini–Hochberg q-values across the {m:,} tests ({found:,} below "
                f"0.05), every feature shown.")
    if method == "stated_count":
        return (f"Each of the {m:,} features was tested in its own linear model with the "
                f"covariates; unadjusted p-values, {m:,} tests ({ROTHMAN}).")
    return (f"Each of the {m:,} features was tested in its own linear model with the covariates; no "
            f"multiplicity control, recorded as a limitation.")


def _register() -> None:
    from turbotab.core.decisions import register_validator

    _register_repairs()
    register_validator("select_models", _linear_families_need_a_scale)
    register_validator("apply_repair", _logged_options_wait_for_the_zeros)
    register_validator("set_missing", _detection_limit_leash)
    _register_screened_family()
    _register_contracts()
    from turbotab.core.repairs import FAMILIES, register_family

    # The app's own zeros finding offers the pack's "zeros mean not detected" repair.
    register_family(FAMILIES["zeros_nondetect"], [ZEROS_FINDING])
    register_validator("set_multiplicity", _multiplicity_leash)
    from turbotab.core.voice import register_sentence

    @register_sentence("set_multiplicity")
    def _set_multiplicity(d: Any, state: Any, ctx: Any) -> str:
        return multiplicity_sentence(d.method, d.acknowledged)

    # QC drift and batch handling enter with the omics chain (MS7); importing registers them.
    from turbotab.core.methods import batch, qc_drift  # noqa: F401


_register()

__all__ = [
    "CHECK_ALPHA", "COUNT_METHODS", "DECLARED", "FINDING", "INTENSITY_METHODS", "LogCPM",
    "OMICS_LENSES", "QuotientLog", "RAW_COUNTS_COACHING", "ScaleReading", "TRANSFORMS",
    "candidate_columns", "check_on", "describe", "design_normalization", "design_refusal",
    "library_size_check", "log_cpm", "normalization_of", "normalizer", "refusal_for",
    "restate_raw_counts", "scale_finding", "scale_reading", "tmm_factor", "tmm_factors",
    "tmm_reference", "zeros_message", "zeros_refusal",
]

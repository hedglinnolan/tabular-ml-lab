"""Sample fits for the pages that quote a price, cached by what the price depends on.

`ml.cost_model` is the arithmetic and has no Streamlit in it. This is the
thin layer between it and a page: which fit to time (the one the page is
about to make, through the preprocessing it will use), where to keep the
result (session state, keyed by data, settings and machine, as plain values
so a session save can carry them), and how to say it (one sentence per
button, with the provenance of the number attached).

Everything here degrades to *no number* rather than a wrong one: a probe
whose sample fit raises is `None`, and the sentence then names the model as
not estimated. The page must never say "about 2 minutes" about a model whose
fit it never timed.
"""
from __future__ import annotations

import hashlib
import logging
import os
import platform
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import streamlit as st
from sklearn.base import clone

from ml.cost_model import (
    ModelCost, ScalingProbe, cv_worker_bytes, humanize_bytes, humanize_seconds,
    probe_fit_seconds, provenance_clause, training_run_cost,
)

logger = logging.getLogger(__name__)

#: Below this many bytes held by the fold workers together, the memory line
#: is not shown: a 262 KB note under a 600-row study is friction, not
#: disclosure. Above it the number is worth a line — the audit's case is a
#: 200 x 100,000 clinical export at 160 MB per copy.
MEMORY_LINE_MIN_BYTES = 16 * 1024 * 1024

#: Session key holding every probe taken this session, keyed by `probe_key`.
#: Plain dicts of numbers, so a session save's JSON coercion carries them;
#: the key includes the machine, so a session restored on another box misses
#: and probes again rather than quoting this one's clock as its own.
PROBE_STORE_KEY = "_fit_probes"


def frame_fingerprint(X: Any, y: Any = None) -> str:
    """Shape, dtypes and a strided sample of the values — cheap and stable.

    At most a 64 x 64 sample of cells is hashed, so a 20,000-column frame
    costs the same as a 20-column one; the shape and dtypes catch everything
    a sample can miss except an edit inside the unsampled cells, and the
    price of that miss is a stale timing, not a wrong result.
    """
    h = hashlib.sha1()
    if hasattr(X, "iloc"):
        n, p = X.shape
        sample = X.iloc[::max(n // 64, 1), ::max(p // 64, 1)]
        h.update(repr(X.shape).encode())
        h.update(repr(list(map(str, X.dtypes))).encode())
        h.update(repr(sample.to_numpy(dtype=object).tolist()).encode())
    else:
        arr = np.asarray(X)
        h.update(repr(arr.shape).encode())
        h.update(str(arr.dtype).encode())
        if arr.ndim == 2:
            n, p = arr.shape
            sample = arr[::max(n // 64, 1), ::max(p // 64, 1)]
        else:
            sample = arr.reshape(-1)[::max(arr.size // 4096, 1)]
        h.update(repr(np.asarray(sample, dtype=object).tolist()).encode())
    if y is not None:
        y_arr = np.asarray(y.to_numpy() if hasattr(y, "to_numpy") else y)
        h.update(repr(y_arr.shape).encode())
        h.update(repr(np.asarray(y_arr.reshape(-1)[::max(y_arr.size // 512, 1)], dtype=object).tolist()).encode())
    return h.hexdigest()[:16]


def estimator_signature(estimator: Any) -> str:
    """The class and its constructor parameters, in a stable order."""
    try:
        params = estimator.get_params(deep=False)
    except Exception:
        params = {}
    body = sorted((str(k), repr(v)) for k, v in params.items())
    return f"{type(estimator).__module__}.{type(estimator).__name__}{body}"


def pipeline_signature(pipeline: Any) -> str:
    if pipeline is None:
        return "no-pipeline"
    try:
        return repr(pipeline)
    except Exception:
        return type(pipeline).__name__


def machine_signature() -> str:
    return f"{platform.node()}|{os.cpu_count()}|{platform.machine()}"


def probe_key(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode("utf-8", "replace")).hexdigest()[:20]


class FitThrough:
    """The fit a training run makes for one model, as a callable a probe can time.

    Clones the pipeline and the estimator on every call — a fitted object is
    never reused — fits the pipeline on the sample rows and the estimator on
    the transformed rows, exactly as pages/06 does, and remembers the
    transformed width so the memory arithmetic can use the width the
    estimator actually sees rather than the raw column count.
    """

    def __init__(self, estimator: Any, pipeline: Any = None):
        self.estimator = estimator
        self.pipeline = pipeline
        self.width: Optional[int] = None

    def __call__(self, X_sub: Any, y_sub: Any) -> None:
        if self.pipeline is not None:
            transformed = clone(self.pipeline).fit_transform(X_sub, y_sub)
        else:
            transformed = X_sub.to_numpy() if hasattr(X_sub, "to_numpy") else np.asarray(X_sub)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        shape = getattr(transformed, "shape", None)
        self.width = int(shape[1]) if shape is not None and len(shape) == 2 else None
        clone(self.estimator).fit(transformed, y_sub)


def _freeze(probe: Optional[ScalingProbe], width: Optional[int]) -> Optional[Dict[str, Any]]:
    if probe is None:
        return None
    return {
        "points": [[int(s), float(t)] for s, t in probe.points],
        "slope": float(probe.slope),
        "slope_assumed": bool(probe.slope_assumed),
        "size_target": int(probe.size_target),
        "seconds_at_target": float(probe.seconds_at_target),
        "width": width,
    }


def _thaw(entry: Optional[Dict[str, Any]]) -> Optional[ScalingProbe]:
    if not entry:
        return None
    try:
        return ScalingProbe(
            points=tuple((int(s), float(t)) for s, t in entry["points"]),
            slope=float(entry["slope"]),
            slope_assumed=bool(entry["slope_assumed"]),
            size_target=int(entry["size_target"]),
            seconds_at_target=float(entry["seconds_at_target"]),
        )
    except Exception:
        return None


def _store() -> Dict[str, Any]:
    store = st.session_state.get(PROBE_STORE_KEY)
    if not isinstance(store, dict):
        store = {}
        st.session_state[PROBE_STORE_KEY] = store
    return store


def probe_is_cached(key: str) -> bool:
    return key in _store()


def cached_probe(key: str, fit: FitThrough, X: Any, y: Any) -> Tuple[Optional[ScalingProbe], Optional[int]]:
    """The probe for `key`, taken now if it has not been taken this session.

    Returns the probe (None when the sample fit raised) and the transformed
    width the fit saw. A cached `None` is kept as `None`: a fit that raised
    on a sample will raise on the run too, and re-probing it every rerun
    would only re-raise it.
    """
    store = _store()
    if key in store:
        entry = store[key]
        return _thaw(entry), (entry or {}).get("width")
    try:
        probe = probe_fit_seconds(fit, X, y)
    except Exception:
        logger.exception("sample fit probe failed for %s", key)
        probe = None
    store[key] = _freeze(probe, fit.width)
    return probe, fit.width


# ── the training run on pages/06, priced ─────────────────────────────────────

@dataclass(frozen=True)
class TrainingPrice:
    """Both buttons' cost in seconds, the memory the folds hold, and the words."""

    costs: Tuple[ModelCost, ...]
    not_probed: Tuple[str, ...]
    n_train: int
    cv_folds: int
    train_matrix_bytes: int

    @property
    def standard_seconds(self) -> float:
        return sum(c.final_fit_seconds + c.cv_seconds for c in self.costs)

    @property
    def optimized_seconds(self) -> float:
        return sum(c.seconds for c in self.costs)

    @property
    def per_model(self) -> Dict[str, Dict[str, float]]:
        return {c.key: {"fit": c.final_fit_seconds, "cv": c.cv_seconds, "optuna": c.optuna_seconds}
                for c in self.costs if c.estimated}

    @property
    def unestimated(self) -> Tuple[str, ...]:
        return tuple(c.key for c in self.costs if not c.estimated) + self.not_probed

    @property
    def provenance(self) -> str:
        return provenance_clause(self.costs)

    def time_sentence(self, optimized_available: bool) -> str:
        if not any(c.estimated for c in self.costs):
            names = ", ".join(m.upper() for m in self.unestimated)
            return (f"**Time on this machine:** not estimated — the sample fit "
                    f"{'for ' + names + ' ' if names else ''}could not be timed.")
        parts = [f"**Time on this machine:** Train Models {humanize_seconds(self.standard_seconds)}"]
        if optimized_available:
            parts.append(f"; with hyperparameter optimization {humanize_seconds(self.optimized_seconds)} "
                         f"at the default settings, which is a floor — Optuna explores larger ones")
        sentence = "".join(parts) + f" ({self.provenance})."
        per = " · ".join(
            f"{c.key.upper()} {humanize_seconds(c.final_fit_seconds + c.cv_seconds)}"
            for c in self.costs if c.estimated)
        if per:
            sentence += f" Per model: {per}."
        if self.unestimated:
            names = ", ".join(m.upper() for m in self.unestimated)
            sentence += (f" Not estimated: {names}"
                         f"{' — the neural network states its own epochs while it trains' if 'nn' in self.unestimated else ''}.")
        return sentence

    def memory_sentence(self, cores: Optional[int]) -> Optional[str]:
        if not self.cv_folds or self.train_matrix_bytes <= 0:
            return None
        workers = max(min(self.cv_folds, int(cores) if cores else self.cv_folds), 1)
        total = cv_worker_bytes(self.train_matrix_bytes, self.cv_folds, cores)
        if total < MEMORY_LINE_MIN_BYTES:
            return None
        return (f"**Memory while cross-validating:** each of the {workers} fold workers holds "
                f"its own copy of the {humanize_bytes(self.train_matrix_bytes)} training matrix, "
                f"so about {humanize_bytes(total)} in all — exact arithmetic, no estimate.")

    def as_state(self) -> Dict[str, Any]:
        """What the training loop reads back to quote each model as it starts."""
        return {
            "standard": self.standard_seconds,
            "optimized": self.optimized_seconds,
            "per_model": self.per_model,
            "provenance": self.provenance,
            "n_train": self.n_train,
            "unestimated": list(self.unestimated),
            "train_matrix_bytes": int(self.train_matrix_bytes),
            "cv_folds": int(self.cv_folds),
        }


def price_training_run(estimators: Dict[str, Any], pipelines: Dict[str, Any],
                       X_train: Any, y_train: Any, *, task_type: str,
                       cv_folds: int, cv_models: Sequence[str],
                       optuna_trials: Dict[str, int], not_probed: Sequence[str],
                       spinner_text: str = "Timing a sample fit of each selected model…") -> Optional[TrainingPrice]:
    """Probe every estimator the run will fit and price both buttons.

    `estimators` are the unfitted estimators the run will build, keyed by
    model; `pipelines` the preprocessing each will go through (None for
    none). Probes are cached across reruns by data, estimator, pipeline and
    machine, so the spinner is shown only when at least one is missing.
    Returns None if the pricing itself failed for a reason that is not a
    single model's fit — the page then says nothing rather than something
    wrong.
    """
    try:
        data_fp = frame_fingerprint(X_train, y_train)
        machine = machine_signature()
        keys = {
            m: probe_key(m, task_type, estimator_signature(est),
                         pipeline_signature(pipelines.get(m)), data_fp, machine)
            for m, est in estimators.items()
        }
        need = [m for m, k in keys.items() if not probe_is_cached(k)]
        probes: Dict[str, Optional[ScalingProbe]] = {}
        widths: Dict[str, Optional[int]] = {}

        def _take_all() -> None:
            for m, est in estimators.items():
                fit = FitThrough(est, pipelines.get(m))
                probes[m], widths[m] = cached_probe(keys[m], fit, X_train, y_train)

        if need:
            with st.spinner(spinner_text):
                _take_all()
        else:
            _take_all()

        n_train = int(len(X_train))
        costs = training_run_cost(probes, n_train, cv_folds, cv_models, optuna_trials)
        width = max((w for w in widths.values() if w), default=0)
        matrix_bytes = n_train * width * 8
        return TrainingPrice(costs=costs, not_probed=tuple(not_probed), n_train=n_train,
                             cv_folds=int(cv_folds), train_matrix_bytes=matrix_bytes)
    except Exception:
        logger.exception("pricing the training run failed")
        return None


# ── a loop of refits, priced (pages/08) ──────────────────────────────────────

def price_refits(estimators: Dict[str, Any], pipelines: Dict[str, Any], X: Any, y: Any, *,
                 task_type: str, refits_per_model: int, rows_per_refit: int,
                 spinner_text: str = "Timing a sample fit…") -> Optional[Dict[str, Any]]:
    """Seconds for `refits_per_model` fits of each estimator on `rows_per_refit` rows.

    Returns {"seconds", "provenance", "unestimated", "per_model"} or None
    when nothing could be priced.
    """
    try:
        data_fp = frame_fingerprint(X, y)
        machine = machine_signature()
        keys = {m: probe_key(m, task_type, estimator_signature(est),
                             pipeline_signature(pipelines.get(m)), data_fp, machine)
                for m, est in estimators.items()}
        need = [m for m, k in keys.items() if not probe_is_cached(k)]
        probes: Dict[str, Optional[ScalingProbe]] = {}

        def _take_all() -> None:
            for m, est in estimators.items():
                probes[m], _ = cached_probe(keys[m], FitThrough(est, pipelines.get(m)), X, y)

        if need:
            with st.spinner(spinner_text):
                _take_all()
        else:
            _take_all()
        per_model = {m: p.seconds_at(rows_per_refit) * max(int(refits_per_model), 0)
                     for m, p in probes.items() if p is not None}
        costs = tuple(ModelCost(m, p, p.seconds_at(rows_per_refit) if p else 0.0, 0.0, 0.0)
                      for m, p in probes.items())
        return {
            "seconds": sum(per_model.values()),
            "provenance": provenance_clause(costs),
            "unestimated": [m for m, p in probes.items() if p is None],
            "per_model": per_model,
        }
    except Exception:
        logger.exception("pricing the refit loop failed")
        return None


def refits_sentence(price: Optional[Dict[str, Any]], what: str) -> Optional[str]:
    """'⏱️ About 2 minutes on this machine for 24 refits (projected from …).'"""
    if not price or not price.get("per_model"):
        return None
    sentence = (f"**Time on this machine:** {humanize_seconds(price['seconds'])} for {what} "
                f"({price['provenance']}).")
    if price.get("unestimated"):
        sentence += f" Not estimated: {', '.join(m.upper() for m in price['unestimated'])}."
    return sentence

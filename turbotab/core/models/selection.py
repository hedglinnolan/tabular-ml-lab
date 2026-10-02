"""One declared model: choosing among families, and what the choice costs (AUDIT_REPORT §5 WP8: ME-13).

Two rules, one for each side of the seal.

**Before the seal is opened, the choice is made on cross-validation and its optimism is stated.**
The family with the best cross-validated score was chosen *because* that score was best, so it
flatters the family (Varma & Simon 2006, *BMC Bioinformatics* 7:91: "The CV error estimate for the
classifier with the optimal parameters was found to be a substantially biased estimate of the true
error"). On null binary data the CV-best of the three families averaged an AUC of 0.536 against a
true 0.50 (audit A7). :func:`selection_optimism` estimates that optimism with Tibshirani &
Tibshirani's bias correction (2009, *Ann Appl Stat* 3:822, eq. 3), which "uses only quantities that
have already been computed for the CV estimate … and requires no new model fitting":

    Bias = (1/K) Σ_k [e_k(θ̂) − e_k(θ̂_k)],  θ̂_k the minimizer of e_k(θ)

with e_k(θ) the error of choice θ (here a family) on fold k and θ̂ the choice that is best over all
folds. For a score where higher is better (AUC, macro-F1) e_k is the negated fold score; for R²,
whose cross-validated estimate pools every out-of-fold prediction (``models/metrics.py``), e_k is
fold k's mean squared error, the paper's own loss, and the bias is put on the R² scale by the
pooled no-predictor error the R² divides by, which is the same for every family on the same folds.
"Since Bias is a mean over K folds, we can also use the standard error of the mean as an approximate
estimate for its standard deviation."

**Opening the seal needs the final family declared** (:func:`_a_final_model_is_declared`), chosen
on cross-validation before any held-out score is seen; with one family fitted it is that one. Once
opened, the served fit marks the declared family ``final`` (its held-out score is the reported
result) and the others ``secondary`` (:func:`mark_final`): the best of several held-out scores,
picked after seeing them, is optimistic in turn (+0.023 AUC on 150-row holdouts, audit A7).

Importing this module registers the ``open_seal`` validator and completion.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from turbotab.core import decisions
from turbotab.core.decisions import OpenSeal, Refusal

LOWER_IS_BETTER = frozenset({"rmse", "mae", "brier", "log_loss"})
METHOD = "Tibshirani & Tibshirani (2009)"


# ── the optimism of choosing on cross-validation ─────────────────────────────


def _fold_errors(metric: str, results: Mapping[str, Any]) -> tuple[np.ndarray, float] | None:
    """Errors e_k(m) (folds × families, lower is better) and the factor that turns a bias in them
    into one on ``metric``'s scale; None when the folds' scores cannot be lined up."""
    cvs = list(results.values())
    k = {len(cv.per_fold) for cv in cvs}
    if len(k) != 1 or not k.pop():
        return None
    if metric == "r2" and all(cv.parts for cv in cvs):
        mse = np.array([[part.sse / part.n if part.n else np.nan for part in cv.parts]
                        for cv in cvs], dtype=float).T
        parts = cvs[0].parts
        n = sum(part.n for part in parts)
        null = sum(part.sst for part in parts) / n if n else float("nan")
        if not (np.isfinite(null) and null > 0):
            return None
        return mse, 1.0 / null
    scores = np.array([[fold.get(metric, np.nan) for fold in cv.per_fold] for cv in cvs], dtype=float).T
    return (scores if metric in LOWER_IS_BETTER else -scores), 1.0


def selection_optimism(task: str, metric: str, results: Mapping[str, Any],
                       labels: Mapping[str, str] | None = None,
                       metric_label: str | None = None) -> dict[str, Any] | None:
    """How much the best family's cross-validated ``metric`` flatters it, by Tibshirani &
    Tibshirani's correction over the folds the families share; None with fewer than two families.

    ``results`` maps each family to its :class:`~turbotab.core.models.metrics.CrossValidated`.
    Returns the ``selection`` artifact field: the best family, its CV score, the optimism (an amount
    in the score's units that the score overstates the family's performance by), its approximate
    standard error, the corrected score, and the sentence that states it.
    """
    keys = list(results)
    if len(keys) < 2:
        return None
    lined = _fold_errors(metric, results)
    if lined is None:
        return None
    errors, scale = lined
    keep = np.isfinite(errors).all(axis=1)
    errors = errors[keep]
    if errors.shape[0] < 2:
        return None
    estimates = [results[k].summary(task)[metric]["estimate"] for k in keys]
    if not all(e is not None and math.isfinite(e) for e in estimates):
        return None
    lower = metric in LOWER_IS_BETTER
    best = int(np.argmin(estimates) if lower else np.argmax(estimates))
    per_fold = (errors[:, best] - errors.min(axis=1)) * scale
    optimism = float(per_fold.mean())
    se = float(per_fold.std(ddof=1) / math.sqrt(len(per_fold)))
    cv = float(estimates[best])
    corrected = cv + optimism if lower else cv - optimism
    names = labels or {}
    label = metric_label or metric
    family = names.get(keys[best], keys[best])
    text = (f"Choosing the best of {len(keys)} families by cross-validated {label} flatters it: "
            f"{family}'s CV {label} of {_num(cv)} is {'low' if lower else 'high'} by about "
            f"{_num(optimism)} (± {_num(se)}, {METHOD}), so about {_num(corrected)} is expected on "
            f"new rows. The held-out score of a family declared before the seal is opened carries "
            f"no such optimism.")
    return {"metric": metric, "families": keys, "best": keys[best], "cv": cv, "optimism": optimism,
            "optimism_se": se, "corrected": corrected, "folds": int(len(per_fold)), "method": METHOD,
            "text": text}


def _num(value: float) -> str:
    return f"{value:.3f}".replace("-", "−")


# ── declaring the final family before the seal is opened ─────────────────────


def _ctx(ctx: Any, name: str) -> Any:
    if ctx is None:
        return None
    if isinstance(ctx, Mapping):
        return ctx.get(name)
    return getattr(ctx, name, None)


def _openable_fit(ctx: Any) -> Mapping[str, Any] | None:
    """The fit the seal would open on, when the seal can be opened at all; else None (the seal's
    own validator says why it cannot)."""
    state = _ctx(ctx, "state")
    if state is None or getattr(state, "seal_opened", None):
        return None
    split = getattr(state, "split", None)
    if split is None or float(getattr(split, "holdout", 0) or 0) == 0:
        return None
    artifact = _ctx(ctx, "artifact")
    if not callable(artifact):
        return None
    try:
        fit = artifact("fit")
    except Exception:  # noqa: BLE001 - no fit: the seal's validator refuses
        return None
    if not isinstance(fit, Mapping) or not int(fit.get("n_holdout") or 0):
        return None
    return fit


def _cv_of(fit: Mapping[str, Any], model: Mapping[str, Any]) -> float | None:
    metric = fit.get("primary_metric")
    entry = (model.get("cv") or {}).get(metric) or {}
    value = entry.get("estimate", entry.get("mean"))
    return float(value) if value is not None and math.isfinite(float(value)) else None


def _a_final_model_is_declared(decision: Any, ctx: Any) -> None:
    fit = _openable_fit(ctx)
    if fit is None:
        return
    models = [m for m in fit.get("models") or [] if m.get("family")]
    fitted = [str(m["family"]) for m in models]
    metric = str(fit.get("primary_metric") or "")
    label = (fit.get("metric_labels") or {}).get(metric, metric)
    lower = metric in LOWER_IS_BETTER
    ranked = sorted(models, key=lambda m: (_cv_of(fit, m) is None,
                                           (1 if lower else -1) * (_cv_of(fit, m) or 0.0)))
    exits = []
    for i, m in enumerate(ranked):
        cv = _cv_of(fit, m)
        said = f" (CV {label} {cv:.3f}{', the best' if i == 0 and len(ranked) > 1 else ''})" \
            if cv is not None else ""
        exits.append({"label": f"Open with {m.get('label') or m['family']} as the final model{said}",
                      "decision": OpenSeal(family=str(m["family"]))})
    family = getattr(decision, "family", None)
    if family is not None and family not in fitted:
        raise Refusal("not_a_fitted_family",
                      f"`{family}` is not one of the fitted families, so it cannot be the final "
                      f"model.", exits=exits)
    if family is None and len(fitted) > 1:
        raise Refusal(
            "final_model_needed",
            f"Name the final model before the held-out rows are opened, choosing it on "
            f"cross-validation: its held-out {label} is then the result. The best of "
            f"{len(fitted)} held-out scores, picked after seeing them, would flatter itself.",
            exits=exits)


def _the_only_family_is_the_final_one(decision: Any, ctx: Any) -> Any:
    """With one family fitted, the opening declares it (nothing else could be the final model)."""
    if getattr(decision, "family", None) is not None:
        return decision
    fit = _openable_fit(ctx)
    if fit is None:
        return decision
    fitted = [str(m["family"]) for m in fit.get("models") or [] if m.get("family")]
    return OpenSeal(family=fitted[0]) if len(fitted) == 1 else decision


decisions.register_validator("open_seal", _a_final_model_is_declared)
decisions.register_completion("open_seal", _the_only_family_is_the_final_one)


# ── the served fit: final and secondary ──────────────────────────────────────


def declared_family(records: Sequence[Any]) -> str | None:
    """The family the opening declared final, or None (not opened, or opened before a final model
    had to be declared)."""
    from turbotab.core.seal import opening

    record = opening(records)
    return getattr(record.decision, "family", None) if record is not None else None


def mark_final(out: dict[str, Any], *, opened: bool, family: str | None) -> dict[str, Any]:
    """Mark the served fit's declared final family and its secondary ones (in place; returned).

    Before the opening nothing is final: ``final_model`` and every ``role`` are null.
    """
    models = out.get("models") or []
    for m in models:
        m["role"] = None
    out["final_model"] = None
    out["final_note"] = None
    if not opened or not int(out.get("n_holdout") or 0):
        return out
    metric = str(out.get("primary_metric") or "")
    label = (out.get("metric_labels") or {}).get(metric, metric)
    if family is None:
        out["final_note"] = (f"The seal was opened before a final model had to be declared, so no "
                             f"held-out {label} is the reported result, and the best of them, "
                             f"picked after seeing them, is optimistic.")
        return out
    named = next((m for m in models if m.get("family") == family), None)
    if named is None:
        for m in models:
            m["role"] = "secondary"
        out["final_note"] = (f"`{family}` was declared the final model when the seal was opened, "
                             f"but it is not among the families fitted now, so every held-out "
                             f"score here is secondary.")
        return out
    out["final_model"] = family
    for m in models:
        m["role"] = "final" if m is named else "secondary"
    others = len(models) - 1
    rest = (f"; the other {others} {'family' if others == 1 else 'families'}' held-out scores are "
            f"secondary" if others else "")
    out["final_note"] = (f"{named.get('label') or family} was declared the final model on "
                         f"cross-validation before the held-out rows were opened, so its held-out "
                         f"{label} is the reported result{rest}.")
    return out


__all__ = ["LOWER_IS_BETTER", "METHOD", "declared_family", "mark_final", "selection_optimism"]

"""Batch handling by purpose (MODELING_SEQUENCE §1 row 4, §2 "Batch", §4; V2 definition of done,
genomics extended).

* **Inference — batch as a covariate, first.** Nygaard, Rødland & Hovig (2016, Biostatistics
  17:29): "For an investigator facing an unbalanced data set with batch effects, our primary
  advice would be to account for batch in the statistical analysis." The batch column enters the
  outcome model (each feature-wise test) as a categorical covariate.
* **Prediction — ComBat with a reference batch, fitted in each training fold, without the
  outcome.** Zhang, Jenkins, Manimaran & Johnson (2018, BMC Bioinformatics 19:262): "establishing
  the training set as a 'reference batch' to which all future batches will be standardized ...
  would allow the training data and biomarker to be fixed a priori." :class:`ReferenceComBat`
  fits ComBat (Johnson, Li & Rabinovic 2007; sva's parametric empirical Bayes, ported line by line
  in :func:`combat_fit`) on the training rows with their largest batch as the reference, keeps
  every estimate, and adjusts a held-out row of a batch it has seen by those estimates; a batch
  absent from the training rows is adjusted by its own estimates against the frozen reference
  (Hornung et al. 2017's add-on batch-effect removal). The outcome is never in its model.
* **ComBat with the outcome protected** (``mod = model.matrix(~ outcome)``) is refused for testing
  under inference — Nygaard et al.: "this approach may systematically induce incorrect group
  differences in downstream analyses when groups are distributed between the batches in an
  unbalanced manner" — and allowed there for figures only; under prediction it is refused, because
  it reads the held-out rows' outcomes (leakage).
* **Batch perfectly confounded with the outcome** — every batch holds one outcome level, or the
  outcome is constant within each batch — is refused under both purposes: no model can tell the
  batch from the outcome.
* **Batch correction precedes in-fold screening**: the batch step is a shared step, and a family's
  screen runs after every shared step (``models.pipeline.family_steps``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

CONV = 1e-4  # sva's it.sol convergence
MAX_ITERATIONS = 10_000  # sva has no cap; this one only stops a loop that cannot end
NYGAARD = "Nygaard, Rødland & Hovig 2016"
ZHANG = "Zhang et al. 2018"
HORNUNG = "Hornung et al. 2017"
JOHNSON = "Johnson, Li & Rabinovic 2007"
METHODS = ("covariate", "reference_combat", "outcome_combat", "none", "not_a_batch")


# ── ComBat: sva::ComBat, parametric empirical Bayes ──────────────────────────


def _rowvar(a: np.ndarray, axis: int) -> np.ndarray:
    return np.var(a, axis=axis, ddof=1)


def _aprior(delta_hat: np.ndarray) -> float:
    m, s2 = float(np.mean(delta_hat)), float(np.var(delta_hat, ddof=1))
    return (2 * s2 + m ** 2) / s2


def _bprior(delta_hat: np.ndarray) -> float:
    m, s2 = float(np.mean(delta_hat)), float(np.var(delta_hat, ddof=1))
    return (m * s2 + m ** 3) / s2


def _postmean(g_hat: Any, g_bar: float, n: Any, d_star: Any, t2: float) -> np.ndarray:
    return (t2 * n * g_hat + d_star * g_bar) / (t2 * n + d_star)


def _postvar(sum2: np.ndarray, n: Any, a: float, b: float) -> np.ndarray:
    return (0.5 * sum2 + b) / (n / 2 + a - 1)


def _it_sol(sdat: np.ndarray, g_hat: np.ndarray, d_hat: np.ndarray, g_bar: float, t2: float,
            a: float, b: float, conv: float = CONV) -> tuple[np.ndarray, np.ndarray]:
    """sva's ``it.sol``, samples in rows: alternate the posterior mean and variance until the
    largest relative change (as sva computes it, with the signed denominator) is below ``conv``."""
    n = np.sum(np.isfinite(sdat), axis=0).astype(float)
    g_old, d_old = g_hat.copy(), d_hat.copy()
    change, count = 1.0, 0
    while change > conv:
        g_new = _postmean(g_hat, g_bar, n, d_old, t2)
        sum2 = np.nansum((sdat - g_new[None, :]) ** 2, axis=0)
        d_new = _postvar(sum2, n, a, b)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.r_[np.abs(g_new - g_old) / g_old, np.abs(d_new - d_old) / d_old]
        # sva divides by the signed old value; an exact 0/0 (no change from zero) is no change
        ratios = ratios[~np.isnan(ratios)]
        change = float(np.max(ratios)) if ratios.size else 0.0
        count += 1
        if count > MAX_ITERATIONS:
            raise ValueError("ComBat's empirical-Bayes iteration did not converge")
        g_old, d_old = g_new, d_new
    return g_new, d_new


@dataclass
class ComBatFit:
    """Everything ComBat estimated, so rows it never saw can be adjusted the same way."""

    levels: list[str]
    ref: int | None  # index into levels of the reference batch
    mean_only: bool
    keep: np.ndarray  # features adjusted (False: uniform within some batch, left unchanged)
    B_hat: np.ndarray  # design coefficients × kept features
    n_batch: int
    grand_mean: np.ndarray
    var_pooled: np.ndarray
    gamma_star: np.ndarray  # batch × kept features
    delta_star: np.ndarray
    mod_columns: int  # covariate columns (after the batch indicators) in the design
    mod_keep: np.ndarray | None = None  # which of ``mod``'s columns the design kept

    def batch_index(self, labels: Sequence[Any]) -> np.ndarray:
        where = {b: i for i, b in enumerate(self.levels)}
        return np.array([where.get(_label(v), -1) for v in labels], dtype=int)


def _label(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "(missing)"
    if isinstance(value, (int, float, np.integer, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _levels(batch: Sequence[Any]) -> list[str]:
    """R's factor levels for a character batch: sorted. Labels are compared as text."""
    return sorted(dict.fromkeys(_label(b) for b in batch))


def combat_fit(values: np.ndarray, batch: Sequence[Any], mod: np.ndarray | None = None,
               ref_batch: Any = None, mean_only: bool = False, conv: float = CONV) -> ComBatFit:
    """``sva::ComBat(dat, batch, mod, par.prior = TRUE, mean.only, ref.batch)`` on ``values``
    (samples in rows, features in columns), returning its estimates.

    Line by line from sva 3.60: features that are uniform within a batch of more than one sample
    are left unchanged; one batch of one sample makes it mean-only; with a reference batch its
    indicator becomes the intercept, the grand mean is its mean and the pooled variance is its
    residual variance (divided by its size), and its rows are returned unchanged; otherwise the
    grand mean weights the batches by size. Covariates in ``mod`` (an intercept column, as
    ``model.matrix`` writes it, is dropped) are kept in the standardized mean.
    """
    dat = np.asarray(values, dtype=float)
    if not np.isfinite(dat).all():
        raise ValueError("ComBat here needs complete values: fill blanks before the batch step.")
    labels = [_label(b) for b in batch]
    levels = _levels(labels)
    n, p = dat.shape
    idx = np.array([levels.index(b) for b in labels])
    batches = [np.flatnonzero(idx == i) for i in range(len(levels))]
    n_batches = np.array([len(b) for b in batches])
    zero = np.zeros(p, dtype=bool)
    for rows in batches:
        if len(rows) > 1:
            zero |= _rowvar(dat[rows], 0) == 0
    keep = ~zero
    dat = dat[:, keep]
    if np.any(n_batches == 1):
        mean_only = True
    n_batch = len(levels)
    batchmod = np.zeros((n, n_batch))
    batchmod[np.arange(n), idx] = 1.0
    ref = None
    if ref_batch is not None:
        if _label(ref_batch) not in levels:
            raise ValueError("the reference batch is not one of the batches")
        ref = levels.index(_label(ref_batch))
        batchmod[:, ref] = 1.0
    design = batchmod if mod is None else np.column_stack([batchmod, np.asarray(mod, dtype=float)])
    check = np.all(design == 1, axis=0)
    if ref is not None:
        check[ref] = False
    mod_keep = None if mod is None else ~check[n_batch:]
    design = design[:, ~check]
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("A covariate is confounded with batch: ComBat cannot separate them.")
    B_hat = np.linalg.solve(design.T @ design, design.T @ dat)
    if ref is not None:
        grand_mean = B_hat[ref]
        rows = batches[ref]
        resid = dat[rows] - design[rows] @ B_hat
        var_pooled = np.mean(resid ** 2, axis=0)
    else:
        grand_mean = (n_batches / n) @ B_hat[:n_batch]
        var_pooled = np.mean((dat - design @ B_hat) ** 2, axis=0)
    stand_mean = np.tile(grand_mean, (n, 1))
    tmp = design.copy()
    tmp[:, :n_batch] = 0.0
    stand_mean = stand_mean + tmp @ B_hat
    s_data = (dat - stand_mean) / np.sqrt(var_pooled)
    batch_design = design[:, :n_batch]
    gamma_hat = np.linalg.solve(batch_design.T @ batch_design, batch_design.T @ s_data)
    delta_hat = np.array([np.ones(dat.shape[1]) if mean_only else _rowvar(s_data[rows], 0)
                          for rows in batches])
    gamma_bar = gamma_hat.mean(axis=1)
    t2 = _rowvar(gamma_hat, 1)
    gamma_star = np.empty_like(gamma_hat)
    delta_star = np.empty_like(gamma_hat)
    for i, rows in enumerate(batches):
        if i == ref:
            continue  # sva computes the reference batch's estimates and then sets them to 0 and 1
        if mean_only:
            gamma_star[i] = _postmean(gamma_hat[i], gamma_bar[i], 1.0, 1.0, t2[i])
            delta_star[i] = 1.0
        else:
            a, b = _aprior(delta_hat[i]), _bprior(delta_hat[i])
            gamma_star[i], delta_star[i] = _it_sol(s_data[rows], gamma_hat[i], delta_hat[i],
                                                   gamma_bar[i], t2[i], a, b, conv)
    if ref is not None:
        gamma_star[ref] = 0.0
        delta_star[ref] = 1.0
    return ComBatFit(levels=levels, ref=ref, mean_only=mean_only, keep=keep, B_hat=B_hat,
                     n_batch=n_batch, grand_mean=grand_mean, var_pooled=var_pooled,
                     gamma_star=gamma_star, delta_star=delta_star,
                     mod_columns=design.shape[1] - n_batch, mod_keep=mod_keep)


def combat_apply(fit: ComBatFit, values: np.ndarray, batch: Sequence[Any],
                 mod: np.ndarray | None = None) -> np.ndarray:
    """The adjusted values of rows from batches ``fit`` estimated, by its frozen estimates (sva's
    "Adjusting the Data" step); a reference-batch row comes back unchanged. A row of a batch the
    fit never saw raises: :func:`add_on` adjusts those."""
    dat = np.asarray(values, dtype=float)
    out = dat.copy()
    kept = dat[:, fit.keep]
    idx = fit.batch_index(batch)
    if np.any(idx < 0):
        raise ValueError("a row's batch was not in the rows ComBat was fit on")
    n = len(kept)
    stand_mean = np.tile(fit.grand_mean, (n, 1))
    if fit.mod_columns:
        if mod is None:
            raise ValueError("this ComBat fit has covariates; give their values")
        covariates = np.asarray(mod, dtype=float)[:, fit.mod_keep]
        stand_mean = stand_mean + covariates @ fit.B_hat[fit.n_batch:]
    s = (kept - stand_mean) / np.sqrt(fit.var_pooled)
    offset = fit.gamma_star[idx]
    if fit.ref is not None:
        offset = offset + fit.gamma_star[fit.ref]  # the reference column is every row's intercept
    adjusted = (s - offset) / np.sqrt(fit.delta_star[idx])
    adjusted = adjusted * np.sqrt(fit.var_pooled) + stand_mean
    if fit.ref is not None:
        at_ref = idx == fit.ref
        adjusted[at_ref] = kept[at_ref]
    out[:, fit.keep] = adjusted
    return out


def add_on(fit: ComBatFit, values: np.ndarray, conv: float = CONV) -> np.ndarray:
    """A new batch's rows adjusted to the fit's reference (no covariates): standardized by the
    frozen grand mean and pooled variance, its own location and scale estimated and shrunk across
    features as ComBat shrinks each batch, then adjusted. The fit's rows never move (Hornung et
    al. 2017: add-on batch-effect removal)."""
    dat = np.asarray(values, dtype=float)
    out = dat.copy()
    kept = dat[:, fit.keep]
    s = (kept - fit.grand_mean) / np.sqrt(fit.var_pooled)
    g_hat = s.mean(axis=0)
    g_bar, t2 = float(g_hat.mean()), float(np.var(g_hat, ddof=1)) if g_hat.size > 1 else 0.0
    if len(kept) < 2 or fit.mean_only or t2 <= 0:
        g_star = _postmean(g_hat, g_bar, 1.0, 1.0, t2) if t2 > 0 else g_hat
        d_star = np.ones_like(g_hat)
    else:
        d_hat = _rowvar(s, 0)
        g_star, d_star = _it_sol(s, g_hat, d_hat, g_bar, t2, _aprior(d_hat), _bprior(d_hat), conv)
    adjusted = (s - g_star) / np.sqrt(d_star) * np.sqrt(fit.var_pooled) + fit.grand_mean
    out[:, fit.keep] = adjusted
    return out


def combat(values: np.ndarray, batch: Sequence[Any], mod: np.ndarray | None = None,
           ref_batch: Any = None, mean_only: bool = False) -> np.ndarray:
    """``sva::ComBat``'s adjusted data (samples in rows)."""
    fit = combat_fit(values, batch, mod=mod, ref_batch=ref_batch, mean_only=mean_only)
    return combat_apply(fit, values, batch, mod)


# ── the in-fold step ─────────────────────────────────────────────────────────


def reference_level(batch: Sequence[Any]) -> str:
    """The reference batch: the largest among the rows given (ties: the first in sorted order)."""
    counts = pd.Series([_label(b) for b in batch]).value_counts()
    top = counts[counts == counts.max()].index
    return sorted(str(t) for t in top)[0]


class ReferenceComBat(TransformerMixin, BaseEstimator):
    """ComBat with a reference batch, fitted on the rows it is given and never on the outcome.

    ``columns`` are the features adjusted, ``batch`` the column naming each row's batch. Fit picks
    the largest batch of the fitting rows as the reference and runs :func:`combat_fit` there (no
    covariates: the outcome is never in its model, and ``y`` is ignored); transform adjusts each
    row of a fitted batch by the frozen estimates (the reference batch unchanged), and the rows of
    a batch the fit never saw by :func:`add_on`. ``drop`` takes the batch column out of the
    output (when it is not itself a predictor)."""

    def __init__(self, columns: Sequence[str] = (), batch: str = "", drop: bool = True):
        self.columns = columns
        self.batch = batch
        self.drop = drop

    def _block(self, X: pd.DataFrame) -> tuple[list[str], np.ndarray]:
        cols = [c for c in self.columns if c in X.columns]
        return cols, X[cols].to_numpy(dtype=float, na_value=np.nan)

    def fit(self, X: pd.DataFrame, y: Any = None) -> "ReferenceComBat":
        self.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)
        self.n_features_in_ = X.shape[1]
        cols, values = self._block(X)
        labels = X[self.batch].tolist()
        self.reference_ = reference_level(labels)
        self.fitted_columns_ = cols
        self.fit_ = combat_fit(values, labels, ref_batch=self.reference_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "fit_"):
            raise ValueError("ReferenceComBat is not fitted yet.")
        cols = self.fitted_columns_
        values = X[cols].to_numpy(dtype=float, na_value=np.nan)
        labels = np.asarray([_label(v) for v in X[self.batch].tolist()], dtype=object)
        out = values.copy()
        known = np.isin(labels, self.fit_.levels)
        if known.any():
            out[known] = combat_apply(self.fit_, values[known], labels[known])
        for b in sorted(set(labels[~known].tolist())):
            rows = labels == b
            out[rows] = add_on(self.fit_, values[rows])
        result = X.copy()
        result[cols] = out
        if self.drop and self.batch in result.columns:
            result = result.drop(columns=[self.batch])
        return result

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        names = [c for c in self.feature_names_in_ if not (self.drop and c == self.batch)]
        return np.asarray(names, dtype=object)

    def lineage(self) -> list[dict[str, Any]]:
        block = set(self.fitted_columns_)
        return [{"output": c, "inputs": [c, self.batch] if c in block else [c],
                 "operation": "batch-adjusted (reference ComBat)" if c in block else "kept"}
                for c in self.get_feature_names_out()]


# ── what the data say about the batch: descriptive scope ─────────────────────


@dataclass(frozen=True)
class Confounding:
    perfect: bool
    sentence: str
    table: dict[str, dict[str, int]]  # batch -> outcome level -> rows (categorical outcomes)
    cramers_v: float | None


def confounding(batch: Sequence[Any], outcome: Sequence[Any], task: str | None,
                batch_name: str = "batch", outcome_name: str = "the outcome") -> Confounding | None:
    """Whether ``batch`` is perfectly confounded with ``outcome`` over every row with both.

    Categorical outcomes: perfect when every batch holds a single outcome level (and there are at
    least two levels); Cramér's V measures the imbalance short of that. A numeric outcome: perfect
    when it is constant within every batch and differs between them. Reads every row because it
    answers "can this data separate batch from outcome at all?", which informs no modeling choice
    but whether any estimate is possible (BLUEPRINT §13, descriptive scope)."""
    b = pd.Series([_label(v) for v in batch], dtype=object)
    yv = pd.Series(list(outcome), dtype=object)
    ok = yv.notna().to_numpy() & (b != "(missing)").to_numpy()
    b, yv = b[ok].reset_index(drop=True), yv[ok].reset_index(drop=True)
    if b.nunique() < 2 or len(b) < 2:
        return None
    if task in (None, "regression") and pd.api.types.is_numeric_dtype(pd.to_numeric(yv, errors="coerce")) \
            and pd.to_numeric(yv, errors="coerce").nunique() > 10:
        y_num = pd.to_numeric(yv, errors="coerce")
        within = y_num.groupby(b).nunique()
        perfect = bool((within <= 1).all() and y_num.groupby(b).first().nunique() > 1)
        said = (f"`{outcome_name}` is constant within each `{batch_name}` and differs between them: "
                f"batch and outcome cannot be told apart." if perfect else "")
        return Confounding(perfect, said, {}, None)
    y_lab = yv.map(_label)
    table = pd.crosstab(b, y_lab)
    perfect = bool(y_lab.nunique() >= 2 and ((table > 0).sum(axis=1) == 1).all())
    from scipy.stats import chi2_contingency

    v = None
    if table.shape[0] > 1 and table.shape[1] > 1:
        chi2 = float(chi2_contingency(table.to_numpy(), correction=False)[0])
        v = math.sqrt(chi2 / (len(b) * (min(table.shape) - 1)))
    said = ""
    if perfect:
        pairs = ", ".join(f"`{i}` holds only `{table.columns[(table.loc[i] > 0).to_numpy()][0]}`"
                          for i in table.index[:4])
        more = " …" if len(table.index) > 4 else ""
        said = (f"Every `{batch_name}` holds one level of `{outcome_name}` ({pairs}{more}): batch "
                f"and outcome cannot be told apart, so no model can estimate either.")
    counts = {str(i): {str(c): int(table.loc[i, c]) for c in table.columns} for i in table.index}
    return Confounding(perfect, said, counts, v)


def imbalance_concern(found: Confounding | None, batch_name: str, outcome_name: str) -> str | None:
    """Nygaard et al.'s warning, stated when outcome groups are unevenly spread over batches."""
    if found is None or found.perfect or found.cramers_v is None or found.cramers_v < 0.1:
        return None
    return (f"`{outcome_name}` is unevenly spread over `{batch_name}` (Cramér's V "
            f"{found.cramers_v:.2f}): batch adjustment without the outcome deflates group "
            f"differences, and with the outcome protected it inflates them ({NYGAARD}); batch as a "
            f"covariate in the outcome model avoids both.")



# ── what the design reads ────────────────────────────────────────────────────


def _spec(state: Any) -> Any:
    return getattr(state, "batch", None)


def batch_inputs(state: Any) -> list[str]:
    """Columns the pipeline takes for the batch step though no model sees them: the batch column
    under reference ComBat (dropped after the step unless it is also a predictor)."""
    spec = _spec(state)
    if spec is None or spec.method != "reference_combat":
        return []
    return [spec.column]


def design_batch(state: Any, inputs: Sequence[str], predictors: Sequence[str],
                 normalization: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """``{column, method, columns, drop}`` for the in-fold batch step, or None. The features it
    adjusts are the normalized assay columns among the inputs, else the settled exposures."""
    spec = _spec(state)
    if spec is None or spec.method != "reference_combat" or spec.column not in set(inputs):
        return None
    if normalization and normalization.get("columns"):
        columns = [c for c in normalization["columns"] if c in set(inputs)]
    else:
        from turbotab.core.readings import settled_roles

        roles = settled_roles(state)
        columns = [c for c in predictors if roles.get(c) == "exposure" and c != spec.column]
    if not columns:
        return None
    return {"column": spec.column, "method": "reference_combat", "columns": columns,
            "drop": spec.column not in set(predictors)}


def describe_step(spec: Mapping[str, Any]) -> tuple[str, str]:
    n = len(spec.get("columns") or [])
    return ("Remove batch effects (reference ComBat)",
            f"Aligns {n:,} assay column{'s' if n != 1 else ''} across the levels of "
            f"`{spec.get('column')}` by ComBat with a reference batch, the largest batch of each "
            f"training fold, fitted without the outcome; held-out rows are adjusted by the fold's "
            f"estimates, and a batch absent from the fold by add-on adjustment.")


# ── the batch finding: descriptive scope ─────────────────────────────────────

FINDING = "batch_confounding"


def batch_columns(frame: pd.DataFrame, target: str | None) -> list[str]:
    """Columns named as a batch or a plate (the recognizer's whole-word acquisition reading). A
    name is a guess: what it starts is a question, never a number."""
    from turbotab.core.recognizers import acquisition_kind

    return [str(c) for c in frame.columns
            if str(c) != target and acquisition_kind(str(c)) in ("batch", "plate")
            and frame[c].nunique(dropna=True) >= 2]


def batch_findings(frame: pd.DataFrame, lenses: Sequence[str] | None,
                   target: str | None) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """For each batch column, whether it can be told apart from the outcome over every row: a
    critical finding when it is perfectly confounded (no model can separate them), a warning when
    the outcome is unevenly spread over the batches (Nygaard et al.'s case)."""
    if target is None or target not in frame.columns:
        return []
    columns = batch_columns(frame, target)
    # A batch name is read as a batch only where the table is an assay, or records how samples were
    # acquired in a second way (``stages.rows.acquisition_corroborated``: in a plate-size feeding
    # study `plate` is the intervention).
    from turbotab.core.recognizers import acquisition_kind

    kinds = {acquisition_kind(str(c)) for c in frame.columns} - {None}
    if not any(k in ("genomics", "metabolomics") for k in (lenses or [])) and len(kinds) < 2:
        return []
    out = []
    for column in columns:
        found = confounding(frame[column].tolist(), frame[target].tolist(), None, column, target)
        if found is None:
            continue
        concern = imbalance_concern(found, column, target)
        if not found.perfect and concern is None:
            continue
        fid = f"{FINDING}__{column}"
        if found.perfect:
            finding = {
                "id": fid, "severity": "critical",
                "title": f"`{column}` is perfectly confounded with `{target}`.",
                "detail": found.sentence,
                "why_it_matters": ("A batch effect and a real difference look the same here, so "
                                   "every estimate would be of both at once. No model is fitted "
                                   "until this is answered: choose another outcome, or say the "
                                   "column is not a batch."),
                "summary": f"Every `{column}` holds one level of `{target}`.",
            }
        else:
            finding = {
                "id": fid, "severity": "warning",
                "title": f"`{target}` is unevenly spread over `{column}`.",
                "detail": concern,
                "why_it_matters": ("How batch is handled changes the group differences: as a "
                                   "covariate under inference, by ComBat without the outcome in "
                                   "each training fold under prediction."),
                "summary": f"`{target}` is unevenly spread over `{column}` (Cramér's V "
                           f"{found.cramers_v:.2f}).",
            }
        finding.update({"affected_columns": [column, target], "source": "structural", "lens": None,
                        "evidence": None, "routes_to": "models", "lever_label": "Say how batch is handled",
                        "group": None})
        params = {"column": column, "perfect": found.perfect, "cramers_v": found.cramers_v,
                  "table": found.table}
        out.append(({"params": params, "confidence": "high"}, finding))
    return out


def _confounded(ctx: Any) -> dict[str, dict[str, Any]]:
    """Batch columns the findings read as perfectly confounded with the outcome: column -> params."""
    reader = getattr(ctx, "artifact", None) if not isinstance(ctx, Mapping) else ctx.get("artifact")
    if not callable(reader):
        return {}
    try:
        artifact = reader("findings")
    except Exception:  # noqa: BLE001 - no findings yet: nothing read as confounded
        return {}
    data = getattr(artifact, "data", artifact)
    out = {}
    for f in (data or {}).get("findings") or [] if isinstance(data, Mapping) else []:
        if str(f.get("id", "")).startswith(FINDING + "__") and f.get("severity") == "critical":
            column = str(f["id"]).split("__", 1)[1].split("#")[0]
            out[column] = dict(f)
    return out


def _not_a_batch(state: Any, column: str) -> bool:
    spec = _spec(state)
    return spec is not None and spec.column == column and spec.method == "not_a_batch"


def confounding_refusal(column: str, target: str | None) -> Any:
    from turbotab.core.decisions import Refusal

    return Refusal(
        "batch_confounded_with_outcome",
        f"`{column}` is perfectly confounded with `{target}`: every batch holds one outcome level, "
        f"so batch and outcome cannot be told apart and no model can estimate either. This is "
        f"refused under both purposes.",
        exits=[{"label": "Choose another outcome", "decision": None},
               {"label": f"`{column}` is not a batch: read it as an ordinary column",
                "decision": {"kind": "set_batch", "column": column, "method": "not_a_batch"}}])


def _no_model_while_confounded(decision: Any, ctx: Any) -> None:
    from turbotab.core.decisions import _state

    state = _state(ctx)
    for column in _confounded(ctx):
        if not _not_a_batch(state, column):
            raise confounding_refusal(column, getattr(state, "target", None))


def design_refusal(state: Any, store: Any, rows: Any, task: str | None) -> str | None:
    """The design stage's backstop: a batch column perfectly confounded with the outcome on the
    rows the design describes, unless the user said it is not a batch."""
    target = getattr(state, "target", None)
    if target is None or target not in store.columns:
        return None
    spec = _spec(state)
    candidates = [c for c in batch_columns_of(store.columns, target)]
    if spec is not None and spec.method != "not_a_batch" and spec.column in store.columns:
        candidates = list(dict.fromkeys([spec.column, *candidates]))
    for column in candidates:
        if _not_a_batch(state, column):
            continue
        frame = store.materialize([column, target], rows)
        found = confounding(frame[column].tolist(), frame[target].tolist(), task, column, target)
        if found is not None and found.perfect:
            return confounding_refusal(column, target).message
    return None


def batch_columns_of(columns: Sequence[str], target: str | None) -> list[str]:
    from turbotab.core.recognizers import acquisition_kind

    return [str(c) for c in columns if str(c) != target
            and acquisition_kind(str(c)) in ("batch", "plate")]


# ── the leash on the batch answer ────────────────────────────────────────────


def _batch_answer_fits(decision: Any, ctx: Any) -> None:
    """MODELING_SEQUENCE §2 and §4 for batch, by purpose:

    * perfectly confounded with the outcome: refused under both purposes;
    * ComBat with the outcome protected: refused under prediction (it reads held-out outcomes) and
      for testing under inference (figures only);
    * as a covariate: the column must be a settled covariate whose levels are categories, asked
      in one block when it is not (BLUEPRINT §14.2)."""
    from turbotab.core.decisions import Refusal, _columns_of, _state

    state = _state(ctx)
    columns = _columns_of(ctx)
    target = getattr(state, "target", None)
    if columns is not None and decision.column not in columns:
        raise Refusal("unknown_column", f"This dataset has no column named `{decision.column}`.",
                      exits=[{"label": "Choose one of the dataset's columns", "decision": None}])
    if decision.column == target:
        raise Refusal("batch_is_the_outcome", f"`{decision.column}` is the outcome; a batch is a "
                      f"column the samples were measured in.",
                      exits=[{"label": "Choose the batch column", "decision": None}])
    if decision.method == "not_a_batch":
        return
    if decision.column in _confounded(ctx):
        raise confounding_refusal(decision.column, target)
    purpose = getattr(state, "purpose", None)
    if decision.method == "outcome_combat":
        if purpose == "inference":
            raise Refusal(
                "outcome_combat_for_testing",
                f"ComBat with the outcome protected inflates group differences when groups are "
                f"spread unevenly over batches ({NYGAARD}), so its values are refused for testing. "
                f"Include `{decision.column}` as a covariate in the tests; ComBat may serve the "
                f"figures.",
                exits=[{"label": "Batch as a covariate; ComBat for figures only",
                        "decision": {"kind": "set_batch", "column": decision.column,
                                     "method": "covariate", "figures": True}},
                       {"label": "Batch as a covariate",
                        "decision": {"kind": "set_batch", "column": decision.column,
                                     "method": "covariate"}}])
        raise Refusal(
            "outcome_combat_leaks",
            f"ComBat with the outcome protected reads every row's outcome, the held-out rows' "
            f"included, so the score would see what it predicts (leakage). Fit ComBat in each "
            f"training fold without the outcome, with a reference batch ({ZHANG}).",
            exits=[{"label": "Reference ComBat in each training fold, without the outcome",
                    "decision": {"kind": "set_batch", "column": decision.column,
                                 "method": "reference_combat"}}])
    if decision.figures and purpose != "inference":
        raise Refusal("figures_under_prediction",
                      "ComBat for figures only is an inference answer: under prediction the "
                      "values the model sees are the ones a figure should show.",
                      exits=[{"label": "Reference ComBat in each training fold, without the outcome",
                              "decision": {"kind": "set_batch", "column": decision.column,
                                           "method": "reference_combat"}}])
    if decision.method == "covariate":
        from turbotab.core.readings import settled_roles

        roles = settled_roles(state) if state is not None else {}
        items = []
        if roles.get(decision.column) != "covariate":
            items.append({"reading": "role", "column": decision.column, "value": "covariate"})
        info = (_column_info(ctx) or {}).get(decision.column)
        dtype = (info.get("dtype") if isinstance(info, Mapping) else getattr(info, "dtype", None)) if info else None
        if dtype in ("numeric", "integer"):
            from turbotab.core.readings import confirmation

            if confirmation(state, "code_or_count", decision.column) != "code":
                items.append({"reading": "code_or_count", "column": decision.column, "value": "code"})
        if items:
            raise Refusal(
                "batch_covariate_unsettled",
                f"As a covariate, `{decision.column}` enters the outcome model as categories, one "
                f"indicator per batch. Confirm that first.",
                exits=[{"label": "Confirm it as a categorical covariate",
                        "decision": {"kind": "confirm_readings", "items": items}}])


def _column_info(ctx: Any) -> Any:
    return getattr(ctx, "column_info", None) if not isinstance(ctx, Mapping) else ctx.get("column_info")


def batch_sentence(column: str, method: str, figures: bool = False) -> str:
    """The record's sentence for a batch answer."""
    if method == "covariate":
        text = (f"`{column}` enters the outcome model as a covariate, so batch differences are not "
                f"read as biology ({NYGAARD})")
        if figures:
            text += "; ComBat with the outcome protected serves the figures only, never the tests"
        return text
    if method == "reference_combat":
        return (f"Batch effects across `{column}` are removed by ComBat with a reference batch, the "
                f"largest batch of each training fold, fitted without the outcome; held-out rows are "
                f"adjusted by the fold's estimates ({ZHANG}), and a batch absent from the fold by "
                f"add-on adjustment ({HORNUNG})")
    if method == "none":
        return f"`{column}` is left as it is: any batch effect stays in the values"
    if method == "not_a_batch":
        return f"`{column}` is not a batch; it is read as an ordinary column"
    return f"`{column}`: ComBat with the outcome protected"


def methods_clause(run: Mapping[str, Any]) -> str | None:
    """The methods paragraph's clause for the batch answer (``run["batch"]``)."""
    spec = run.get("batch") or {}
    method = spec.get("method")
    if method == "covariate":
        text = "batch was included as a covariate"
        if spec.get("figures"):
            text += "; ComBat was used for visualization only"
        return text
    if method == "reference_combat":
        return ("batch effects were removed by ComBat with a reference batch, fitted within each "
                "training fold without the outcome")
    return None


def _register() -> None:
    from turbotab.core.decisions import register_validator
    from turbotab.core.methods.contract import (CONTRACTS, ContractOption, MethodContract, Relation,
                                                register_contract)
    from turbotab.core.voice import register_sentence

    register_validator("set_batch", _batch_answer_fits)
    register_validator("select_models", _no_model_while_confounded)

    @register_sentence("set_batch")
    def _set_batch(d: Any, state: Any, ctx: Any) -> str:
        return batch_sentence(d.column, d.method, d.figures)

    if "batch" in CONTRACTS:
        return
    both = ("prediction", "inference")

    def opt(key: str, label: str, customary: str, sound: tuple[str, str], rung: tuple[str, str],
            order: tuple[int, int]) -> ContractOption:
        return ContractOption(key, label, customary, dict(zip(both, sound)), dict(zip(both, rung)),
                              dict(zip(both, order)))

    register_contract(MethodContract(
        key="batch", label="Batch handling", slot="in_fold", scope="training_fold", run_order=5.0,
        needs=("a batch column",), question="The samples were measured in batches: how is batch handled?",
        options=(
            opt("covariate", "As a covariate",
                "Recommended practice: batch in the design (limma, DESeq2)",
                ("Ranked lower: a new sample's batch must be one the model saw.",
                 f"Sound, first: account for batch in the analysis ({NYGAARD})."),
                ("rank_lower", "recommended"), (1, 0)),
            opt("reference_combat", "Reference ComBat in-fold",
                "Customary: ComBat before analysis is widespread",
                (f"Sound, first: fitted per fold without the outcome, frozen for new rows ({ZHANG}).",
                 f"Ranked lower: without the outcome it deflates group differences under imbalance "
                 f"({NYGAARD})."), ("recommended", "rank_lower"), (0, 1)),
            opt("outcome_combat", "ComBat, outcome protected", "Customary, and widespread",
                ("Refused: it reads held-out outcomes (leakage).",
                 f"Refused for testing; figures only ({NYGAARD})."), ("refused", "refused"), (3, 3)),
            opt("none", "Leave it", "Common when batch is ignored",
                ("Ranked lower: batch effects stay in the values.",
                 "Ranked lower: batch differences can read as biology."),
                ("rank_lower", "rank_lower"), (2, 2)),
        ),
        storyboard=("Standardize each feature", "Estimate each batch's location and scale",
                    "Shrink them across features", "Align every batch to the reference"),
        relations=(
            Relation("conflicts", "perfect_confounding",
                     "Batch perfectly confounded with the outcome is refused under both purposes.",
                     rung="refused"),
            Relation("conflicts", "testing",
                     "Outcome-protected ComBat is refused for testing; figures only.",
                     purposes=("inference",), rung="refused", when=("outcome_combat",)),
            Relation("conflicts", "leakage",
                     "Outcome-protected ComBat is refused under prediction: leakage.",
                     purposes=("prediction",), rung="refused", when=("outcome_combat",)),
            Relation("precedes", "screen", "Batch correction precedes in-fold screening.",
                     when=("reference_combat",)),
        ),
        sources=(JOHNSON, NYGAARD, ZHANG, HORNUNG),
        short="reference-batch ComBat without the outcome",
        clause=methods_clause,
        option_slots={"covariate": "model"},
        option_scopes={"covariate": "model", "outcome_combat": "model"},
    ))


_register()


__all__ = [
    "CONV", "ComBatFit", "Confounding", "FINDING", "HORNUNG", "JOHNSON", "METHODS", "NYGAARD",
    "ReferenceComBat", "ZHANG", "add_on", "batch_columns", "batch_findings", "batch_inputs",
    "batch_sentence", "combat", "combat_apply", "combat_fit", "confounding", "confounding_refusal",
    "describe_step", "design_batch", "design_refusal", "imbalance_concern", "methods_clause",
    "reference_level",
]

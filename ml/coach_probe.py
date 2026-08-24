"""Evidence probe: the model coach's empirical eyes.

Static shape (n, p, EPV) tells the coach what the data *permits*; a probe
tells it what the data *contains*. This module runs three cheap, seeded
measurements on TRAINING ROWS ONLY and returns effect sizes the coach can
cite:

1. Signal floor — cross-validated score of a penalized linear model vs the
   same model on permuted targets. Answers "is there any learnable signal
   above chance?" before the user invests in modeling.
2. Non-linearity gain — shallow gradient-boosted trees vs the linear model,
   same folds. Answers "will complexity pay?" with a measured delta rather
   than a prior.
3. Learning-curve slope — the linear model on half vs all training rows.
   Answers "would more data help more than more models?"

Honesty contract:
- The probe must ONLY ever see training rows (callers apply the lockbox
  mask). It never touches the held-out test set.
- Probe numbers are advisory diagnostics, not reportable results: they are
  cross-validation on training data with aggressive subsampling. They are
  never exported to the manuscript as performance claims — only as
  model-selection rationale.
- Everything is seeded and time-bounded: rows are subsampled to
  MAX_ROWS, features to the MAX_FEATURES highest-variance columns.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

MAX_ROWS = 1500
MAX_FEATURES = 100
MAX_FEATURES_WIDE = 2000
N_FOLDS = 3
N_PERMUTATIONS = 3
SEED = 42

# Interpretation thresholds (score points: R² for regression, AUC for
# classification). Deliberately coarse — the probe informs advice, it does
# not adjudicate close calls.
SIGNAL_MARGIN = 0.05      # above the permuted band → real signal
NONLINEAR_GAIN = 0.04     # trees beat linear by this → complexity pays
LEARNING_SLOPE = 0.03     # full-data beats half-data by this → data-hungry


@dataclass
class ProbeResult:
    """Measured evidence from the training rows. All scores are CV means."""
    task_type: str
    n_rows_used: int
    n_features_used: int
    subsampled: bool

    linear_score: float = float("nan")     # R² (regression) or AUC (classification)
    permuted_score: float = float("nan")   # same model, shuffled targets (mean)
    permuted_spread: float = 0.0           # std across permutations
    tree_score: float = float("nan")       # shallow HistGB, same folds
    half_data_score: float = float("nan")  # linear model on 50% of rows

    metric_name: str = "R²"
    # Whether the folds were grouped by a recorded entity (`AUDIT-009`).
    grouped: bool = False
    notes: List[str] = field(default_factory=list)

    # ── verdicts the coach consumes ──
    @property
    def has_signal(self) -> Optional[bool]:
        if np.isnan(self.linear_score) and np.isnan(self.tree_score):
            return None
        best = np.nanmax([self.linear_score, self.tree_score])
        # Beating a (possibly very negative) permuted baseline is not enough:
        # R² below 0 is worse than predicting the mean, AUC below 0.5 is
        # worse than coin-flipping. Signal must clear absolute chance too.
        chance_floor = {"R²": 0.0, "AUC": 0.5}.get(self.metric_name, self.permuted_score)
        threshold = max(chance_floor + SIGNAL_MARGIN,
                        self.permuted_score + max(SIGNAL_MARGIN, 2 * self.permuted_spread))
        return bool(best > threshold)

    @property
    def nonlinearity_gain(self) -> Optional[float]:
        if np.isnan(self.tree_score) or np.isnan(self.linear_score):
            return None
        return float(self.tree_score - self.linear_score)

    @property
    def data_hungry(self) -> Optional[bool]:
        if np.isnan(self.half_data_score) or np.isnan(self.linear_score):
            return None
        return bool(self.linear_score - self.half_data_score > LEARNING_SLOPE)

    @property
    def underpowered(self) -> bool:
        """At very small n the probe cannot separate weak signal from noise —
        an honest 'cannot tell' beats a false 'no signal'."""
        return self.n_rows_used < 60

    def summary(self) -> str:
        """One-line advisory summary in the coach's voice."""
        parts = []
        if self.has_signal is False:
            best = float(np.nanmax([self.linear_score, self.tree_score]))
            chance_floor = {"R²": 0.0, "AUC": 0.5}.get(self.metric_name, self.permuted_score)
            if self.underpowered and best > chance_floor:
                parts.append(
                    f"signal unconfirmed — at n={self.n_rows_used} the probe is "
                    f"underpowered (best {self.metric_name} = {best:.2f}, within "
                    f"the noise band)")
            else:
                parts.append(
                    f"little signal above chance (best probe {self.metric_name} = "
                    f"{best:.2f}; permuted baseline {self.permuted_score:.2f})")
        elif self.has_signal:
            parts.append(f"learnable signal (probe {self.metric_name} ≈ "
                         f"{np.nanmax([self.linear_score, self.tree_score]):.2f})")
        gain = self.nonlinearity_gain
        if gain is not None and self.has_signal:
            if gain > NONLINEAR_GAIN:
                parts.append(f"non-linear structure (+{gain:.2f} {self.metric_name} for trees)")
            elif gain < NONLINEAR_GAIN / 2:
                parts.append("linear ≈ trees in the probe")
        if self.data_hungry and self.has_signal:
            parts.append("scores still rising with more rows")
        return "; ".join(parts) if parts else "probe inconclusive"


def _prepare(X, y, task_type, rng, groups=None):
    """Numeric-only, imputed, subsampled copies for probing.

    `groups` rides through every mask and subsample the rows do, because a
    grouping that stops matching its rows is worse than none: it would put a
    fold boundary somewhere nobody chose.
    """
    import pandas as pd

    Xd = X.select_dtypes(include=[np.number]).copy() if hasattr(X, "select_dtypes") else pd.DataFrame(X)
    y = np.asarray(y)
    g = None if groups is None else np.asarray(groups)
    mask = np.isfinite(y) if task_type == "regression" else ~pd.isna(y)
    mask = np.asarray(mask)
    Xd, y = Xd.loc[mask], y[mask]
    if g is not None:
        g = g[mask]

    notes = []
    subsampled = False
    if len(Xd) > MAX_ROWS:
        idx = rng.choice(len(Xd), MAX_ROWS, replace=False)
        Xd, y = Xd.iloc[idx], y[idx]
        if g is not None:
            g = g[idx]
        subsampled = True
        notes.append(f"subsampled to {MAX_ROWS:,} rows")
    wide = Xd.shape[1] > len(Xd)
    feature_cap = MAX_FEATURES_WIDE if wide else MAX_FEATURES
    if Xd.shape[1] > feature_cap:
        variances = Xd.var(numeric_only=True).fillna(0)
        keep = variances.nlargest(feature_cap).index
        Xd = Xd[keep]
        subsampled = True
        notes.append(f"top {feature_cap} features by variance")

    Xv = Xd.fillna(Xd.median(numeric_only=True)).fillna(0).to_numpy(dtype=float)
    return Xv, y, notes, subsampled, g


def _linear_pipeline(task_type, wide: bool = False):
    from sklearn.linear_model import LassoCV, LogisticRegression, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if wide:
        # p >> n: ridge on a variance-screened subset is signal-blind (the
        # screen never sees y, and may drop the informative columns). L1
        # models handle p > n natively — and mirror what the coach actually
        # recommends on this shape.
        if task_type == "regression":
            import inspect
            # sklearn renamed n_alphas → alphas (int); support both
            _params = inspect.signature(LassoCV.__init__).parameters
            _alpha_kw = {"n_alphas": 20} if "n_alphas" in _params else {"alphas": 20}
            est = LassoCV(cv=3, random_state=SEED, **_alpha_kw)
        else:
            est = LogisticRegression(penalty="l1", solver="liblinear",
                                     C=0.5, max_iter=500)
    else:
        est = (Ridge(alpha=1.0) if task_type == "regression"
               else LogisticRegression(max_iter=500, C=1.0))
    return Pipeline([("scale", StandardScaler()), ("model", est)])


def _tree_estimator(task_type):
    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                  HistGradientBoostingRegressor)

    kw = dict(max_depth=3, max_iter=60, random_state=SEED)
    return (HistGradientBoostingRegressor(**kw) if task_type == "regression"
            else HistGradientBoostingClassifier(**kw))


def _cv_score(est, X, y, task_type, groups=None):
    """The probe's cross-validated score, grouped where an entity is recorded.

    **`AUDIT-002`'s class, on the surface that ranks the shortlist**
    (`AUDIT-009`). This built `KFold`/`StratifiedKFold` with `shuffle=True` and
    no groups, and the mean it returns is what the Model Coach ranks its picks
    on — so on any table with repeated measures one person's rows sat on both
    sides of every fold and the ranking was computed on a number it should not
    have trusted.

    `ml/eval.py:113` documents the exact hazard and selects
    `StratifiedGroupKFold`; `ml/splits.py` implements the whole priority order.
    Neither was reachable from here, because this function had no `groups`
    parameter to reach one with — which is why the fix is a signature change
    rather than a call swap, and why it stayed open through two audits.

    `groups` is the recorded entity per row, or `None`. **`None` means the
    caller did not record one**, never *there is no grouping*: the scheme falls
    back to the ungrouped one and `run_probe` says so in its notes.
    """
    from sklearn.model_selection import (GroupKFold, KFold, StratifiedGroupKFold,
                                         StratifiedKFold, cross_val_score)

    grouped = groups is not None
    if task_type == "regression":
        n_splits = N_FOLDS
        if grouped:
            n_groups = int(len(np.unique(groups)))
            if n_groups < 2:
                return float("nan")
            n_splits = int(min(N_FOLDS, n_groups))
            cv = GroupKFold(n_splits=n_splits)
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        scoring = "r2"
    else:
        _, counts = np.unique(y, return_counts=True)
        n_splits = int(min(N_FOLDS, counts.min()))
        if grouped:
            n_splits = int(min(n_splits, len(np.unique(groups))))
        if n_splits < 2:
            return float("nan")
        cv = (StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                   random_state=SEED) if grouped
              else StratifiedKFold(n_splits=n_splits, shuffle=True,
                                   random_state=SEED))
        scoring = "roc_auc" if len(counts) == 2 else "accuracy"
    try:
        return float(np.mean(cross_val_score(est, X, y, cv=cv, groups=groups,
                                             scoring=scoring)))
    except Exception:
        return float("nan")


def run_probe(X_train, y_train, task_type: str = "regression",
              groups=None) -> ProbeResult:
    """Run the evidence probe on TRAINING data only (caller enforces the
    lockbox mask). Deterministic, a few seconds on typical shapes.

    `groups` is the recorded entity per row. Passing it makes every fold
    grouped, so no entity spans a split — `AUDIT-009`. Omitting it scores the
    ungrouped way and SAYS SO in the notes, because a number computed one way
    and read as the other is the whole defect.
    """
    rng = np.random.default_rng(SEED)
    X, y, notes, subsampled, groups = _prepare(X_train, y_train, task_type,
                                               rng, groups)

    metric = "R²" if task_type == "regression" else (
        "AUC" if len(np.unique(y)) == 2 else "accuracy")
    result = ProbeResult(
        task_type=task_type, n_rows_used=len(X), n_features_used=X.shape[1],
        subsampled=subsampled, metric_name=metric, notes=notes,
        grouped=groups is not None,
    )
    # SAID WHERE THE READER IS, not only in the return value. A grouped score
    # is usually LOWER than the leaked one, and a user who is not told reads
    # the drop as a regression rather than as the app becoming correct.
    if groups is not None:
        result.notes.append(
            f"folds grouped by the recorded entity — {len(np.unique(groups)):,} "
            f"of them — so no entity spans a split. An ungrouped score on a "
            f"table with repeated measures comes out higher than it should")
    else:
        result.notes.append(
            "folds are ungrouped: no entity column was passed, so if a row "
            "can repeat within a subject these numbers are optimistic")
    if len(X) < 20 or X.shape[1] == 0:
        result.notes.append("too little data to probe")
        return result

    wide = X.shape[1] > len(X)
    linear = _linear_pipeline(task_type, wide=wide)
    result.linear_score = _cv_score(linear, X, y, task_type, groups)

    perm_scores = []
    for i in range(N_PERMUTATIONS):
        y_perm = rng.permutation(y)
        perm_scores.append(_cv_score(_linear_pipeline(task_type, wide=wide),
                                     X, y_perm, task_type, groups))
    perm_scores = [s for s in perm_scores if not np.isnan(s)]
    if perm_scores:
        result.permuted_score = float(np.mean(perm_scores))
        result.permuted_spread = float(np.std(perm_scores))

    result.tree_score = _cv_score(_tree_estimator(task_type), X, y, task_type,
                                  groups)

    if len(X) >= 80:
        half = rng.choice(len(X), len(X) // 2, replace=False)
        result.half_data_score = _cv_score(
            _linear_pipeline(task_type, wide=wide), X[half], y[half], task_type,
            None if groups is None else groups[half])

    return result

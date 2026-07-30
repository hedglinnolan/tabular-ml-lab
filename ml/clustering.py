"""
K-means cluster exploration for the EDA page.

k-means is a partitioning algorithm, not a detector. Handed uniform noise, a
single Gaussian blob, or genuine subgroups, it returns exactly k non-empty
clusters and all three outputs look identical. Nothing the fit prints is, by
itself, evidence that groups exist. Everything in this module past the fit
exists to supply the evidence the fit cannot:

  * a permutation null — refit on column-shuffled copies that keep every
    marginal distribution intact but destroy every relationship between
    variables, so the user sees what their silhouette would have been on data
    with no structure at all;
  * seed stability — the same k refit under different random starts, scored by
    adjusted Rand index, because a fixed random_state buys reproducibility, not
    stability, and showing one seed hides the difference;
  * single-feature dominance — if one column explains the partition, the user
    re-derived a column they already had.

Preprocessing notes, since they change the answer more than k does:
  * Scaling is mandatory, not a preference. The k-means objective is not
    invariant to per-feature rescaling, so unscaled mixed-unit data clusters on
    unit choices (age in years vs days, glucose in mg/dL vs mmol/L).
  * Heavy skew is transformed before scaling. Under z-scoring a long right tail
    becomes a handful of points at +8 SD, and an objective that minimizes
    squared distance will spend a whole cluster on them.
  * One-hot columns are NOT standardized. Raw dummies give each categorical
    variable a total squared-distance contribution of 2(1 - sum p_l^2) <= 2,
    which is at most one standardized numeric feature. Standardizing them
    instead multiplies a level of prevalence p by 1/sqrt(p(1-p)), so a 1%-
    prevalence level lands its dozen rows ~10 SD from everyone else and k-means
    awards them their own "subtype". Rare levels are collapsed for the same
    reason.

Contract mirrors ml/macro_shape.py: compute_* returns a plain dict carrying an
"error" key instead of raising, plot_* is pure and returns a plotly Figure.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, RobustScaler, StandardScaler


# --- Budget knobs ----------------------------------------------------------
# The EDA page may be handed a 100k-row frame. Structure that shows up in
# 100k rows but not in a random 10k of them is not structure, so we sweep on a
# fixed subsample and say so on screen rather than hiding it.
MAX_FIT_ROWS = 10_000
MAX_SILHOUETTE_ROWS = 4_000      # silhouette is O(n^2); sklearn's own sample_size knob
MAX_NULL_SILHOUETTE_ROWS = 2_000  # inside the null loop we need the shape, not decimals
MAX_CLUSTER_FEATURES = 60
MINIBATCH_ABOVE = 20_000
MAX_CATEGORICAL_LEVELS = 15
RARE_LEVEL_FLOOR = 0.01           # levels under 1% collapse into "Other"
MAX_MISSING_FRACTION = 0.40
SKEW_TRANSFORM_ABOVE = 2.0
SEED = 42

# How far above the shuffled baseline a silhouette has to sit before this tool
# will call it structure. Both conditions must hold: an absolute margin, so a
# hair's-breadth win on noise is not promoted, and a margin wider than the
# baseline's own spread, so the win is not inside the noise of the comparison.
MIN_MEANINGFUL_EXCESS = 0.05
MIN_EXCESS_SD_MULTIPLE = 2.0

# Silhouette reading anchors. Deliberately unflattering — these are read
# against the permutation null, never as absolute quality scores.
SILHOUETTE_BANDS = (
    (0.50, "reasonable separation"),
    (0.25, "weak, overlapping structure"),
    (-1.0, "no substantial structure"),
)


def _estimator(k: int, n_rows: int, n_init: int, seed: int):
    """Full Lloyd below the mini-batch threshold, MiniBatch above it.

    n_init is set explicitly everywhere: sklearn's 'auto' resolves to 1 restart
    with k-means++, which is the most likely source of "my clusters changed
    when I reran it".
    """
    if n_rows > MINIBATCH_ABOVE:
        return MiniBatchKMeans(
            n_clusters=k, init="k-means++", n_init=max(3, n_init // 3),
            batch_size=2048, max_no_improvement=20, random_state=seed,
        )
    return KMeans(
        n_clusters=k, init="k-means++", n_init=n_init,
        max_iter=300, tol=1e-4, algorithm="lloyd", random_state=seed,
    )


def _silhouette(X: np.ndarray, labels: np.ndarray, cap: int, seed: int = SEED) -> float:
    """Silhouette, subsampled above `cap` because the exact form is O(n^2)."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    kwargs = {}
    if len(X) > cap:
        kwargs = {"sample_size": cap, "random_state": seed}
    try:
        return float(silhouette_score(X, labels, **kwargs))
    except ValueError:
        return float("nan")


def silhouette_reading(value: float) -> str:
    """Plain-English band for a silhouette score."""
    if not np.isfinite(value):
        return "not computable"
    for floor, label in SILHOUETTE_BANDS:
        if value >= floor:
            return label
    return "no substantial structure"


# ---------------------------------------------------------------------------
# Matrix preparation
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def prepare_cluster_matrix(
    _df: pd.DataFrame,
    features: Sequence[str],
    scaler: str = "standard",
    categorical_weight: float = 1.0,
    max_rows: int = MAX_FIT_ROWS,
    data_id: Any = None,
) -> Dict[str, Any]:
    """Build the scaled matrix k-means will actually see.

    Returns a dict with the matrix, the row index it corresponds to, the
    per-column provenance, and every decision made along the way so the UI can
    show its work.
    """
    features = [f for f in features if f in _df.columns]
    if not features:
        return {"error": "No features selected for clustering."}

    frame = _df[features]

    # Row subsample first, so every downstream cost is bounded.
    sampled = len(frame) > max_rows
    if sampled:
        frame = frame.sample(max_rows, random_state=SEED)
    row_index = frame.index

    dropped: List[str] = []
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []

    for col in features:
        series = frame[col]
        if series.isna().mean() > MAX_MISSING_FRACTION:
            dropped.append(f"{col} (>{int(MAX_MISSING_FRACTION * 100)}% missing)")
            continue
        if pd.api.types.is_numeric_dtype(series):
            if series.nunique(dropna=True) < 2:
                dropped.append(f"{col} (constant)")
                continue
            numeric_cols.append(col)
        else:
            n_levels = series.nunique(dropna=True)
            if n_levels < 2:
                dropped.append(f"{col} (constant)")
            elif n_levels > MAX_CATEGORICAL_LEVELS * 2:
                dropped.append(f"{col} ({n_levels} levels — too many to encode)")
            else:
                categorical_cols.append(col)

    if not numeric_cols and not categorical_cols:
        return {"error": "No usable features left after dropping constant, sparse, and high-cardinality columns."}

    blocks: List[np.ndarray] = []
    column_names: List[str] = []
    transformed: List[str] = []
    # Column spans of each ORIGINAL variable. The permutation null shuffles by
    # span, not by column: shuffling the columns of a one-hot block
    # independently produces rows with two 1s or none for the same variable,
    # which is not a null of "no structure" — it is a null of "not even valid
    # data", and it scores artificially badly.
    variable_spans: List[Tuple[int, int]] = []

    # -- Numeric: median impute -> tame skew -> scale -------------------------
    if numeric_cols:
        num = frame[numeric_cols].astype(float)
        medians = num.median()
        num = num.fillna(medians).fillna(0.0)

        skews = num.skew().abs()
        skewed = [c for c in numeric_cols if np.isfinite(skews.get(c, 0.0)) and skews[c] > SKEW_TRANSFORM_ABOVE]
        if skewed:
            try:
                pt = PowerTransformer(method="yeo-johnson", standardize=False)
                num[skewed] = pt.fit_transform(num[skewed])
                transformed = list(skewed)
            except Exception:
                transformed = []

        scaler_obj = RobustScaler() if scaler == "robust" else StandardScaler()
        num_scaled = scaler_obj.fit_transform(num.values)
        num_scaled = np.nan_to_num(num_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        blocks.append(num_scaled)
        variable_spans.extend((len(column_names) + i, len(column_names) + i + 1) for i in range(len(numeric_cols)))
        column_names.extend(numeric_cols)

    # -- Categorical: collapse rare levels -> one-hot, deliberately unscaled --
    encoded_levels: Dict[str, int] = {}
    if categorical_cols:
        cat = frame[categorical_cols].astype("object").where(frame[categorical_cols].notna(), "Missing")
        cat = cat.astype(str)
        try:
            enc = OneHotEncoder(
                drop="if_binary",
                min_frequency=RARE_LEVEL_FLOOR,
                handle_unknown="infrequent_if_exist",
                sparse_output=False,
            )
            cat_mat = enc.fit_transform(cat)
            names = list(enc.get_feature_names_out(categorical_cols))
        except TypeError:
            # Older sklearn: no min_frequency / sparse_output spelling.
            enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
            cat_mat = enc.fit_transform(cat)
            names = list(enc.get_feature_names_out(categorical_cols))

        cursor = len(column_names)
        for col in categorical_cols:
            width = sum(1 for n in names if n.startswith(f"{col}_"))
            encoded_levels[col] = width
            if width:
                variable_spans.append((cursor, cursor + width))
                cursor += width

        # No standardization here — see the module docstring.
        blocks.append(cat_mat.astype(float) * float(categorical_weight))
        column_names.extend(names)

    X = np.hstack(blocks) if len(blocks) > 1 else blocks[0]
    X = np.ascontiguousarray(X, dtype=np.float64)

    # Share of total variance carried by the categorical block, so the
    # numeric/categorical balance is a visible decision rather than an accident.
    cat_share = None
    if categorical_cols and numeric_cols:
        variances = X.var(axis=0)
        total = float(variances.sum())
        n_num = len(numeric_cols)
        cat_share = float(variances[n_num:].sum() / total) if total > 0 else None

    return {
        "X": X,
        "variable_spans": tuple(variable_spans),
        "row_index": row_index,
        "column_names": column_names,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "encoded_levels": encoded_levels,
        "dropped": dropped,
        "skew_transformed": transformed,
        "scaler": scaler,
        "sampled": sampled,
        "n_rows": int(X.shape[0]),
        "n_source_rows": int(len(_df)),
        "effective_p": int(X.shape[1]),
        "categorical_variance_share": cat_share,
    }


# ---------------------------------------------------------------------------
# k sweep + permutation null
# ---------------------------------------------------------------------------

def max_supported_k(n_rows: int, hard_cap: int = 8) -> int:
    """Cap k by power, not by taste.

    Cluster-analysis power simulations put the floor near 20-30 observations per
    subgroup even when separation is large, so k is capped at n/30.
    """
    return int(max(2, min(hard_cap, n_rows // 30)))


@st.cache_data(show_spinner=False)
def sweep_k(
    X: np.ndarray,
    k_values: Tuple[int, ...],
    variable_spans: Tuple[Tuple[int, int], ...] = (),
    n_init: int = 10,
    null_reps: int = 8,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Fit every k, score it, and score the same k on shuffled copies.

    The shuffled copies preserve each variable's marginal distribution exactly —
    its skew, its zero-inflation, its category prevalences — while destroying
    every relationship between variables. Whatever silhouette they earn is what
    this pipeline scores on data with no structure in it.

    Shuffling is per VARIABLE, not per column: a categorical variable's one-hot
    columns move together so every shuffled row is still a valid encoding.
    """
    if X.shape[0] < 10:
        return {"error": "Too few rows to cluster."}

    spans = list(variable_spans) or [(j, j + 1) for j in range(X.shape[1])]
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []

    for k in k_values:
        if k >= X.shape[0]:
            continue
        model = _estimator(k, X.shape[0], n_init, seed)
        labels = model.fit_predict(X)
        sizes = np.bincount(labels, minlength=k)

        observed = _silhouette(X, labels, MAX_SILHOUETTE_ROWS, seed)

        null_scores: List[float] = []
        for rep in range(null_reps):
            shuffled = X.copy()
            for start, end in spans:
                shuffled[:, start:end] = shuffled[rng.permutation(X.shape[0]), start:end]
            null_labels = _estimator(k, shuffled.shape[0], 3, seed + rep).fit_predict(shuffled)
            null_scores.append(_silhouette(shuffled, null_labels, MAX_NULL_SILHOUETTE_ROWS, seed))

        null_scores = [s for s in null_scores if np.isfinite(s)]
        null_mean = float(np.mean(null_scores)) if null_scores else float("nan")
        null_sd = float(np.std(null_scores)) if len(null_scores) > 1 else 0.0
        # Empirical p: how often noise beat the real data at this k.
        if null_scores and np.isfinite(observed):
            p_value = (1 + sum(1 for s in null_scores if s >= observed)) / (1 + len(null_scores))
        else:
            p_value = float("nan")

        rows.append({
            "k": int(k),
            "silhouette": observed,
            "null_silhouette": null_mean,
            "null_sd": null_sd,
            "excess": observed - null_mean if np.isfinite(observed) and np.isfinite(null_mean) else float("nan"),
            "p_value": p_value,
            "calinski_harabasz": float(calinski_harabasz_score(X, labels)) if len(np.unique(labels)) > 1 else float("nan"),
            "davies_bouldin": float(davies_bouldin_score(X, labels)) if len(np.unique(labels)) > 1 else float("nan"),
            "inertia": float(getattr(model, "inertia_", np.nan)),
            "min_cluster_size": int(sizes.min()),
        })

    if not rows:
        return {"error": "No valid k values for this dataset."}

    # Recommend the k with the largest margin over its own null, but only if
    # that margin is both meaningful and wider than the baseline's own spread.
    # Otherwise recommend nothing — "there are no clusters" has to be an answer
    # this tool can give, and it is the correct answer far more often than the
    # published literature suggests.
    best = None
    for row in rows:
        if not np.isfinite(row["excess"]):
            continue
        if row["excess"] < MIN_MEANINGFUL_EXCESS:
            continue
        if row["excess"] <= MIN_EXCESS_SD_MULTIPLE * row["null_sd"]:
            continue
        if best is None or row["excess"] > best["excess"]:
            best = row

    return {
        "table": rows,
        "recommended_k": int(best["k"]) if best else None,
        "recommended_excess": float(best["excess"]) if best else None,
        "null_reps": null_reps,
    }


# ---------------------------------------------------------------------------
# Final fit + honesty checks
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def fit_clusters(X: np.ndarray, k: int, n_init: int = 25, seed: int = SEED) -> Dict[str, Any]:
    """Fit the chosen k properly and score every row's silhouette."""
    if k < 2 or k >= X.shape[0]:
        return {"error": f"k={k} is not valid for {X.shape[0]} rows."}

    model = _estimator(k, X.shape[0], n_init, seed)
    labels = model.fit_predict(X)
    sizes = np.bincount(labels, minlength=k)

    sample_idx = np.arange(X.shape[0])
    if X.shape[0] > MAX_SILHOUETTE_ROWS:
        sample_idx = np.random.default_rng(seed).choice(X.shape[0], MAX_SILHOUETTE_ROWS, replace=False)
        sample_idx.sort()
    try:
        per_row = silhouette_samples(X[sample_idx], labels[sample_idx])
    except ValueError:
        per_row = np.full(len(sample_idx), np.nan)

    return {
        "labels": labels,
        "centroids": np.asarray(model.cluster_centers_),
        "sizes": sizes.tolist(),
        "silhouette": float(np.nanmean(per_row)) if len(per_row) else float("nan"),
        "silhouette_samples": per_row,
        "silhouette_index": sample_idx,
        "inertia": float(getattr(model, "inertia_", np.nan)),
        "k": int(k),
    }


@st.cache_data(show_spinner=False)
def seed_stability(X: np.ndarray, k: int, n_seeds: int = 10, seed: int = SEED) -> Dict[str, Any]:
    """Refit under different random starts and score agreement by adjusted Rand.

    A fixed random_state buys reproducibility. It does not buy stability, and
    reporting one seed hides the difference.
    """
    if k < 2 or X.shape[0] < 20:
        return {"error": "Too few rows for a stability check."}

    labelings = [
        _estimator(k, X.shape[0], 10, seed + i).fit_predict(X)
        for i in range(n_seeds)
    ]
    scores = [
        adjusted_rand_score(labelings[i], labelings[j])
        for i in range(len(labelings))
        for j in range(i + 1, len(labelings))
    ]
    if not scores:
        return {"error": "Stability check produced no comparisons."}

    mean_ari = float(np.mean(scores))
    if mean_ari >= 0.75:
        verdict = "stable"
    elif mean_ari >= 0.50:
        verdict = "suggestive"
    else:
        verdict = "unstable"
    return {
        "mean_ari": mean_ari,
        "min_ari": float(np.min(scores)),
        "n_seeds": n_seeds,
        "verdict": verdict,
    }


@st.cache_data(show_spinner=False)
def feature_dominance(
    _df: pd.DataFrame,
    labels: np.ndarray,
    _row_index: pd.Index,
    features: Sequence[str],
    top_n: int = 5,
    data_id: Any = None,
) -> Dict[str, Any]:
    """How much of each raw feature the partition accounts for, on a 0-1 scale.

    If one column is almost fully explained by the split, the user re-derived a
    column they already had rather than discovering a subgroup.

    Numeric columns are scored by the correlation ratio (between-cluster sum of
    squares over total), categoricals by Cramer's V. Both read as "share of this
    column the split accounts for". An earlier version binned numerics into
    quartiles and used adjusted mutual information, which capped the score well
    below 1 for exactly the tightly-grouped column this check exists to catch.

    `labels` carries the cache key: it changes whenever the matrix, the config,
    or k changes, which is exactly when this result goes stale.
    """
    from scipy import stats

    aligned = _df.loc[_row_index]
    scored: List[Dict[str, Any]] = []

    for col in features:
        if col not in aligned.columns:
            continue
        series = aligned[col]
        valid = series.notna().values
        if valid.sum() < 5:
            continue
        # A near-unique CATEGORICAL column determines any partition perfectly,
        # so Cramer's V returns 1.0 on nothing. It is an identifier, not an
        # explanation. Numeric columns are naturally near-unique and are scored
        # by the correlation ratio, which does not have this failure.
        if (
            not pd.api.types.is_numeric_dtype(series)
            and series.nunique(dropna=True) > 0.5 * len(series)
        ):
            continue
        lab = np.asarray(labels)[valid]
        if len(np.unique(lab)) < 2:
            continue
        try:
            if pd.api.types.is_numeric_dtype(series):
                values = series[valid].astype(float).values
                grand = values.mean()
                ss_total = float(((values - grand) ** 2).sum())
                if ss_total <= 0:
                    continue
                ss_between = float(sum(
                    (lab == c).sum() * (values[lab == c].mean() - grand) ** 2
                    for c in np.unique(lab)
                ))
                score = ss_between / ss_total
                metric = "correlation ratio"
            else:
                crosstab = pd.crosstab(lab, series[valid].astype(str).values)
                if min(crosstab.shape) < 2:
                    continue
                chi2 = float(stats.chi2_contingency(crosstab)[0])
                n = int(crosstab.values.sum())
                score = float(np.sqrt(chi2 / (n * (min(crosstab.shape) - 1))))
                metric = "Cramer's V"
        except Exception:
            continue
        scored.append({
            "Feature": col,
            "Explained by split": round(min(max(score, 0.0), 1.0), 3),
            "Measure": metric,
        })

    if not scored:
        return {"error": "Could not score feature dominance."}

    scored.sort(key=lambda r: r["Explained by split"], reverse=True)
    top = scored[:top_n]

    # "Dominant" means one column carries the split BY ITSELF. Several columns
    # all scoring high is the opposite finding — that is multivariate structure,
    # and flagging it as a re-labeling would be wrong.
    dominant = None
    if top and top[0]["Explained by split"] >= 0.80:
        runner_up = top[1]["Explained by split"] if len(top) > 1 else 0.0
        if runner_up <= 0.50:
            dominant = top[0]["Feature"]

    return {
        "table": top,
        "dominant": dominant,
        "dominant_score": top[0]["Explained by split"] if top else None,
    }


@st.cache_data(show_spinner=False)
def cluster_profile(
    _df: pd.DataFrame,
    labels: np.ndarray,
    _row_index: pd.Index,
    numeric_features: Sequence[str],
    top_n: int = 15,
    data_id: Any = None,
) -> Dict[str, Any]:
    """Per-cluster means in z-units, ranked by between-cluster separation.

    Reported in z-units of the raw column so the heatmap reads directly:
    "cluster 2 sits +1.8 SD on CRP". Ranking is by the between-cluster F ratio
    so the columns shown are the ones that actually separate the clusters.
    """
    cols = [c for c in numeric_features if c in _df.columns]
    if not cols:
        return {"error": "No numeric features available to profile."}

    aligned = _df.loc[_row_index, cols].astype(float)
    means = aligned.mean()
    stds = aligned.std(ddof=0).replace(0, np.nan)
    z = (aligned - means) / stds

    z = z.assign(_cluster=labels)
    centroid_z = z.groupby("_cluster")[cols].mean()

    # Between-cluster F ratio per column: how much of the variance the
    # partition explains.
    f_scores: Dict[str, float] = {}
    for col in cols:
        series = z[col]
        grand = series.mean()
        between, within = 0.0, 0.0
        for cluster_id, group in series.groupby(z["_cluster"]):
            g = group.dropna()
            if len(g) < 2:
                continue
            between += len(g) * (g.mean() - grand) ** 2
            within += ((g - g.mean()) ** 2).sum()
        f_scores[col] = float(between / within) if within > 0 else 0.0

    ranked = sorted(cols, key=lambda c: f_scores.get(c, 0.0), reverse=True)[:top_n]
    return {
        "centroids_z": centroid_z[ranked].round(3),
        "ranked_features": ranked,
        "f_scores": {c: round(f_scores.get(c, 0.0), 3) for c in ranked},
    }


@st.cache_data(show_spinner=False)
def target_association(
    _df: pd.DataFrame,
    labels: np.ndarray,
    _row_index: pd.Index,
    target_col: str,
    task_type: str,
    data_id: Any = None,
) -> Dict[str, Any]:
    """Test the clusters against the target — the one non-circular check here.

    Testing whether clusters differ on the variables they were built from is
    circular and returns p ~ 0 by construction. The target is held out of the
    clustering by design, so this is the one association in the block that is
    actually evidence.
    """
    from scipy import stats

    if target_col not in _df.columns:
        return {"error": "Target not available."}

    y = _df.loc[_row_index, target_col]
    frame = pd.DataFrame({"cluster": labels, "target": y.values}).dropna()
    if len(frame) < 10 or frame["cluster"].nunique() < 2:
        return {"error": "Not enough complete rows to test against the target."}

    groups = [g["target"].values for _, g in frame.groupby("cluster") if len(g) > 1]
    if len(groups) < 2:
        return {"error": "Not enough clusters with data to test."}

    if task_type == "classification" or not pd.api.types.is_numeric_dtype(frame["target"]):
        table = pd.crosstab(frame["cluster"], frame["target"])
        try:
            chi2, p, _, _ = stats.chi2_contingency(table)
            n = int(table.values.sum())
            min_dim = min(table.shape) - 1
            cramers_v = float(np.sqrt(chi2 / (n * min_dim))) if n > 0 and min_dim > 0 else float("nan")
        except Exception:
            return {"error": "Chi-square test could not be computed."}
        proportions = table.div(table.sum(axis=1), axis=0).round(3)
        # Class labels become column names here. Left as-is they are ints
        # beside the string "cluster", and a mixed-type column index does not
        # round-trip through Arrow when Streamlit renders it.
        proportions.columns = [f"{target_col} = {c}" for c in proportions.columns]
        proportions.index.name = "Cluster"
        return {
            "kind": "classification",
            "table": proportions.reset_index(),
            "p_value": float(p),
            "effect": cramers_v,
            "effect_name": "Cramer's V",
        }

    try:
        h_stat, p = stats.kruskal(*groups)
    except Exception:
        return {"error": "Kruskal-Wallis test could not be computed."}

    grand = frame["target"].mean()
    ss_total = float(((frame["target"] - grand) ** 2).sum())
    ss_between = float(sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups))
    eta_sq = ss_between / ss_total if ss_total > 0 else float("nan")

    summary = frame.groupby("cluster")["target"].agg(["count", "mean", "median", "std"]).round(3)
    summary.columns = ["n", "Mean", "Median", "SD"]
    return {
        "kind": "regression",
        "table": summary.reset_index(),
        "values": frame,
        "p_value": float(p),
        "effect": float(eta_sq),
        "effect_name": "eta-squared",
    }


@st.cache_data(show_spinner=False)
def project_for_display(X: np.ndarray, seed: int = SEED) -> Dict[str, Any]:
    """Two-component PCA of the clustering matrix, for display only.

    Linear, and chosen without reference to the labels, so it cannot manufacture
    the separation it is used to show. The variance-explained figures are the
    interpretive key, not decoration.
    """
    if X.shape[1] < 2:
        return {"error": "Need at least 2 columns to project."}
    n_comp = min(2, X.shape[1], X.shape[0])
    pca = PCA(n_components=n_comp, random_state=seed)
    coords = pca.fit_transform(X)
    return {
        "coords": coords,
        "explained": pca.explained_variance_ratio_.tolist(),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

_CLUSTER_PALETTE = px.colors.qualitative.Set2


def cluster_colors(k: int) -> List[str]:
    """One colorblind-friendly color per cluster, stable for a given k."""
    return [_CLUSTER_PALETTE[i % len(_CLUSTER_PALETTE)] for i in range(k)]


def plot_k_sweep(sweep: Dict[str, Any]) -> go.Figure:
    """Silhouette across k with the permutation-null band shaded behind it."""
    rows = sweep["table"]
    ks = [r["k"] for r in rows]
    observed = [r["silhouette"] for r in rows]
    null_mean = [r["null_silhouette"] for r in rows]
    null_hi = [r["null_silhouette"] + r["null_sd"] for r in rows]
    null_lo = [r["null_silhouette"] - r["null_sd"] for r in rows]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ks + ks[::-1], y=null_hi + null_lo[::-1],
        fill="toself", fillcolor="rgba(148,163,184,0.25)",
        line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
        name="Shuffled data (±1 SD)",
    ))
    fig.add_trace(go.Scatter(
        x=ks, y=null_mean, mode="lines",
        line=dict(color="#94a3b8", dash="dash", width=2),
        name="Shuffled data (no structure)",
    ))
    fig.add_trace(go.Scatter(
        x=ks, y=observed, mode="lines+markers",
        line=dict(color="#667eea", width=3), marker=dict(size=9),
        name="Your data",
    ))
    fig.update_layout(
        template="plotly_white", height=380,
        title="Silhouette vs number of clusters, against a no-structure baseline",
        xaxis_title="k (number of clusters)", yaxis_title="Mean silhouette",
        xaxis=dict(tickmode="array", tickvals=ks),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def plot_cluster_scatter(
    coords: np.ndarray,
    labels: np.ndarray,
    explained: Sequence[float],
) -> go.Figure:
    """PCA projection colored by cluster."""
    k = int(labels.max()) + 1
    palette = cluster_colors(k)
    fig = go.Figure()
    for cluster_id in range(k):
        mask = labels == cluster_id
        fig.add_trace(go.Scatter(
            x=coords[mask, 0], y=coords[mask, 1],
            mode="markers", name=f"Cluster {cluster_id}",
            marker=dict(size=6, color=palette[cluster_id], opacity=0.6),
        ))
    x_label = f"PC1 ({explained[0]:.1%} of variance)" if len(explained) > 0 else "PC1"
    y_label = f"PC2 ({explained[1]:.1%} of variance)" if len(explained) > 1 else "PC2"
    fig.update_layout(
        template="plotly_white", height=450,
        title="Clusters in the first two principal components",
        xaxis_title=x_label, yaxis_title=y_label,
    )
    return fig


def plot_silhouette_knife(
    per_row: np.ndarray,
    labels: np.ndarray,
    mean_silhouette: float,
) -> go.Figure:
    """Per-row silhouette, sorted within cluster — what the mean conceals."""
    k = int(labels.max()) + 1
    palette = cluster_colors(k)
    fig = go.Figure()
    y_cursor = 0
    for cluster_id in range(k):
        values = np.sort(per_row[labels == cluster_id])
        if len(values) == 0:
            continue
        y_positions = np.arange(y_cursor, y_cursor + len(values))
        fig.add_trace(go.Bar(
            x=values, y=y_positions, orientation="h",
            marker=dict(color=palette[cluster_id], line=dict(width=0)),
            name=f"Cluster {cluster_id}", hovertemplate="silhouette %{x:.3f}<extra></extra>",
        ))
        y_cursor += len(values) + max(5, len(per_row) // 50)

    if np.isfinite(mean_silhouette):
        fig.add_vline(
            x=mean_silhouette, line=dict(color="#ef4444", dash="dash"),
            annotation_text=f"mean {mean_silhouette:.2f}", annotation_position="top",
        )
    fig.add_vline(x=0, line=dict(color="#94a3b8", width=1))
    fig.update_layout(
        template="plotly_white", height=420, bargap=0,
        title="Silhouette of every row, grouped by cluster",
        xaxis_title="Silhouette coefficient",
        yaxis=dict(showticklabels=False, title=""),
    )
    return fig


def plot_cluster_profile(centroids_z: pd.DataFrame) -> go.Figure:
    """Cluster centroids in z-units — the plot that gives a cluster a name."""
    limit = float(np.nanmax(np.abs(centroids_z.values))) if centroids_z.size else 1.0
    limit = max(limit, 0.5)
    fig = px.imshow(
        centroids_z.values,
        x=list(centroids_z.columns),
        y=[f"Cluster {i}" for i in centroids_z.index],
        color_continuous_scale="RdBu_r",
        zmin=-limit, zmax=limit,
        aspect="auto",
        labels=dict(color="SD from<br>overall mean"),
    )
    fig.update_layout(
        template="plotly_white",
        height=max(280, len(centroids_z) * 46 + 160),
        title="Cluster centroids, in standard deviations from the overall mean",
        xaxis_title="", yaxis_title="",
    )
    fig.update_xaxes(tickangle=-40)
    return fig

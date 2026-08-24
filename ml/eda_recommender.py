"""
EDA Recommendation System for Medical/Nutritional Tabular Data.
Generates contextual recommendations based on dataset signals.
"""
import pandas as pd
from pandas.api import types as _pdt
import numpy as np
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass, field
from ml.clinical_units import infer_unit
from ml.physiology_reference import load_reference_bundle, match_variable_key, get_improbability_band
from ml.outliers import detect_outliers
from scipy import stats
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif


@dataclass
class DatasetSignals:
    """Computed signals from the dataset for EDA recommendations."""
    n_rows: int
    n_cols: int
    numeric_cols: List[str] = field(default_factory=list)
    categorical_cols: List[str] = field(default_factory=list)
    datetime_cols: List[str] = field(default_factory=list)
    text_like_cols: List[str] = field(default_factory=list)
    missing_rate_by_col: Dict[str, float] = field(default_factory=dict)
    high_missing_cols: List[str] = field(default_factory=list)
    high_cardinality_categoricals: List[str] = field(default_factory=list)
    duplicate_row_rate: float = 0.0
    target_name: Optional[str] = None
    task_type_final: Optional[Literal["regression", "classification"]] = None
    cohort_type_final: Optional[Literal["cross_sectional", "longitudinal"]] = None
    entity_id_final: Optional[str] = None
    target_stats: Dict = field(default_factory=dict)
    leakage_flags: List[str] = field(default_factory=list)
    leakage_candidate_cols: List[str] = field(default_factory=list)
    # `MINE-004`. Empty leakage lists mean "the scan found nothing" ONLY if the
    # scan ran. This carries the reason it did not, so a failure and a clean
    # dataset stop being the same downstream signal.
    leakage_scan_error: str = ""
    collinearity_summary: Dict = field(default_factory=dict)
    physio_plausibility_flags: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class EDARecommendation:
    """A single EDA recommendation card."""
    id: str
    title: str
    priority: int  # Lower = higher priority
    cost: Literal["low", "medium", "high"]
    why: List[str]  # Concrete triggered reasons
    what_you_learn: List[str]
    model_implications: List[str]
    run_action: str  # Name of analysis function to call
    description: Optional[str] = None  # Plain-language explanation
    enabled: bool = True
    disabled_reason: Optional[str] = None


# ── AUDIT-003: what the values say before a log transform is advised ────────
#
# `research/METABOLOMICS_PACK.md`, "Value-state diagnostics":
#
#   "Already-transformed detection. Any negative values, or a max below ~40
#    with a positive min and low dynamic range, or column means ~ 0 ->
#    probably already log-transformed and/or scaled. Warn hard; a second log
#    transform is a silent catastrophe."
#   "Dynamic range: raw untargeted intensities span 10^2-10^9. A ratio below
#    10^2 means something has already been done to the data."
#
# The two clauses do NOT carry the same weight on a single arbitrary column,
# and collapsing them is the very defect this row is about.
#
#   * Non-positive values are ARITHMETIC. `log(x <= 0)` is undefined, on any
#     table in any field. Read once, certain, and it withdraws the advice.
#   * A compressed range is a DOMAIN reading calibrated to untargeted assay
#     intensity blocks. `clinical_risk.csv::creatinine_mg_dl` runs 0.3-3.85
#     mg/dL and trips it while being perfectly raw. Asserting "already
#     transformed" there would be the same class of false claim one surface
#     over, so this is REPORTED and never asserted - the same
#     derived/offered split `turbotab/packs.py::_already_transformed` makes.
#
# `_block(df)` in `turbotab/packs.py` needs 30+ numeric columns, so the pack's
# own detector cannot answer for one target column. The THRESHOLDS are still
# imported from it rather than restated: no threshold moves in this loop
# (AGENT_ONBOARD.md section 08, check 2), and a second copy is the defect
# `ml/eda_recommender.py:130` already corrected once for HIGH_MISSING_SHARE.

#: The reading the card cannot make when nothing measured the target's values.
LOG_STATE_NOT_READ = "not_read"


def read_log_transform_state(series: pd.Series) -> Dict:
    """Whether a log transform is defined on these values, and what they look like.

    Returns the measurements whether or not anything fired, so a caller that
    wants the numbers is not forced through the sentence. `reading` is one of:

    ``log_undefined``   at least one value is <= 0
    ``compressed_scale``  strictly positive, but sitting in the range an
                          already-log-transformed column sits in
    ``no_signature``    strictly positive and spread like raw measurements

    Never ``None`` for a numeric series: "we did not look" is the absence of
    the key in `target_stats`, not a value inside this dict.
    """
    from turbotab.packs import _RAW_DYNAMIC_RANGE, _TRANSFORMED_MAX

    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"reading": LOG_STATE_NOT_READ, "n": 0,
                "sentence": "the target has no finite values to read"}

    n_nonpositive = int((values <= 0).sum())
    lo, hi = float(values.min()), float(values.max())
    ratio = (hi / lo) if lo > 0 else None
    compressed = (lo > 0 and hi < _TRANSFORMED_MAX
                  and ratio is not None and ratio < _RAW_DYNAMIC_RANGE)

    if n_nonpositive:
        reading = "log_undefined"
        sentence = (f"{n_nonpositive:,} of {values.size:,} target values are "
                    f"zero or negative (the smallest is {lo:,.4g})")
    elif compressed:
        reading = "compressed_scale"
        sentence = (f"the target runs {lo:,.4g} to {hi:,.4g}, a span of only "
                    f"{ratio:,.1f}x")
    else:
        reading = "no_signature"
        sentence = (f"the target is strictly positive ({lo:,.4g} to {hi:,.4g}"
                    + (f", a span of {ratio:,.0f}x" if ratio is not None else "")
                    + ")")

    return {"reading": reading, "n": int(values.size),
            "n_nonpositive": n_nonpositive, "min": lo, "max": hi,
            "dynamic_range": ratio, "sentence": sentence,
            "raw_range_floor": _RAW_DYNAMIC_RANGE,
            "transformed_max_ceiling": _TRANSFORMED_MAX}


def log_transform_advice(state: Optional[Dict]) -> Dict[str, str]:
    """The R5 card's three log-transform sentences, from one reading.

    `AUDIT-003` — BEFORE, unconditional and identical for all four cases below:

        what_you_learn:      "Need for log transformation or robust loss"
        model_implications:  "High skew -> consider log transform or robust
                              loss (Huber)"

    AFTER: the same subject, a weaker claim, and true in each case. The advice
    is corrected rather than deleted - `no_signature` still recommends the
    transform, which is why this cannot be satisfied by dropping the word.
    Every branch names what was NOT checked, because none of them can see
    provenance: a log applied by the upstream tool and recorded nowhere in the
    table reads exactly like raw data here.
    """
    unchecked = ("Provenance is not checked: a transform applied before this "
                 "file was written is recorded nowhere the app can read.")

    if not state or state.get("reading") == LOG_STATE_NOT_READ:
        return {
            "why": "Log transform: the target's values were not read",
            "learn": ("Whether a log transform is defined on this target — "
                      "not read here"),
            "implication": (
                "High skew → robust loss (Huber) needs no transform. A log "
                "transform is NOT recommended from the skew alone: nothing "
                "read this target's values, so whether a log is even defined "
                "on them is unknown."),
        }

    reading, said = state["reading"], state["sentence"]

    if reading == "log_undefined":
        return {
            "why": f"Log transform: not defined — {said}",
            "learn": "Why a log transform is unavailable on this target",
            "implication": (
                f"High skew → a log transform is NOT available here: {said}, "
                f"and the log of a non-positive number is undefined. "
                f"Yeo-Johnson accepts them; robust loss (Huber) needs no "
                f"transform at all. {unchecked}"),
        }

    if reading == "compressed_scale":
        return {
            "why": f"Log transform: compressed scale — {said}",
            "learn": ("Whether this target is already on a transformed scale, "
                      "and what the app can and cannot tell from the values"),
            "implication": (
                f"High skew → before any log transform, check what scale this "
                f"target is already on: {said}, where a raw untargeted assay "
                f"run spans 10² to 10⁹. On an assay column that reads as "
                f"already log-transformed, and a second log is silent — it "
                f"produces numbers and every plot still renders. On a clinical "
                f"measurement in its own units (creatinine in mg/dL) the same "
                f"span is ordinary. The app cannot tell those two apart from "
                f"the numbers. {unchecked}"),
        }

    return {
        "why": f"Log transform: available — {said}",
        "learn": "Need for log transformation or robust loss",
        "implication": (
            f"High skew → consider log transform or robust loss (Huber). A log "
            f"is defined here: {said}, carrying none of the signatures of an "
            f"already-transformed column. {unchecked}"),
    }


# ── AUDIT-012: which tier the outlier rate is, before a remedy is named ─────
#
# `research/CLINICAL_SURVEY_PACK.md` §A1.2 keeps two bound sets apart and says
# what each is for — plausibility bounds are "values incompatible with a living
# patient. Use for flagging as suspected data error"; a reference interval is
# the central 95% of a healthy population and is for "annotation. Never for
# exclusion." Cross-cutting 7 ranks the collapse seventh by damage: "Excluding
# abnormal-but-possible clinical values as 'outliers'. Removes the sickest
# patients. Physiologically impossible ≠ abnormal, and generic outlier rules
# (±3 SD, IQR fences) are wrong here."
#
# `ml/outliers.py:39` is an IQR fence and nothing else. R5 and R9 turned its
# rate straight into "use Huber loss or winsorization" and "Consider
# winsorization or outlier removal" — the last of which is the sentence the
# pack names as the damage, offered without anything having checked which tier
# the values are in.

def outlier_tier_advice(rate: float,
                        reading: Optional[Dict]) -> Dict[str, str]:
    """The R5/R9 outlier sentences, from the reading rather than the rate alone.

    `AUDIT-012` — BEFORE, one wording for all three situations below:

        R5 model_implications: "High outlier rate -> use Huber loss or
                                winsorization"
        R9 model_implications: "High outlier rate -> use Huber loss or robust
                                regression" / "Consider winsorization or
                                outlier removal"

    AFTER: Huber is still recommended in every branch — the shelf is not
    shortened and `rate` is still reported — but what the app claims to have
    distinguished shrinks to what it read, and where it read nothing it says
    so and names why.

    `reading` is `ml.dataset_profile.read_target_impossibility`'s dict, or
    `None` where the frame was never read for it.
    """
    pct = f"{rate:.1%}"
    state = (reading or {}).get("physio_read", "unread")
    band = (reading or {}).get("impossibility_band")
    n_imp = (reading or {}).get("impossible_count")
    var = (reading or {}).get("physio_variable")

    if state == "matched" and band and n_imp:
        lo, hi, unit = band
        return {
            "why": (f"Outlier rate: {pct} — {n_imp} of them are outside "
                    f"{lo:g}–{hi:g} {unit}, the published plausibility band "
                    f"for {var}"),
            "learn": ("Which fence hits are entries a living person could not "
                      "produce, and which are extreme but attainable"),
            "implication": (
                f"Outlier rate {pct} is an IQR fence count and it is carrying "
                f"two findings: {n_imp} value(s) fall outside {lo:g}–{hi:g} "
                f"{unit}, which is the published plausibility band for {var}. "
                f"Those are suspected entry errors — repair them on the "
                f"plausibility card. Huber and winsorization DOWNWEIGHT rather "
                f"than repair, and outlier REMOVAL applied to the rest would "
                f"drop abnormal-but-real measurements, which removes the "
                f"sickest rows in the table. Robust loss (Huber) is still the "
                f"right handling for what is left once the impossible entries "
                f"are dealt with."),
        }

    if state == "matched" and band:
        lo, hi, unit = band
        return {
            "why": (f"Outlier rate: {pct} — none outside {lo:g}–{hi:g} {unit}, "
                    f"the published plausibility band for {var}"),
            "learn": ("That the fence hits are inside what a living person "
                      "can produce, so they are extremes rather than errors"),
            "implication": (
                f"Outlier rate {pct}, and every one of those values is inside "
                f"{lo:g}–{hi:g} {unit} — the published plausibility band for "
                f"{var} — so they read as real extremes and not as entry "
                f"errors. Robust loss (Huber) MODELS them at reduced weight; "
                f"winsorization or removal would discard the phenomenon under "
                f"study, and excluding abnormal-but-possible values removes "
                f"the sickest rows in the table."),
        }

    why_unknown = (f"'{var}' has no published plausibility band"
                   if var else
                   f"the target matches no variable in the physiologic "
                   f"reference")
    if state == "unread":
        why_unknown = "the physiologic reference was not read for this target"
    return {
        "why": f"Outlier rate: {pct} — IQR fence only, no plausibility band read",
        "learn": ("How far the extreme values sit from the rest — and NOT "
                  "whether they are possible, which is not checked here"),
        "implication": (
            f"Outlier rate {pct} is an IQR fence count and nothing more: the "
            f"app cannot tell a physiologically impossible entry from an "
            f"abnormal-but-real one here, because {why_unknown}. Robust loss "
            f"(Huber) is a defensible handling under that uncertainty because "
            f"it downweights rather than deletes. Winsorization and outlier "
            f"REMOVAL are not offered on this reading: an impossible entry "
            f"wants repair and an extreme-but-real one wants keeping, and "
            f"nothing here distinguished them."),
    }


def compute_dataset_signals(
    df: pd.DataFrame,
    target: Optional[str],
    task_type_final: Optional[str],
    cohort_type_final: Optional[str],
    entity_id_final: Optional[str],
    sample_size: int = 5000,
    outlier_method: str = "iqr",
    feature_cols: Optional[List[str]] = None,
) -> DatasetSignals:
    """
    Compute dataset signals for EDA recommendations.
    
    Args:
        df: DataFrame
        target: Target column name
        task_type_final: Final task type (regression/classification)
        cohort_type_final: Final cohort type (cross_sectional/longitudinal)
        entity_id_final: Final entity ID column name
        sample_size: Sample size for expensive computations
        
    Returns:
        DatasetSignals object
    """
    signals = DatasetSignals(
        n_rows=len(df),
        n_cols=len(df.columns),
        target_name=target,
        task_type_final=task_type_final,
        cohort_type_final=cohort_type_final,
        entity_id_final=entity_id_final
    )
    
    # Sample for expensive computations
    df_sample = df.sample(min(sample_size, len(df)), random_state=42) if len(df) > sample_size else df
    
    # Column type classification
    for col in df.columns:
        dtype = str(df[col].dtype)
        if dtype.startswith('int') or dtype.startswith('float'):
            signals.numeric_cols.append(col)
        elif (_pdt.is_object_dtype(dtype) or _pdt.is_string_dtype(dtype)
              or isinstance(dtype, pd.CategoricalDtype) or _pdt.is_bool_dtype(dtype)):
            if df[col].dtype == 'bool':
                signals.categorical_cols.append(col)
            elif df[col].dtype.name == 'category':
                signals.categorical_cols.append(col)
            else:
                # Check if it's text-like (high cardinality, mostly unique)
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio > 0.8 and (_pdt.is_object_dtype(df[col])
                                           or _pdt.is_string_dtype(df[col])):
                    signals.text_like_cols.append(col)
                else:
                    signals.categorical_cols.append(col)
        elif 'datetime' in dtype:
            signals.datetime_cols.append(col)
    
    # Missingness
    missing_counts = df.isnull().sum()
    signals.missing_rate_by_col = (missing_counts / len(df)).to_dict()
    # ONE THRESHOLD, READ RATHER THAN RESTATED. `GUIDED-189`.
    #
    # This said `rate > 0.05` and `ml/missingness_plan.HIGH_MISSING_SHARE` says
    # 0.20, and the two decide different halves of one affordance: this one
    # raises the Explore chip, that one fills the cards the chip opens onto. A
    # table whose worst column sits between them — `multiclass_stage.csv`, with
    # `crp` at 10.0% and `bmi` at 7.1% — got a solid-bordered chip whose own
    # tooltip read *"2 columns with >5% missing values"* and which opened onto
    # an empty panel. `GUIDED-006`'s sentence: a control that silently does
    # nothing asserts a capability that does not exist.
    #
    # **Neither threshold moved**, which is the row's own `act` and is also
    # `AGENT_ONBOARD.md` §08 check 2 — the loop that pressured a threshold does
    # not get to move it. The one that FILLS the panel is the real one, because
    # it decides whether there is anything to look at; this one now reads it
    # instead of holding a second copy. §06.2 was not invoked and did not need
    # to be.
    from ml.missingness_plan import HIGH_MISSING_SHARE

    signals.high_missing_cols = [
        col for col, rate in signals.missing_rate_by_col.items()
        if rate > HIGH_MISSING_SHARE
    ]
    
    # Duplicate rows
    signals.duplicate_row_rate = df.duplicated().sum() / len(df)
    
    # High cardinality categoricals
    for col in signals.categorical_cols:
        if df[col].nunique() > len(df) * 0.5:
            signals.high_cardinality_categoricals.append(col)
    
    # Target statistics
    if target and target in df.columns:
        target_series = df[target].dropna()
        if len(target_series) > 0:
            signals.target_stats['n_unique'] = target_series.nunique()
            signals.target_stats['n_missing'] = df[target].isnull().sum()
            signals.target_stats['missing_rate'] = signals.target_stats['n_missing'] / len(df)
            
            if task_type_final == 'regression' and _pdt.is_numeric_dtype(target_series):
                signals.target_stats['mean'] = target_series.mean()
                signals.target_stats['median'] = target_series.median()
                signals.target_stats['std'] = target_series.std()
                signals.target_stats['skew'] = stats.skew(target_series)
                signals.target_stats['kurtosis'] = stats.kurtosis(target_series)
                
                outlier_mask, _ = detect_outliers(target_series, method=outlier_method)
                signals.target_stats['outlier_rate'] = float(outlier_mask.sum() / len(target_series)) if len(target_series) > 0 else 0.0

                # `AUDIT-003`. R5 used to advise a log transform from the skew
                # alone. The reading that decides whether a log is even legal is
                # taken HERE, because `recommend_eda` gets a `DatasetSignals`
                # and never sees the frame. Absent from `target_stats` means
                # NOT READ — the card says so rather than assuming raw.
                signals.target_stats['log_transform_state'] = read_log_transform_state(target_series)

                # `AUDIT-012`, the same shape one tier over. The fence rate
                # above answers "is this value far from the others"; the
                # published plausibility band answers "could a living person
                # produce it". R5 and R9 recommended Huber, winsorization and
                # outlier REMOVAL from the first number alone. Read here for
                # the same reason as the line above — `recommend_eda` never
                # sees the frame. Absent means NOT READ.
                try:
                    from ml.dataset_profile import read_target_impossibility
                    signals.target_stats['impossibility'] = \
                        read_target_impossibility(df, target)
                except Exception:
                    pass
            elif task_type_final == 'classification':
                value_counts = target_series.value_counts()
                signals.target_stats['class_counts'] = value_counts.to_dict()
                signals.target_stats['n_classes'] = len(value_counts)
                if len(value_counts) > 0:
                    max_class = value_counts.max()
                    min_class = value_counts.min()
                    signals.target_stats['class_imbalance_ratio'] = min_class / max_class if max_class > 0 else 0.0
                    signals.target_stats['majority_class_prop'] = max_class / len(target_series)
    
    # Leakage detection (simple heuristics)
    if target:
        # Check for perfect or near-perfect correlations
        # Only use columns that are truly numeric (can be converted to float)
        numeric_for_corr = []
        for col in signals.numeric_cols:
            if col != target:
                try:
                    # Verify column is actually numeric
                    pd.to_numeric(df[col], errors='raise')
                    numeric_for_corr.append(col)
                except (ValueError, TypeError):
                    pass  # Skip columns that can't be converted
        
        if len(numeric_for_corr) > 0 and target in df.columns:
            try:
                # Also verify target can be used in correlation
                target_numeric = pd.to_numeric(df[target], errors='coerce')
                if target_numeric.notna().sum() > 0:
                    corr_df = df[numeric_for_corr].copy()
                    corr_df['_target'] = target_numeric
                    corr_with_target = corr_df.corr()['_target'].abs()
                    high_corr = corr_with_target[corr_with_target > 0.95].drop('_target', errors='ignore')
                    if len(high_corr) > 0:
                        signals.leakage_flags.append(f"{len(high_corr)} columns with >0.95 correlation to target")
                        signals.leakage_candidate_cols = high_corr.index.tolist()
            except Exception as exc:
                # The scan may fail — the p x p correlation matrix is built
                # uncapped — but its failure must not read as a clean bill of
                # health. Nothing downstream can tell an empty
                # leakage_candidate_cols from a scan that never ran, and the
                # EDA page's "no blocking data-quality issues" sentence is
                # emitted on exactly that emptiness.
                signals.leakage_scan_error = f"{type(exc).__name__}: {exc}"
                signals.notes.append(
                    f"Target-leakage screen did not complete: {signals.leakage_scan_error}"
                )
    
    # Collinearity (sample if too many columns)
    # Filter to truly numeric columns within user's selected features
    _corr_candidates = signals.numeric_cols
    if feature_cols:
        _feature_set = set(feature_cols)
        _corr_candidates = [c for c in _corr_candidates if c in _feature_set]
    numeric_cols_for_corr = []
    for col in _corr_candidates:
        try:
            pd.to_numeric(df[col], errors='raise')
            numeric_cols_for_corr.append(col)
        except (ValueError, TypeError):
            pass
    
    if len(numeric_cols_for_corr) > 50:
        # Sample by variance
        try:
            variances = df[numeric_cols_for_corr].var().sort_values(ascending=False)
            numeric_cols_for_corr = variances.head(50).index.tolist()
        except Exception:
            numeric_cols_for_corr = numeric_cols_for_corr[:50]
    
    if len(numeric_cols_for_corr) > 1:
        try:
            corr_matrix = df[numeric_cols_for_corr].corr().abs()
            # Zero the diagonal on a copy, never through DataFrame.values.
            # Under copy-on-write — the default from pandas 3 — .values hands
            # back a read-only array and np.fill_diagonal raises, which the
            # except below then swallowed. The whole collinearity analysis
            # silently disappeared: no correlation clusters in the ledger, no
            # collinearity coaching, and no sign anything had gone wrong.
            corr_values = corr_matrix.to_numpy(dtype=float, copy=True)
            np.fill_diagonal(corr_values, 0.0)
            max_corr = float(np.nanmax(corr_values)) if corr_values.size else 0.0
            signals.collinearity_summary['max_corr'] = max_corr
            signals.collinearity_summary['high_corr_pairs'] = []
            if max_corr > 0.85:
                # Find high correlation pairs
                for i, col1 in enumerate(numeric_cols_for_corr):
                    for col2 in numeric_cols_for_corr[i+1:]:
                        try:
                            corr_val = abs(df[col1].corr(df[col2]))
                            if corr_val > 0.85:
                                signals.collinearity_summary['high_corr_pairs'].append((col1, col2, corr_val))
                        except Exception:
                            pass
        except Exception:
            pass  # Skip collinearity analysis if it fails
    
    # Empirical plausibility flags (NHANES percentile reference)
    reference_bundle = load_reference_bundle()
    nhanes_ref = reference_bundle["nhanes"]
    for col in signals.numeric_cols:
        var_key = match_variable_key(col, nhanes_ref)
        if not var_key:
            continue
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
        inferred_unit_info = infer_unit(col, col_data)
        improbability = get_improbability_band(nhanes_ref, var_key)
        if inferred_unit_info.get('conversion_factor') and improbability:
            improbable_low, improbable_high, improbable_unit = improbability
            converted = col_data * inferred_unit_info['conversion_factor']
            out_rate = ((converted < improbable_low) | (converted > improbable_high)).sum() / len(converted)
            if out_rate > 0.05:
                # `MISC-018`: p01–p99 is an improbability band, not a reference
                # interval, and the caption on page 02 disavows the second name
                # two lines above where this warning prints.
                signals.physio_plausibility_flags.append(
                    f"{col}: {out_rate:.1%} values outside the NHANES improbability band "
                    f"({improbable_low}-{improbable_high} {improbable_unit})"
                )
    
    return signals


def recommend_eda(signals: DatasetSignals) -> List[EDARecommendation]:
    """
    Generate EDA recommendations based on dataset signals.
    
    Args:
        signals: DatasetSignals object
        
    Returns:
        List of EDARecommendation objects, sorted by priority
    """
    recommendations = []
    
    # R1: Physiologic plausibility / range check (always)
    recommendations.append(EDARecommendation(
        id="r1_plausibility",
        title="Physiologic Plausibility Check",
        priority=1,
        cost="low",
        why=[
            f"Dataset has {signals.n_rows} rows with {len(signals.numeric_cols)} numeric columns",
            "Medical/nutritional data should be checked for plausible ranges"
        ],
        what_you_learn=[
            "Out-of-range values that may indicate data entry errors",
            "Potential unit mismatches (e.g., mmol/L vs mg/dL)",
            "Columns requiring transformation or winsorization"
        ],
        model_implications=[
            "Outliers may require robust loss (Huber) or winsorization",
            "Unit mismatches can cause model instability"
        ],
        run_action="plausibility_check"
    ))
    
    # R2: Missingness mechanism scan
    if len(signals.high_missing_cols) > 0:
        recommendations.append(EDARecommendation(
            id="r2_missingness",
            title="Missingness Pattern Analysis",
            priority=2,
            cost="low",
            why=[
                f"{len(signals.high_missing_cols)} columns with >5% missing values",
                f"Max missing rate: {max(signals.missing_rate_by_col.values()):.1%}"
            ],
            what_you_learn=[
                "Which columns have missing data and at what rate",
                "Whether missingness is associated with target (informative missingness)",
                "Patterns suggesting MCAR (Missing Completely At Random) vs MAR (Missing At Random) vs MNAR (Missing Not At Random)"
            ],
            model_implications=[
                "Informative missingness may require missingness indicators",
                "High missing rates may need specialized imputation strategies"
            ],
            run_action="missingness_scan",
            description=(
                "**What this is:** Analyzes patterns in missing data across columns and checks if missingness "
                "is associated with the target variable.\n\n"
                "**Why it matters:** If missingness is informative (associated with target), it contains signal "
                "that models can use. MCAR means missing is random; MAR means missing depends on observed data; "
                "MNAR means missing depends on unobserved values.\n\n"
                "**How to interpret:** If target mean differs between missing/non-missing groups, missingness is "
                "informative and you may want to add missingness indicator features."
            )
        ))
    
    # R3: Cohort structure + split warning
    if signals.cohort_type_final == "longitudinal":
        recommendations.append(EDARecommendation(
            id="r3_cohort_structure",
            title="Longitudinal Data Split Guidance",
            priority=2,
            cost="low",
            why=[
                f"Cohort type detected: {signals.cohort_type_final}",
                f"Entity ID: {signals.entity_id_final or 'Not specified'}"
            ],
            what_you_learn=[
                "Distribution of rows per entity",
                "Risk of data leakage with random splits",
                "Recommended split strategy"
            ],
            model_implications=[
                "Must use group-based splitting to prevent leakage",
                "Consider time-based splits if temporal ordering exists"
            ],
            run_action="cohort_split_guidance"
        ))
    
    # R4: Leakage risk scan
    if len(signals.leakage_flags) > 0 or len(signals.leakage_candidate_cols) > 0:
        recommendations.append(EDARecommendation(
            id="r4_leakage",
            title="Target Leakage Risk Assessment",
            priority=3,
            cost="low",
            why=signals.leakage_flags + [
                f"{len(signals.leakage_candidate_cols)} suspicious columns identified"
            ],
            what_you_learn=[
                "Columns with suspiciously high correlation to target",
                "Potential ID-like columns that should be excluded",
                "Features that may contain target information"
            ],
            model_implications=[
                "Leakage columns must be excluded from features",
                "High correlation may indicate data quality issues"
            ],
            run_action="leakage_scan",
            description=(
                "**What this is:** Identifies columns that have suspiciously high correlation (>0.95) with the target, "
                "which may indicate data leakage.\n\n"
                "**Why it matters:** Data leakage occurs when features contain information that would not be available "
                "at prediction time, leading to unrealistically high performance that won't generalize.\n\n"
                "**How to interpret:** Columns flagged should be excluded from features unless you can verify they "
                "are legitimate predictors available at prediction time."
            )
        ))
    
    # R5: Target distribution
    if signals.target_name:
        if signals.task_type_final == "regression":
            outlier_rate = signals.target_stats.get('outlier_rate', 0)
            skew = signals.target_stats.get('skew', 0)
            # `AUDIT-003`. The log clause is composed from a reading of the
            # target's values instead of from the skew alone, and the reading
            # goes into `why` as well — `ml/router.py:397` renders `why` on the
            # Guided pull chip and renders neither of the other two lists, so a
            # correction that landed only in `model_implications` would be true
            # on the wire and invisible to a person (trap 6).
            log_advice = log_transform_advice(
                signals.target_stats.get('log_transform_state'))
            # `AUDIT-012`. The bare `Outlier rate: x%` line stood in `why`, the
            # one field `ml/router.py:397` carries onto the Guided pull chip,
            # and the remedy beside it was chosen from that number alone.
            outlier_advice = outlier_tier_advice(
                outlier_rate, signals.target_stats.get('impossibility'))
            recommendations.append(EDARecommendation(
                id="r5_target_regression",
                title="Target Distribution & Outliers",
                priority=3,
                cost="low",
                why=[
                    f"Target: {signals.target_name}",
                    f"Skewness: {skew:.2f}",
                    outlier_advice["why"],
                    log_advice["why"],
                ],
                what_you_learn=[
                    "Target distribution shape (normal, skewed, multimodal)",
                    outlier_advice["learn"],
                    log_advice["learn"],
                ],
                model_implications=[
                    log_advice["implication"],
                    outlier_advice["implication"],
                    "Multimodal → may benefit from tree-based models"
                ],
                run_action="target_profile"
            ))
        elif signals.task_type_final == "classification":
            imbalance_ratio = signals.target_stats.get('class_imbalance_ratio', 1.0)
            n_classes = signals.target_stats.get('n_classes', 0)
            recommendations.append(EDARecommendation(
                id="r5_target_classification",
                title="Class Balance & Baseline",
                priority=3,
                cost="low",
                why=[
                    f"Target: {signals.target_name}",
                    f"Classes: {n_classes}",
                    f"Imbalance ratio: {imbalance_ratio:.2f}"
                ],
                what_you_learn=[
                    "Class distribution and balance",
                    "Baseline accuracy (majority class)",
                    # `GUIDED-049`: was "Need for class weighting or
                    # resampling", which presupposes the answer.
                    "Whether the outcome is rare enough to threaten the fit",
                ],
                model_implications=[
                    # Was "Imbalanced classes → use class_weight or F1/PR-AUC
                    # metrics". Rebalancing degrades calibration without
                    # improving discrimination (van den Goorbergh et al., JAMIA
                    # 2022;29:1525), so the metric half survives and the
                    # rebalancing half does not.
                    "Imbalanced classes → report PR-AUC and calibration, and "
                    "choose the threshold from the costs",
                    "Low baseline → model must significantly outperform random",
                    "Binary vs multiclass affects model choice"
                ],
                run_action="target_profile"
            ))
    
    # R6: Dose-response trends
    if signals.target_name and len(signals.numeric_cols) > 1 and signals.n_rows > 100:
        recommendations.append(EDARecommendation(
            id="r6_dose_response",
            title="Dose-Response Trends",
            priority=4,
            cost="medium",
            why=[
                f"{len(signals.numeric_cols)} numeric features available",
                f"Dataset size: {signals.n_rows} rows"
            ],
            what_you_learn=[
                "Top features by association with target",
                "Nonlinear relationships (monotonic, U-shaped, etc.)",
                "Feature ranges where target behavior changes"
            ],
            model_implications=[
                "Nonlinear trends → prefer RF or NN over GLM",
                "Monotonic trends → GLM with splines may suffice",
                "U-shaped → tree models or polynomial features"
            ],
            run_action="dose_response_trends"
        ))
    
    # R7: Interaction radar (if age/sex/BMI present)
    interaction_cols = []
    for col in signals.numeric_cols + signals.categorical_cols:
        col_lower = col.lower()
        if any(term in col_lower for term in ['age', 'sex', 'gender', 'bmi']):
            interaction_cols.append(col)
    
    if len(interaction_cols) > 0 and signals.target_name:
        recommendations.append(EDARecommendation(
            id="r7_interactions",
            title="Stratified Trends by Demographics",
            priority=5,
            cost="medium",
            why=[
                f"Found demographic columns: {', '.join(interaction_cols)}",
                "Medical data often shows age/sex/BMI interactions"
            ],
            what_you_learn=[
                "How feature-target relationships vary by demographics",
                "Potential interaction terms for GLM",
                "Subgroup-specific patterns"
            ],
            model_implications=[
                "Significant interactions → include interaction terms in GLM",
                "Tree models (RF) automatically capture interactions",
                "NN can learn complex interactions if data is sufficient"
            ],
            run_action="interaction_analysis"
        ))
    
    # R8: Collinearity map
    max_corr = signals.collinearity_summary.get('max_corr', 0)
    if max_corr > 0.85 and len(signals.numeric_cols) > 5:
        recommendations.append(EDARecommendation(
            id="r8_collinearity",
            title="Collinearity Heatmap",
            priority=4,
            cost="low",
            why=[
                f"Maximum correlation: {max_corr:.2f}",
                f"{len(signals.collinearity_summary.get('high_corr_pairs', []))} highly correlated pairs"
            ],
            what_you_learn=[
                "Feature clusters with high correlation",
                "Redundant features that can be removed",
                "Multicollinearity risks for GLM (Generalized Linear Model)"
            ],
            model_implications=[
                "High collinearity → GLM coefficients unstable, use regularization",
                "RF (Random Forest) and NN (Neural Network) are more robust to collinearity",
                "Consider PCA (Principal Component Analysis) or feature selection"
            ],
            run_action="collinearity_map",
            description=(
                "**What this is:** Shows correlation heatmap between numeric features. High correlations (>0.85) "
                "indicate collinearity (features are highly related).\n\n"
                "**Why it matters:** Collinearity makes GLM coefficients unstable and hard to interpret. "
                "Tree-based models (RF) and neural networks are more robust.\n\n"
                "**How to interpret:** Clusters of highly correlated features may be redundant. Consider removing "
                "one from each cluster or using dimensionality reduction."
            )
        ))
    
    # R9: Outlier influence (regression)
    if signals.task_type_final == "regression":
        outlier_rate = signals.target_stats.get('outlier_rate', 0)
        if outlier_rate > 0.05:
            # `AUDIT-012`. This card carried the sharpest form of the row:
            # "Consider winsorization or outlier removal", composed from the
            # fence rate alone. CLINICAL_SURVEY_PACK Cross-cutting 7 names
            # exactly that offer as the damage — it is how the sickest patients
            # leave the table.
            outlier_advice = outlier_tier_advice(
                outlier_rate, signals.target_stats.get('impossibility'))
            recommendations.append(EDARecommendation(
                id="r9_outlier_influence",
                title="Outlier Influence Analysis",
                priority=5,
                cost="medium",
                why=[
                    outlier_advice["why"],
                    "Outliers can heavily influence regression models"
                ],
                what_you_learn=[
                    "Location and magnitude of outliers",
                    "Impact on model predictions",
                    outlier_advice["learn"],
                ],
                model_implications=[
                    outlier_advice["implication"],
                    "NN with robust loss may outperform GLM"
                ],
                run_action="outlier_influence"
            ))
    
    # R10 (Quick Baseline Models) is deliberately absent. Its action split the
    # full frame with a bare train_test_split and never consulted the test
    # lockbox, so its "held-out" scores were fit and scored partly on sealed
    # rows. The evidence probe on the Preprocess page answers the same question
    # on training rows only, with a permuted-target null and a learning-curve
    # slope, and the Baselines tab on Train & Compare scores real baselines
    # through each model's own fitted pipeline with bootstrap intervals.

    # Sort by priority
    recommendations.sort(key=lambda x: x.priority)
    
    return recommendations

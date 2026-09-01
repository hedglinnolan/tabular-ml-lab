"""
EDA Analysis Actions - Runnable functions for EDA recommendations.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Any
from scipy import stats
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score
# No module-level Streamlit import: every use in this file is already a
# function-level import (the host is only needed when an action renders).

from ml.eval import calculate_regression_metrics, calculate_classification_metrics
from ml.clinical_units import infer_unit
from ml.physiology_reference import load_reference_bundle, match_variable_key, get_improbability_band
from ml.outliers import detect_outliers
# The compute-cap axis. `ml.regime` holds the thresholds AND the sentence that
# discloses each one, so the engine, the page and the tests quote the same
# number and the same words. It imports nothing but pandas/numpy, so this stays
# headless (`tests/test_engine_is_headless.py`).
from ml.regime import ols_diagnostic_availability, vif_availability
from ml.stats_tests import (
    correlation_test,
    two_sample_location_test,
    categorical_association_test,
    normality_check,
)
# Insights are built here but written by the caller. These are plain functions,
# and get_ledger() reaches into st.session_state, which silently no-ops outside
# a script run — bare-mode Streamlit warns and drops the write, so a smoke test
# would pass while the finding went nowhere. Returning them in
# result['insights'] keeps the analysis testable and lets the page upsert them
# where a session actually exists.
from utils.insight_ledger import Insight, ISSUE_MODEL_RELEVANCE


def plausibility_check(
    df: pd.DataFrame,
    target: Optional[str],
    features: List[str],
    signals: Any,
    session_state: Any
) -> Dict[str, Any]:
    """
    Check physiologic plausibility for common clinical columns with unit inference.
    
    Returns:
        Dict with 'findings', 'warnings', 'figures'
    """
    findings = []
    warnings = []
    figures = []
    insights: List[Insight] = []
    reference_bundle = load_reference_bundle()
    nhanes_ref = reference_bundle["nhanes"]
    clinical_guidelines = reference_bundle["clinical"]
    
    # Get unit overrides from session state
    unit_overrides = session_state.get('unit_overrides', {})
    
    checked_cols = []
    out_of_range = []
    empirical_ranges = []
    unit_inferences = []
    clinical_comparison = []
    
    for col in df.columns:
        col_lower = col.lower()
        # Check if this matches an NHANES reference variable
        var_key = match_variable_key(col, nhanes_ref)
        
        if var_key and col in signals.numeric_cols:
            checked_cols.append(col)
            col_data = df[col].dropna()
            
            if len(col_data) > 0:
                # Infer unit (or use override)
                if col in unit_overrides:
                    inferred_unit_info = {
                        'inferred_unit': unit_overrides[col],
                        'canonical_unit': 'unknown',
                        'confidence': 'override',
                        'explanation': f'User override: {unit_overrides[col]}',
                        'conversion_factor': 1.0
                    }
                else:
                    inferred_unit_info = infer_unit(col, col_data)
                
                # Build unit inference row with threshold bands if available
                unit_row = {
                    'Column': col,
                    'Inferred Unit': inferred_unit_info.get('inferred_unit', 'Unknown'),
                    'Canonical Unit': inferred_unit_info.get('canonical_unit', 'N/A'),
                    'Confidence': inferred_unit_info.get('confidence', 'low'),
                    'Explanation': inferred_unit_info.get('explanation', '')
                }
                
                # Add fasting note if applicable
                if inferred_unit_info.get('fasting_note'):
                    unit_row['Note'] = 'Fasting assumption (the improbability band assumes a fasting state)'
                else:
                    unit_row['Note'] = ''
                
                unit_inferences.append(unit_row)
                
                # Empirical plausibility from NHANES reference (percentile-based)
                improbability = get_improbability_band(nhanes_ref, var_key)
                if inferred_unit_info.get('conversion_factor') and improbability:
                    improbable_low, improbable_high, improbable_unit = improbability
                    converted = col_data * inferred_unit_info['conversion_factor']

                    below_min = (converted < improbable_low).sum()
                    above_max = (converted > improbable_high).sum()
                    total_out = below_min + above_max
                    out_rate = total_out / len(col_data)

                    if total_out > 0:
                        out_of_range.append(col)

                    empirical_ranges.append({
                        'Column': col,
                        # `MISC-018`. Was 'Reference Interval (NHANES p01–p99)',
                        # which names the central 95% and then prints the
                        # central 98% beside it. The label was the defect and
                        # the parenthesis was the proof.
                        'Improbability band (NHANES p01–p99)': f"{improbable_low}-{improbable_high} {improbable_unit}",
                        'Min (canonical)': f"{converted.min():.1f}",
                        'Max (canonical)': f"{converted.max():.1f}",
                        'Out of Range %': f"{out_rate:.1%}" if total_out > 0 else "0%"
                    })

                    if out_rate > 0.05:
                        warnings.append(
                            f"{col}: {out_rate:.1%} values outside the NHANES improbability band "
                            f"({improbable_low}-{improbable_high} {improbable_unit}) after conversion from {inferred_unit_info['inferred_unit']}"
                        )

                # Clinical guideline comparison (informational only)
                guideline = clinical_guidelines.get(var_key)
                if guideline:
                    thresholds = guideline.get('thresholds_by_unit', {}).get(inferred_unit_info.get('inferred_unit'))
                    if thresholds:
                        threshold_bands = {}
                        for band_name, (band_min, band_max) in thresholds.items():
                            if band_max is None:
                                count = (col_data >= band_min).sum()
                            else:
                                count = ((col_data >= band_min) & (col_data < band_max)).sum()
                            threshold_bands[band_name] = count

                        band_names = {
                            'normal': 'Normal',
                            'prediabetes': 'Prediabetes',
                            'diabetes': 'Diabetes',
                            'borderline_high': 'Borderline High',
                            'high': 'High',
                            'very_high': 'Very High'
                        }
                        band_summary = []
                        for band_name, count in threshold_bands.items():
                            pct = count / len(col_data)
                            if pct > 0:
                                band_summary.append(f"{band_names.get(band_name, band_name)}: {pct:.1%}")

                        clinical_comparison.append({
                            'Column': col,
                            'Unit (clinical)': inferred_unit_info.get('inferred_unit', 'Unknown'),
                            'Distribution': ", ".join(band_summary) if band_summary else "No thresholds triggered",
                            'Note': 'Clinical guideline overlay (informational only)'
                        })
    
    findings.append(f"Checked {len(checked_cols)} columns with medical/nutritional patterns")
    
    if len(unit_inferences) > 0:
        unit_df = pd.DataFrame(unit_inferences)
        figures.append(('table', unit_df))
        findings.append(f"Inferred units for {len(unit_inferences)} clinical variables")
    
    if len(empirical_ranges) > 0:
        findings.append(f"Computed empirical plausibility for {len(empirical_ranges)} columns (NHANES reference)")
        empirical_df = pd.DataFrame(empirical_ranges)
        figures.append(('table', empirical_df))
    if len(clinical_comparison) > 0:
        findings.append(f"Computed clinical guideline overlays for {len(clinical_comparison)} columns (informational)")
        clinical_df = pd.DataFrame(clinical_comparison)
        figures.append(('table', clinical_df))
    if len(out_of_range) > 0:
        findings.append(f"Found {len(out_of_range)} columns with out-of-range values")
    else:
        findings.append("All checked columns within plausible ranges")
    
    # Add unit sanity flags from signals — minus the bands this action already
    # reported above (D9-08: the recommender writes the same sentence without
    # the "after conversion from <unit>" clause, and both used to print).
    if signals.physio_plausibility_flags:
        _already = {w.split(" after conversion from ")[0] for w in warnings}
        _fresh = [f for f in signals.physio_plausibility_flags if f not in _already]
        warnings.extend(_fresh)
        if _fresh:
            findings.append(f"Found {len(_fresh)} empirical plausibility flags")
    
    # Add note about unit overrides
    if unit_overrides:
        findings.append(f"Using {len(unit_overrides)} user-specified unit overrides")
    
    # A unit mismatch (mmol/L read as mg/dL) shifts a whole column without
    # producing a single statistical outlier, so no other check in the app can
    # catch it. This observation has to reach the ledger.
    if signals.physio_plausibility_flags or out_of_range:
        num_flags = len(signals.physio_plausibility_flags) if signals.physio_plausibility_flags else 0
        # MERGE NOTE: main carries the observation on the returned `Insight`
        # (utils.storyline.add_insight was deleted in the ledger migration, and
        # the old call here was swallowed by a bare `except`, so the finding
        # went nowhere). TurboTab's contribution to this block was the
        # vocabulary, `MISC-018`: p01-p99 is an *improbability band*, not a
        # reference interval, which names the central 95% of a healthy
        # population. Both are kept - main's carrier, TurboTab's wording.
        num_out_of_range = len(out_of_range)
        cols_str = ", ".join(out_of_range[:6])
        if num_out_of_range > 6:
            cols_str += f" +{num_out_of_range - 6} more"
        if out_of_range:
            manuscript = (
                f"{num_out_of_range} {'variable' if num_out_of_range == 1 else 'variables'} "
                f"contained values outside the NHANES p01\u2013p99 improbability band for "
                f"the inferred unit ({cols_str}), which may reflect unit inconsistency or "
                f"measurement error rather than true physiological extremes"
            )
        else:
            manuscript = (
                f"{num_flags} empirical plausibility "
                f"{'flag was' if num_flags == 1 else 'flags were'} raised against the "
                f"NHANES p01–p99 improbability band, which may reflect unit "
                f"inconsistency or measurement error rather than true physiological "
                f"extremes"
            )
        insights.append(Insight(
            id="eda_plausibility_out_of_range",
            source_page="02_EDA",
            category="data_quality",
            severity="warning",
            finding=(
                f"Physiologic plausibility: {num_out_of_range} column(s) outside the "
                f"NHANES improbability band (p01\u2013p99)"
                + (f" ({cols_str})" if cols_str else "")
                + f", {num_flags} empirical plausibility flag(s)"
            ),
            implication=(
                "Values outside the improbability band are more often a unit mismatch "
                "or a data-entry error than real physiology, and every downstream model "
                "inherits the error unchallenged. That band is not a reference interval "
                "\u2014 a reference interval is the central 95% of a healthy reference "
                "population, and a value outside this one is unusual rather than "
                "abnormal. Clinical thresholds are informational only."
            ),
            recommended_action=(
                "Confirm the inferred units in Upload & Audit (unit overrides), then correct "
                "or filter the rows that remain implausible."
            ),
            manuscript_text=manuscript,
            affected_features=list(out_of_range),
            relevant_pages=["01_Upload_and_Audit", "10_Report_Export"],
            # Explicit: the category default for data_quality is TRIPOD 9
            # (missing data), and a units check does not satisfy that item.
            tripod_keys=["predictors_defined"],
            # model_scope omitted -> implausible values affect every model family.
            metadata={
                "n_columns_checked": len(checked_cols),
                "n_out_of_range_columns": num_out_of_range,
                "out_of_range_columns": list(out_of_range),
                "n_empirical_flags": num_flags,
                "unit_overrides_applied": sorted(unit_overrides.keys()),
            },
        ))
    
    return {
        'findings': findings,
        'warnings': warnings,
        'figures': figures,
        'insights': insights,
    }


def missingness_scan(
    df: pd.DataFrame,
    target: Optional[str],
    features: List[str],
    signals: Any,
    session_state: Any
) -> Dict[str, Any]:
    """Analyze missingness patterns and association with target.

    The informative-missingness insight is written by the EDA page, which runs
    missingness_target_association() inline next to its missing-data chart.
    Writing it here too would file the same finding twice.
    """
    findings = []
    warnings = []
    figures = []

    if not target or target not in df.columns:
        return {
            'findings': ["Target not available for missingness analysis"],
            'warnings': [],
            'figures': []
        }
    
    # Missingness bar chart
    missing_df = pd.DataFrame({
        'Column': list(signals.missing_rate_by_col.keys()),
        'Missing Rate': list(signals.missing_rate_by_col.values())
    })
    missing_df = missing_df[missing_df['Missing Rate'] > 0].sort_values('Missing Rate', ascending=False)
    
    if len(missing_df) > 0:
        fig = px.bar(
            missing_df.head(20),
            x='Missing Rate',
            y='Column',
            orientation='h',
            title='Missingness by Column (Top 20)'
        )
        figures.append(('plotly', fig))
        findings.append(f"{len(missing_df)} columns have missing values")
    
    # Missingness vs target association. Delegated to the focused function so
    # there is exactly one implementation of this test — the EDA page renders
    # the same table inline next to its missing-data bar chart.
    assoc = missingness_target_association(
        df, target, signals.task_type_final,
        candidate_cols=[c for c in df.columns if c != target],
    )
    assoc_df = assoc.get('table')
    if assoc_df is not None and len(assoc_df) > 0:
        figures.append(('table', assoc_df))
        if assoc['n_significant'] > 0:
            findings.append(
                f"Missingness is informative in {assoc['n_significant']} of "
                f"{assoc['n_tested']} columns tested (Benjamini-Hochberg q < 0.05)"
            )
        else:
            findings.append(
                f"No target association found in the {assoc['n_tested']} columns tested"
            )
        # iterrows, not itertuples: pandas renames 'p-value' to a positional
        # attribute, so the old row.p_value lookup silently found nothing.
        for _, row in assoc_df.head(3).iterrows():
            pv = row['p-value']
            if pd.notna(pv):
                findings.append(f"  {row['Column']}: {row['Test']} p={pv:.4g}")
    if assoc.get('skipped_low_n'):
        findings.append(
            f"{len(assoc['skipped_low_n'])} column(s) had too few rows on one side "
            f"of the missing/present split to test"
        )
    
    return {
        'findings': findings,
        'warnings': warnings,
        'figures': figures
    }


def missingness_target_association(
    df: pd.DataFrame,
    target: Optional[str],
    task_type: Optional[str],
    candidate_cols: Optional[List[str]] = None,
    max_cols: int = 15,
    min_group_n: int = 20,
) -> Dict[str, Any]:
    """Test whether *being missing* in a column is associated with the target.

    This is the informative-missingness check. If the target differs between the
    rows where a column is missing and the rows where it is present, the gap is
    not random (MAR/MNAR) and the fact of being missing is itself a predictor —
    which argues for a missing-indicator column rather than a silent median fill.

    Columns are gated on absolute group size, not on missing *rate*. A column
    that is 2% missing in 50,000 rows has 1,000 missing rows and ample power —
    and is exactly where an "ordered only when the clinician suspected something"
    pattern hides. A column that is 8% missing in 200 rows has 16 and no power.
    The old >5%-rate gate had it backwards on both counts.

    Args:
        df: The working table.
        target: Target column name.
        task_type: 'regression' or 'classification'; anything else returns empty.
        candidate_cols: Columns to consider (defaults to every column but the target).
        max_cols: Cap on columns tested, taken by missing count descending.
        min_group_n: Minimum rows required on BOTH sides of the missing/present split.

    Returns:
        {'table': DataFrame | None,   # one row per tested column, sorted by p
         'n_tested': int, 'n_candidates': int, 'skipped_low_n': List[str],
         'n_significant': int,        # count with BH q < 0.05
         'significant_cols': List[str], 'top': dict | None, 'effect_col': str}
    """
    empty: Dict[str, Any] = {
        'table': None, 'n_tested': 0, 'n_candidates': 0, 'skipped_low_n': [],
        'n_significant': 0, 'significant_cols': [], 'top': None, 'effect_col': '',
    }
    if not target or target not in df.columns or task_type not in ('regression', 'classification'):
        return empty

    cols = candidate_cols if candidate_cols is not None else list(df.columns)
    n_missing_by_col: Dict[str, int] = {}
    skipped: List[str] = []
    for col in cols:
        if col == target or col not in df.columns:
            continue
        n_miss = int(df[col].isnull().sum())
        if n_miss == 0:
            continue
        if n_miss < min_group_n or (len(df) - n_miss) < min_group_n:
            skipped.append(col)
            continue
        n_missing_by_col[col] = n_miss

    empty['n_candidates'] = len(n_missing_by_col)
    empty['skipped_low_n'] = skipped
    ranked = sorted(n_missing_by_col, key=lambda c: n_missing_by_col[c], reverse=True)[:max_cols]
    if not ranked:
        return empty

    rows: List[Dict[str, Any]] = []
    if task_type == 'regression':
        target_vals = pd.to_numeric(df[target], errors='coerce').dropna().to_numpy(dtype=float)
        if len(target_vals) < 3:
            return empty
        _, norm_p, _ = normality_check(target_vals)
        parametric = not (np.isfinite(norm_p) and norm_p < 0.05)
        target_sd = float(np.std(target_vals)) if len(target_vals) > 1 else 0.0
        for col in ranked:
            missing_mask = df[col].isnull()
            t_missing = pd.to_numeric(df.loc[missing_mask, target], errors='coerce').dropna().to_numpy(dtype=float)
            t_present = pd.to_numeric(df.loc[~missing_mask, target], errors='coerce').dropna().to_numpy(dtype=float)
            if len(t_missing) < 2 or len(t_present) < 2:
                skipped.append(col)
                continue
            _stat, p, test_name = two_sample_location_test(t_missing, t_present, parametric)
            diff = float(np.mean(t_missing) - np.mean(t_present))
            rows.append({
                'Column': col,
                'Missing n': int(missing_mask.sum()),
                'Missing %': float(missing_mask.mean()),
                f'Mean {target} | missing': float(np.mean(t_missing)),
                f'Mean {target} | present': float(np.mean(t_present)),
                'Difference': diff,
                'Std. difference': (diff / target_sd) if target_sd > 0 else float('nan'),
                'Test': test_name,
                'p-value': float(p),
            })
        effect_col = 'Difference'
    else:
        for col in ranked:
            missing_mask = df[col].isnull()
            cont = pd.crosstab(missing_mask, df[target])
            if cont.shape[0] < 2 or cont.shape[1] < 2:
                skipped.append(col)
                continue
            _stat, p, test_name = categorical_association_test(
                cont.values, use_fisher=(cont.shape == (2, 2))
            )
            rate_missing = df.loc[missing_mask, target].value_counts(normalize=True)
            rate_present = df.loc[~missing_mask, target].value_counts(normalize=True)
            classes = rate_missing.index.union(rate_present.index)
            max_gap = float(
                (rate_missing.reindex(classes, fill_value=0.0)
                 - rate_present.reindex(classes, fill_value=0.0)).abs().max()
            )
            rows.append({
                'Column': col,
                'Missing n': int(missing_mask.sum()),
                'Missing %': float(missing_mask.mean()),
                'Max class-rate gap': max_gap,
                'Test': test_name,
                'p-value': float(p),
            })
        effect_col = 'Max class-rate gap'

    if not rows:
        return empty

    assoc = pd.DataFrame(rows)
    # Benjamini-Hochberg. Up to 15 tests at alpha=.05 expects a false positive
    # most of the time, so the number worth reading is the q-value.
    p_vals = assoc['p-value'].to_numpy(dtype=float)
    q_vals = np.full(len(p_vals), np.nan)
    finite_idx = np.where(np.isfinite(p_vals))[0]
    if len(finite_idx) > 0:
        order = finite_idx[np.argsort(p_vals[finite_idx], kind='stable')]
        m = len(order)
        stepped = p_vals[order] * m / np.arange(1, m + 1)
        q_vals[order] = np.minimum(np.minimum.accumulate(stepped[::-1])[::-1], 1.0)
    assoc['q-value (BH)'] = q_vals
    assoc = assoc.sort_values('p-value', na_position='last').reset_index(drop=True)

    significant = assoc.loc[assoc['q-value (BH)'] < 0.05, 'Column'].tolist()
    return {
        'table': assoc,
        'n_tested': len(assoc),
        'n_candidates': len(n_missing_by_col),
        'skipped_low_n': skipped,
        'n_significant': len(significant),
        'significant_cols': significant,
        'top': assoc.iloc[0].to_dict(),
        'effect_col': effect_col,
    }


def cohort_split_guidance(
    df: pd.DataFrame,
    target: Optional[str],
    features: List[str],
    signals: Any,
    session_state: Any
) -> Dict[str, Any]:
    """Provide guidance on cohort structure and split strategy."""
    findings = []
    warnings = []
    figures = []
    
    findings.append(f"Cohort type: {signals.cohort_type_final}")
    findings.append(f"Entity ID column: {signals.entity_id_final or 'Not specified'}")
    
    if signals.entity_id_final and signals.entity_id_final in df.columns:
        entity_counts = df[signals.entity_id_final].value_counts()
        median_rows = entity_counts.median()
        mean_rows = entity_counts.mean()
        findings.append(f"Median rows per entity: {median_rows:.1f}")
        findings.append(f"Mean rows per entity: {mean_rows:.1f}")
        findings.append(f"Total unique entities: {len(entity_counts)}")
        
        # Distribution plot
        fig = px.histogram(
            x=entity_counts.values,
            nbins=20,
            title='Distribution of Rows per Entity',
            labels={'x': 'Rows per Entity', 'y': 'Count'}
        )
        figures.append(('plotly', fig))
        
        warnings.append("Must use group-based splitting to prevent data leakage")
        warnings.append("Random splits will leak information across train/test")
    else:
        warnings.append("Entity ID not specified - cannot use group-based splitting")
    
    return {
        'findings': findings,
        'warnings': warnings,
        'figures': figures
    }


def target_profile(
    df: pd.DataFrame,
    target: Optional[str],
    features: List[str],
    signals: Any,
    session_state: Any
) -> Dict[str, Any]:
    """Profile target distribution (regression or classification)."""
    findings = []
    warnings = []
    figures = []
    insights: List[Insight] = []
    
    if not target or target not in df.columns:
        return {
            'findings': ["Target not available"],
            'warnings': [],
            'figures': []
        }
    
    target_series = df[target].dropna()
    
    if signals.task_type_final == 'regression':
        # Histogram
        fig1 = px.histogram(
            target_series,
            nbins=30,
            title=f'Target Distribution: {target}',
            labels={'value': target, 'count': 'Count'}
        )
        figures.append(('plotly', fig1))
        
        # Log histogram if all positive
        if (target_series > 0).all():
            log_target = np.log1p(target_series)
            fig2 = px.histogram(
                log_target,
                nbins=30,
                title=f'Log-Transformed Target Distribution: {target}',
                labels={'value': f'log({target})', 'count': 'Count'}
            )
            figures.append(('plotly', fig2))
            findings.append("Target is positive - log transform may help")
        
        # Outlier summary
        outlier_rate = signals.target_stats.get('outlier_rate', 0)
        skew = signals.target_stats.get('skew', 0)
        findings.append(f"Skewness: {skew:.2f}")
        findings.append(f"Outlier rate: {outlier_rate:.1%}")
        
        if abs(skew) > 1:
            warnings.append("High skewness - consider log transform or robust loss")
        if outlier_rate > 0.05:
            warnings.append(f"High outlier rate ({outlier_rate:.1%}) - consider robust loss")
        
        # Same id as the page-level detector in pages/02_EDA.py: upsert dedupes
        # by id, so whichever runs first writes it and the other refreshes it in
        # place rather than filing the same finding twice.
        if outlier_rate > 0.1:
            insights.append(Insight(
                id="eda_target_outliers",
                source_page="02_EDA",
                category="distribution",
                severity="warning",
                finding=f"High outlier rate in target: {outlier_rate:.1%} of values flagged",
                implication=(
                    "Squared-error losses are dominated by the extremes, so a model can be "
                    "tuned almost entirely by a small tail of observations."
                ),
                recommended_action=(
                    "Use a robust loss (Huber) or tree-based models in Train & Compare, or "
                    "trim the target if the extreme values are measurement artifacts."
                ),
                manuscript_text=(
                    f"{outlier_rate:.1%} of outcome values were flagged as outliers, which "
                    f"inflates squared-error losses and can allow a small number of "
                    f"observations to dominate model fitting"
                ),
                affected_features=[target],
                relevant_pages=["06_Train_and_Compare"],
                model_scope=ISSUE_MODEL_RELEVANCE["outliers"],  # linear, neural, distance
                metadata={"outlier_rate": float(outlier_rate), "skewness": float(skew)},
            ))
    
    elif signals.task_type_final == 'classification':
        # Class counts
        class_counts = target_series.value_counts().sort_index()
        fig = px.bar(
            x=class_counts.index.astype(str),
            y=class_counts.values,
            title=f'Class Distribution: {target}',
            labels={'x': 'Class', 'y': 'Count'}
        )
        figures.append(('plotly', fig))
        
        # Baseline accuracy
        n_classes = len(class_counts)
        if n_classes > 0:
            majority_class_count = class_counts.max()
            baseline_acc = majority_class_count / len(target_series)
            findings.append(f"Classes: {n_classes}")
            findings.append(f"Baseline accuracy (majority class): {baseline_acc:.1%}")
            
            imbalance_ratio = signals.target_stats.get('class_imbalance_ratio', 1.0)
            if imbalance_ratio < 0.5:
                # `GUIDED-049`. Was "consider class weighting", which is the
                # step the registry says damages the property clinical
                # prediction cares about most.
                warnings.append(
                    f"Class imbalance detected (ratio: {imbalance_ratio:.2f})"
                    f" - report PR-AUC and calibration alongside accuracy, and"
                    f" choose the decision threshold explicitly. Rebalancing is"
                    f" contraindicated for a risk model"
                )
    
    return {
        'findings': findings,
        'warnings': warnings,
        'figures': figures,
        'insights': insights,
    }


def dose_response_trends(
    df: pd.DataFrame,
    target: Optional[str],
    features: List[str],
    signals: Any,
    session_state: Any
) -> Dict[str, Any]:
    """Plot dose-response trends for top numeric features."""
    findings = []
    warnings = []
    figures = []
    
    if not target or target not in df.columns:
        return {
            'findings': ["Target not available"],
            'warnings': [],
            'figures': []
        }
    
    numeric_features = [f for f in features if f in signals.numeric_cols and f != target]
    if len(numeric_features) == 0:
        return {
            'findings': ["No numeric features available"],
            'warnings': [],
            'figures': []
        }
    
    # Select top k features by association
    k = min(5, len(numeric_features))
    
    if signals.task_type_final == 'regression':
        # Use correlation
        correlations = []
        for feat in numeric_features:
            corr = abs(df[feat].corr(df[target]))
            if not np.isnan(corr):
                correlations.append((feat, corr))
        correlations.sort(key=lambda x: x[1], reverse=True)
        top_features = [f[0] for f in correlations[:k]]
    else:
        # Use mutual information (sample for speed)
        sample_size = min(1000, len(df))
        df_sample = df.sample(sample_size, random_state=42) if len(df) > sample_size else df
        
        try:
            mi_scores = mutual_info_classif(
                df_sample[numeric_features],
                df_sample[target],
                random_state=42
            )
            feature_mi = list(zip(numeric_features, mi_scores))
            feature_mi.sort(key=lambda x: x[1], reverse=True)
            top_features = [f[0] for f in feature_mi[:k]]
        except:
            # Fallback to correlation
            correlations = []
            for feat in numeric_features:
                corr = abs(df[feat].corr(df[target]))
                if not np.isnan(corr):
                    correlations.append((feat, corr))
            correlations.sort(key=lambda x: x[1], reverse=True)
            top_features = [f[0] for f in correlations[:k]]
    
    # Plot binned trends
    for feat in top_features:
        if feat not in df.columns:
            continue
        
        # Create bins
        feat_data = df[feat].dropna()
        if len(feat_data) < 10:
            continue
        
        n_bins = min(10, len(feat_data) // 10)
        if n_bins < 3:
            continue
        
        bins = pd.qcut(feat_data, q=n_bins, duplicates='drop')
        bin_centers = [interval.mid for interval in bins.cat.categories if pd.notna(interval)]
        bin_labels = df.loc[feat_data.index, target].groupby(bins).mean()
        
        if len(bin_centers) == len(bin_labels):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=bin_centers,
                y=bin_labels.values,
                mode='lines+markers',
                name=feat
            ))
            fig.update_layout(
                title=f'Dose-Response: {feat} vs {target}',
                xaxis_title=feat,
                yaxis_title=f'Mean {target}'
            )
            figures.append(('plotly', fig))
    
    findings.append(f"Analyzed top {len(top_features)} features by association with target")
    
    return {
        'findings': findings,
        'warnings': warnings,
        'figures': figures
    }


def collinearity_map(
    df: pd.DataFrame,
    target: Optional[str],
    features: List[str],
    signals: Any,
    session_state: Any
) -> Dict[str, Any]:
    """Create correlation heatmap for numeric features."""
    findings = []
    warnings = []
    figures = []
    
    numeric_features = [f for f in features if f in signals.numeric_cols]
    if len(numeric_features) < 2:
        return {
            'findings': ["Need at least 2 numeric features for collinearity analysis"],
            'warnings': [],
            'figures': []
        }
    
    # Limit to top 30 by variance
    if len(numeric_features) > 30:
        variances = df[numeric_features].var().sort_values(ascending=False)
        numeric_features = variances.head(30).index.tolist()
        findings.append("Limited to top 30 features by variance")
    
    corr_matrix = df[numeric_features].corr().abs()
    
    fig = px.imshow(
        corr_matrix,
        title='Feature Correlation Heatmap',
        labels=dict(x="Feature", y="Feature", color="|Correlation|"),
        color_continuous_scale='RdBu_r',
        aspect='auto'
    )
    figures.append(('plotly', fig))
    
    # Find high correlation pairs
    high_corr_pairs = signals.collinearity_summary.get('high_corr_pairs', [])
    if high_corr_pairs:
        findings.append(f"Found {len(high_corr_pairs)} highly correlated pairs (>0.85)")
        warnings.append("High collinearity may cause GLM coefficient instability")
    
    # No ledger insight here. pages/02_EDA.py already writes eda_corr_cluster_*
    # from the same signals.collinearity_summary, grouping the correlated
    # features into clusters; a second "max correlation = X" entry would repeat
    # that finding under a different id in the manuscript's limitations list.
    # This function is unreachable in any case — nothing dispatches
    # collinearity_map; its only reference is ml/eda_recommender.py, whose
    # recommend_eda() output pages/02_EDA.py assigns and never renders.
    
    return {
        'findings': findings,
        'warnings': warnings,
        'figures': figures
    }


def leakage_scan(
    df: pd.DataFrame,
    target: Optional[str],
    features: List[str],
    signals: Any,
    session_state: Any
) -> Dict[str, Any]:
    """Scan for target leakage risks."""
    findings = []
    warnings = []
    figures = []
    
    if not target or target not in df.columns:
        return {
            'findings': ["Target not available"],
            'warnings': [],
            'figures': []
        }
    
    # Use leakage candidates from signals
    if signals.leakage_candidate_cols:
        leakage_df = pd.DataFrame({
            'Column': signals.leakage_candidate_cols,
            'Risk': 'High correlation with target'
        })
        figures.append(('table', leakage_df))
        findings.append(f"Found {len(signals.leakage_candidate_cols)} columns with >0.95 correlation to target")
        warnings.append("These columns should be excluded from features to prevent leakage")
    else:
        findings.append("No obvious leakage candidates detected")
    
    return {
        'findings': findings,
        'warnings': warnings,
        'figures': figures
    }


def interaction_analysis(
    df: pd.DataFrame,
    target: Optional[str],
    features: List[str],
    signals: Any,
    session_state: Any
) -> Dict[str, Any]:
    """Analyze interactions with demographic variables."""
    findings = []
    warnings = []
    figures = []
    
    if not target or target not in df.columns:
        return {
            'findings': ["Target not available"],
            'warnings': [],
            'figures': []
        }
    
    # Find demographic columns
    demo_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if any(term in col_lower for term in ['age', 'sex', 'gender', 'bmi']):
            if col in signals.numeric_cols or col in signals.categorical_cols:
                demo_cols.append(col)
    
    if len(demo_cols) == 0:
        return {
            'findings': ["No demographic columns (age/sex/gender/BMI) found"],
            'warnings': [],
            'figures': []
        }
    
    # For each demo column, show stratified trends for top numeric features
    numeric_features = [f for f in features if f in signals.numeric_cols]
    if len(numeric_features) == 0:
        return {
            'findings': ["No numeric features available for interaction analysis"],
            'warnings': [],
            'figures': []
        }
    
    # Select top feature by correlation/MI
    if signals.task_type_final == 'regression':
        correlations = [(f, abs(df[f].corr(df[target]))) for f in numeric_features if not np.isnan(df[f].corr(df[target]))]
        correlations.sort(key=lambda x: x[1], reverse=True)
        top_feature = correlations[0][0] if correlations else None
    else:
        # Use first feature as fallback
        top_feature = numeric_features[0] if numeric_features else None
    
    if top_feature:
        for demo_col in demo_cols[:2]:  # Limit to 2 demo columns
            if demo_col in df.columns and top_feature in df.columns:
                if demo_col in signals.categorical_cols:
                    # Box plot by category
                    fig = px.box(
                        df,
                        x=demo_col,
                        y=target,
                        color=demo_col,
                        title=f'{target} by {demo_col} (stratified)'
                    )
                    figures.append(('plotly', fig))
                else:
                    # Bin demo column and plot
                    demo_binned = pd.qcut(df[demo_col].dropna(), q=3, duplicates='drop', labels=['Low', 'Mid', 'High'])
                    df_temp = df.copy()
                    df_temp['_demo_bin'] = demo_binned
                    fig = px.box(
                        df_temp,
                        x='_demo_bin',
                        y=target,
                        title=f'{target} by {demo_col} (tertiles)'
                    )
                    figures.append(('plotly', fig))
        
        findings.append(f"Analyzed interactions with {len(demo_cols)} demographic variables")
    
    return {
        'findings': findings,
        'warnings': warnings,
        'figures': figures
    }


def outlier_influence(
    df: pd.DataFrame,
    target: Optional[str],
    features: List[str],
    signals: Any,
    session_state: Any
) -> Dict[str, Any]:
    """Analyze outlier influence on regression."""
    findings = []
    warnings = []
    figures = []
    
    if signals.task_type_final != 'regression' or not target:
        return {
            'findings': ["Outlier analysis only available for regression tasks"],
            'warnings': [],
            'figures': []
        }
    
    target_series = df[target].dropna()
    if len(target_series) < 10:
        return {
            'findings': ["Insufficient data for outlier analysis"],
            'warnings': [],
            'figures': []
        }
    
    outlier_method = session_state.get("eda_outlier_method", "iqr")
    outliers, info = detect_outliers(target_series, method=outlier_method)
    n_outliers = outliers.sum()

    if n_outliers > 0:
        fig = px.scatter(
            df,
            x=target,
            y=target,
            color=outliers.reindex(df.index, fill_value=False),
            title=f'Outlier Detection ({outlier_method.upper()}): {target}',
            labels={'color': 'Outlier'}
        )
        figures.append(('plotly', fig))

        findings.append(f"Found {n_outliers} outliers ({n_outliers/len(target_series):.1%}) using {outlier_method.upper()}")
        if info.get("lower") is not None and info.get("upper") is not None:
            findings.append(f"Outlier range: <{info['lower']:.2f} or >{info['upper']:.2f}")
        warnings.append("High outlier rate may require robust loss (Huber) or winsorization")
    else:
        findings.append(f"No outliers detected using {outlier_method.upper()} method")
    
    return {
        'findings': findings,
        'warnings': warnings,
        'figures': figures
    }


def linearity_scatter(
    df: pd.DataFrame,
    target: Optional[str],
    features: List[str],
    signals: Any,
    session_state: Any
) -> Dict[str, Any]:
    """Scatter plots of features vs target (linearity check)."""
    findings = []
    warnings = []
    figures = []
    stats_dict: Dict[str, Any] = {}

    if not target or target not in df.columns:
        return {'findings': ["Target not available"], 'warnings': [], 'figures': [], 'stats': {}}

    numeric = [f for f in features if f in signals.numeric_cols and f != target]
    if not numeric:
        return {'findings': ["No numeric features"], 'warnings': [], 'figures': [], 'stats': {}}

    k = min(6, len(numeric))
    if signals.task_type_final == 'regression':
        corrs = [(f, abs(df[f].corr(df[target]))) for f in numeric if not np.isnan(df[f].corr(df[target]))]
    else:
        try:
            sample = df.sample(min(1000, len(df)), random_state=42) if len(df) > 1000 else df
            mi = mutual_info_classif(sample[numeric], sample[target], random_state=42)
            corrs = list(zip(numeric, [float(m) for m in mi]))
        except Exception:
            corrs = [(f, abs(df[f].corr(df[target]))) for f in numeric if not np.isnan(df[f].corr(df[target]))]
    corrs.sort(key=lambda x: x[1], reverse=True)
    top = [c[0] for c in corrs[:k]]
    stats_dict["feature_correlations"] = corrs[:k]

    if signals.task_type_final == 'regression' and top:
        _norm = normality_check(df[target].dropna().values)
        use_spearman = _norm[1] < 0.05 if not np.isnan(_norm[1]) else False
        method = "spearman" if use_spearman else "pearson"
        corr_with_p = []
        for feat in top:
            r, p, name = correlation_test(df[feat].values, df[target].values, method=method)
            corr_with_p.append((feat, r, p, name))
        stats_dict["correlation_tests"] = corr_with_p
        for feat, r, p, name in corr_with_p[:3]:
            if not np.isnan(p):
                findings.append(f"{feat}: r={r:.3f}, p={p:.4f} ({name})")

    for feat in top:
        if signals.task_type_final == 'regression':
            fig = px.scatter(df, x=feat, y=target, title=f'{target} vs {feat}')
        else:
            fig = px.box(df, x=target, y=feat, title=f'{feat} by {target}')
        figures.append(('plotly', fig))

    findings.append(f"Plotted top {len(top)} features vs target for linearity check.")
    return {'findings': findings, 'warnings': warnings, 'figures': figures, 'stats': stats_dict}


def residual_analysis(
    df: pd.DataFrame,
    target: Optional[str],
    features: List[str],
    signals: Any,
    session_state: Any
) -> Dict[str, Any]:
    """Residual analysis from OLS proxy (pre-training)."""
    findings = []
    warnings = []
    figures = []
    stats_dict: Dict[str, Any] = {}

    if signals.task_type_final != 'regression' or not target:
        return {'findings': ["Residual analysis only for regression"], 'warnings': [], 'figures': [], 'stats': {}}

    numeric = [f for f in features if f in signals.numeric_cols and f != target]
    if len(numeric) < 1:
        return {'findings': ["No numeric features"], 'warnings': [], 'figures': [], 'stats': {}}

    X = df[numeric].fillna(df[numeric].median())
    y = df[target]
    valid = ~(y.isna() | X.isna().any(axis=1))
    X = X[valid].values
    y = y[valid].values
    if len(X) < 10:
        return {'findings': ["Insufficient data"], 'warnings': [], 'figures': [], 'stats': {}}

    from ml.eval import analyze_residuals_extended
    lm = LinearRegression().fit(X, y)
    y_pred = lm.predict(X)
    stats_dict = analyze_residuals_extended(y, y_pred)
    findings.append(f"OLS proxy on {len(numeric)} features; residuals vs fitted.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_pred, y=y - y_pred, mode='markers', name='Residuals'))
    fig.update_layout(title='Residuals vs Fitted (OLS proxy)', xaxis_title='Fitted', yaxis_title='Residuals')
    figures.append(('plotly', fig))

    return {'findings': findings, 'warnings': warnings, 'figures': figures, 'stats': stats_dict}


def _refused_diagnostic(
    reason: str,
    insight_id: str,
    implication: str,
    recommended_action: str,
    manuscript_text: str,
    metadata: Dict[str, Any],
    theory_anchor: str = "",
) -> Dict[str, Any]:
    """The result an OLS-family diagnostic returns when it declines to run.

    Three things make this shape, and all three are the point of the change.

    `findings` is EMPTY, deliberately. `ml.plot_narrative` composes the page's
    **Summary:** line from `stats` and falls through to `findings`, and both the
    VIF and the influence narrative used to end in an all-clear branch — "No
    severe multicollinearity (VIF <= 10)", "No strongly influential points
    detected" — reached whenever the numbers were absent. A refusal that fills
    `findings` with a sentence is therefore worse than the bug it replaces: it
    turns a garbage table into a confident negative result. The narratives are
    now guarded as well (`ml/plot_narrative.py`), but the belt and the braces
    are both cheap and this one is what the tests pin.

    `warnings` carries the sentence, because the page already renders every
    entry with `st.warning` — that is how a headless engine puts text on screen
    without importing Streamlit (`tests/test_engine_is_headless.py`).

    `insights` carries an UNRESOLVED `Insight` with `manuscript_text`, which is
    what `InsightLedger.discussion_points_for_manuscript()` files as a Discussion
    limitation. A refusal has to reach the record too: this app's output is a
    manuscript, and a Methods section that is simply silent about a diagnostic
    the workflow offered is the failure mode the caps exist to prevent.

    `refused` lets the page tell "did not run" from "ran and found nothing" —
    without it, running a refused VIF would still close every open
    `eda_corr_cluster_*` insight on the strength of an analysis that never ran.
    """
    return {
        'findings': [],
        'warnings': [reason],
        'figures': [],
        'stats': {},
        'refused': True,
        'insights': [Insight(
            id=insight_id,
            source_page="02_EDA",
            category="methodology",
            severity="info",
            finding=reason,
            implication=implication,
            recommended_action=recommended_action,
            manuscript_text=manuscript_text,
            relevant_pages=["04_Feature_Selection", "10_Report_Export"],
            theory_anchor=theory_anchor,
            metadata=dict(metadata),
        )],
    }


def influence_diagnostics(
    df: pd.DataFrame,
    target: Optional[str],
    features: List[str],
    signals: Any,
    session_state: Any
) -> Dict[str, Any]:
    """Leverage and Cook's distance from OLS, or a refusal that says why.

    Refuses above `p = n - 2` (`ml.regime.ols_diagnostic_is_defined`). The
    failure above that line is SILENT, which is why it needs a gate and not a
    try/except: `np.linalg.solve` does not raise on the singular Gram matrix at
    any p from 99 to 3,000, so the page's own except clause catches nothing.
    Measured at n=100/p=99 the function returned max leverage
    1.0000000000000566 — leverage is bounded by 1, so that is a mathematically
    impossible statistic printed to four decimals — with max Cook's D 1.03e14
    and a warning that 100 of 100 points are highly influential. At n=500/p=3000
    it flagged 491 of 500.
    """
    findings = []
    warnings = []
    figures = []
    stats_dict: Dict[str, Any] = {}

    if signals.task_type_final != 'regression' or not target:
        return {'findings': ["Influence diagnostics only for regression"], 'warnings': [], 'figures': [], 'stats': {}}

    numeric = [f for f in features if f in signals.numeric_cols and f != target]
    if len(numeric) < 1:
        return {'findings': ["No numeric features"], 'warnings': [], 'figures': [], 'stats': {}}

    X = df[numeric].fillna(df[numeric].median())
    y = df[target]
    valid = ~(y.isna() | X.isna().any(axis=1))

    # Gate on the rows that will actually be fitted, not on len(df) — the
    # dropna above can move n a long way, and p vs n is the whole question.
    # This sits BEFORE X_arr is built on purpose: the hat matrix below is n×n
    # and the Gram matrix is (p+1)×(p+1), so a shape that cannot produce a
    # meaningful answer should not be allocated for either.
    n_obs = int(valid.sum())
    gate = ols_diagnostic_availability(len(numeric), n_obs, "influence")
    if not gate["available"]:
        return _refused_diagnostic(
            reason=gate["reason"],
            insight_id="eda_cap_influence_undefined",
            implication=(
                "No observation was assessed for leverage or influence, so it is "
                "not known whether individual rows drive the fitted relationship."
            ),
            recommended_action=(
                "Reduce the predictor set on Feature Selection until it is "
                f"smaller than the {n_obs:,} usable observations, then re-run."
            ),
            manuscript_text=(
                "regression influence diagnostics were not performed; the number "
                "of predictors exceeded the number of observations"
            ),
            metadata={"p": len(numeric), "n": n_obs,
                      "limit_features": gate["limit_features"]},
        )

    X_arr = np.column_stack([np.ones(valid.sum()), X[valid].values])
    y_arr = y[valid].values
    if len(X_arr) < 10:
        return {'findings': ["Insufficient data"], 'warnings': [], 'figures': [], 'stats': {}}

    lm = LinearRegression().fit(X_arr[:, 1:], y_arr)
    y_pred = lm.predict(X_arr[:, 1:])
    res = y_arr - y_pred
    mse = np.mean(res ** 2) + 1e-12
    H = X_arr @ np.linalg.solve(X_arr.T @ X_arr, X_arr.T)
    h = np.diag(H)
    k = X_arr.shape[1]
    cook = (res ** 2 / (k * mse)) * (h / (1 - h) ** 2)

    stats_dict["max_leverage"] = float(np.max(h))
    stats_dict["max_cooks"] = float(np.max(cook))

    # The conventional "high leverage" line is h > 2k/n with k = p+1. Leverage
    # is bounded above by 1, so that line stops existing once 2(p+1)/n >= 1,
    # i.e. for every p >= n/2 - 1 — and this band is INSIDE the gate above,
    # which only refuses at p > n - 2. Measured at n=100/p=50 the threshold is
    # 1.020 and the count came back 0 while the largest leverage was 0.68; at
    # n=100/p=99 it came back 0 while every observation had leverage 1.0. The
    # count was structurally zero exactly where leverage is most extreme, so
    # "no high-leverage points" was a statement the arithmetic could not have
    # produced any other answer to. Report the count only where the rule is
    # defined, and say so where it is not.
    lev_threshold = 2 * k / len(X_arr)
    stats_dict["leverage_threshold"] = float(lev_threshold)
    if lev_threshold < 1.0:
        stats_dict["n_high_leverage"] = int((h > lev_threshold).sum())
    else:
        stats_dict["n_high_leverage"] = None
        warnings.append(
            f"High-leverage points were not counted: the usual 2k/n cut-off is "
            f"{lev_threshold:.2f} at {k - 1:,} predictors and {len(X_arr):,} "
            f"observations, and leverage cannot exceed 1, so no observation can "
            f"cross it. Max leverage was {stats_dict['max_leverage']:.4f}."
        )
    stats_dict["n_high_cooks"] = int((cook > 1).sum())

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(h)), y=h, mode='markers', name='Leverage'))
    fig.update_layout(title="Leverage (index)", xaxis_title='Index', yaxis_title='Leverage')
    figures.append(('plotly', fig))
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=np.arange(len(cook)), y=cook, mode='markers', name="Cook's D"))
    fig2.update_layout(title="Cook's distance (index)", xaxis_title='Index', yaxis_title="Cook's D")
    figures.append(('plotly', fig2))

    findings.append(f"Max leverage {stats_dict['max_leverage']:.4f}; max Cook's D {stats_dict['max_cooks']:.4f}.")
    if stats_dict["n_high_cooks"] > 0:
        warnings.append(f"{stats_dict['n_high_cooks']} point(s) with Cook's D > 1 may have high influence.")
    return {'findings': findings, 'warnings': warnings, 'figures': figures, 'stats': stats_dict}


def normality_residuals(
    df: pd.DataFrame,
    target: Optional[str],
    features: List[str],
    signals: Any,
    session_state: Any
) -> Dict[str, Any]:
    """Normality of OLS residuals (Q–Q, Shapiro–Wilk), or a refusal that says why.

    Same gate as `influence_diagnostics`, for a different reason. At p >= n the
    fit is exact — measured in-sample R² = 1.000000 with residual sd between
    2.3e-15 and 3.2e-15 at n=500/p=3000, n=100/p=200 and n=100/p=99 — so
    Shapiro–Wilk is applied to floating-point rounding error and its verdict is
    a function of the rounding pattern rather than of the data. Re-measured
    across seeds it came back p=0.79, 0.96 and 0.70 at n=500/p=3000 and p=0.15,
    0.43 and 0.97 at n=100/p=200: noise in both directions, landing on either
    side of 0.05 at random. Reporting either verdict is reporting nothing.
    """
    findings = []
    warnings = []
    figures = []
    stats_dict: Dict[str, Any] = {}

    if signals.task_type_final != 'regression' or not target:
        return {'findings': ["Normality check only for regression"], 'warnings': [], 'figures': [], 'stats': {}}

    numeric = [f for f in features if f in signals.numeric_cols and f != target]
    if len(numeric) < 1:
        return {'findings': ["No numeric features"], 'warnings': [], 'figures': [], 'stats': {}}

    X = df[numeric].fillna(df[numeric].median())
    y = df[target]
    valid = ~(y.isna() | X.isna().any(axis=1))

    n_obs = int(valid.sum())
    gate = ols_diagnostic_availability(len(numeric), n_obs, "normality")
    if not gate["available"]:
        return _refused_diagnostic(
            reason=gate["reason"],
            insight_id="eda_cap_normality_undefined",
            implication=(
                "Whether the residuals of a linear model on these data are "
                "approximately normal is unknown, so parametric confidence "
                "intervals and p-values from such a model are unverified."
            ),
            recommended_action=(
                "Reduce the predictor set on Feature Selection until it is "
                f"smaller than the {n_obs:,} usable observations, then re-run — "
                "or prefer bootstrap intervals, which do not rest on this "
                "assumption."
            ),
            manuscript_text=(
                "residual normality was not tested; the number of predictors "
                "exceeded the number of observations, so the diagnostic fit "
                "reproduced the outcome exactly and left no residual to test"
            ),
            metadata={"p": len(numeric), "n": n_obs,
                      "limit_features": gate["limit_features"]},
        )

    X = X[valid].values
    y = y[valid].values
    if len(X) < 10:
        return {'findings': ["Insufficient data"], 'warnings': [], 'figures': [], 'stats': {}}

    lm = LinearRegression().fit(X, y)
    res = (y - lm.predict(X)).ravel()
    osq, osr = stats.probplot(res, dist='norm')
    stats_dict["shapiro_stat"], stats_dict["shapiro_p"] = stats.shapiro(res[:min(5000, len(res))])

    slope, inter = np.polyfit(osq[0], osq[1], 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=osq[0], y=osq[1], mode='markers', name='Residuals'))
    fig.add_trace(go.Scatter(x=osq[0], y=slope * np.array(osq[0]) + inter, mode='lines', name='Normal'))
    fig.update_layout(title='Q–Q plot of residuals', xaxis_title='Theoretical', yaxis_title='Sample')
    figures.append(('plotly', fig))

    findings.append(f"Shapiro–Wilk p={stats_dict['shapiro_p']:.4f}.")
    if stats_dict['shapiro_p'] < 0.05:
        warnings.append("Residuals deviate from normality (p < 0.05); inference may be affected.")
    return {'findings': findings, 'warnings': warnings, 'figures': figures, 'stats': stats_dict}


def multicollinearity_vif(
    df: pd.DataFrame,
    target: Optional[str],
    features: List[str],
    signals: Any,
    session_state: Any
) -> Dict[str, Any]:
    """VIF table for numeric features, or a refusal that says why.

    Two gates, both held in `ml.regime.vif_availability` so that the threshold
    and the sentence disclosing it live in one place.

    The VALIDITY gate is the important half and it is a correctness fix, not a
    cost cap. BEFORE, once p >= n every one of these regressions fit exactly,
    `r2 == 1` for every feature, and the loop appended the literal 999.0 — a
    sentinel that sorts, formats and reads exactly like a measurement, then
    tripped the fixed "VIF > 10" alarm on every row. Measured on i.i.d. normals
    whose true VIF is 1 BY CONSTRUCTION: n=100/p=100 returned 100 sentinels out
    of 100 and flagged all 100; n=500/p=500 returned 497 sentinels beside three
    finite values as large as 1.56e4, and flagged all 500. The table asserted
    severe multicollinearity everywhere about data that contained none.

    The estimator stops being readable well below that. On independent features
    E[VIF] = (n-1)/(n-p), confirmed at two sample sizes, so at n=500 the measured
    median is 2.02 at p=250 and 9.82 at p=450 — where 203 of 450 independent
    features cross a fixed 10. That is why the gate is `p <= n/2` and why the
    flag threshold below is scaled by the null baseline instead of being a
    constant: "VIF > 10" means different things at p/n = 0.05 and p/n = 0.4, and
    only the scaled version means the same thing at both.

    The WALL-TIME gate is the cheap half: measured at n=500, 4.32 s at p=200,
    202 s at p=800, and at p=1,000 it blew through a 900 s cap while saturating
    nine cores — censored, not finished.
    """
    findings = []
    warnings = []
    figures = []
    stats_dict: Dict[str, Any] = {}

    numeric = [f for f in features if f in signals.numeric_cols]
    # n is len(df) here rather than a post-dropna count because this function
    # median-fills instead of dropping rows; every row below is fitted.
    gate = vif_availability(len(numeric), len(df))
    if not gate["available"]:
        # Two very different refusals share this exit, and the record has to
        # tell them apart: "too few predictors to ask the question" is not the
        # same limitation as "too many for the estimator to answer it".
        too_few = gate["n_features"] < 2
        return _refused_diagnostic(
            reason=gate["reason"],
            insight_id="eda_cap_vif_refused",
            implication=(
                "Multicollinearity among the predictors was not quantified, so "
                "the stability of the coefficients of any linear model fitted on "
                "them is unknown from this page."
            ),
            recommended_action=(
                "Select at least two numeric predictors and re-run."
                if too_few else
                "Reduce the predictor set on Feature Selection and re-run, or "
                "read the pairwise collinearity screen above, which is defined "
                "at any width."
            ),
            manuscript_text=(
                "variance inflation factors were not computed; fewer than two "
                "numeric predictors were available"
                if too_few else
                "variance inflation factors were not computed; the analysis "
                "carried more predictors than the estimator supports"
            ),
            theory_anchor="collinearity",
            metadata={
                "p": gate["n_features"],
                "n": gate["n_rows"],
                "limit_features": gate["limit_features"],
                "limit_ratio": gate["limit_ratio"],
                "null_baseline_vif": gate["null_baseline_vif"],
            },
        )

    baseline = float(gate["null_baseline_vif"])
    flag_threshold = float(gate["flag_threshold"])

    X = df[numeric].fillna(df[numeric].median())
    vifs = []
    for i, col in enumerate(numeric):
        other = [c for j, c in enumerate(numeric) if j != i]
        try:
            lm = LinearRegression().fit(X[other], X[col])
            r2 = r2_score(X[col], lm.predict(X[other]))
            # Inside the gate p <= n/2, so an exact fit is no longer the
            # arithmetic artifact it was above it: r2 == 1 here means the column
            # genuinely IS a linear combination of the others. Report that as
            # infinity, which is what it is. Never 999.0 — a finite number in
            # this column is read as a measurement, and it sorted a degenerate
            # fit in among real ones.
            vif = 1 / (1 - r2) if r2 < 1 else float('inf')
        except Exception:
            vif = None  # the fit failed; None renders as blank, not as a value
        vifs.append((col, float(vif) if vif is not None else None))

    stats_dict["vif"] = vifs
    stats_dict["vif_null_baseline"] = baseline
    stats_dict["vif_flag_threshold"] = flag_threshold
    stats_dict["n_rows"] = int(len(df))
    stats_dict["n_features"] = len(numeric)
    vif_df = pd.DataFrame([
        {
            "Feature": c,
            "VIF": v,
            # What this shape produces on features with no collinearity at all,
            # so the reader can see how much of a VIF is the data and how much
            # is p/n. At p/n = 0.5 the baseline is 2.0 and half of any reported
            # VIF is sample size.
            "VIF if uncorrelated": round(baseline, 2),
            # Kept numeric (inf included) so the column sorts; a mixed
            # float/string column silently stops sorting in the table widget.
            "Ratio to that": (v / baseline) if v is not None else None,
        }
        for c, v in vifs
    ])
    figures.append(('table', vif_df))

    high = [c for c, v in vifs if v is not None and v > flag_threshold]
    findings.append(
        f"VIF computed for {len(numeric)} features on {len(df):,} observations "
        f"(p/n = {len(numeric) / len(df):.2f}). With no collinearity at all this "
        f"shape yields VIF ≈ {baseline:.2f}, so the flag threshold here is "
        f"{flag_threshold:.1f} rather than a fixed 10."
    )
    if high:
        warnings.append(
            f"VIF > {flag_threshold:.1f}: {', '.join(high)}; consider dropping or regularizing."
        )
    return {'findings': findings, 'warnings': warnings, 'figures': figures, 'stats': stats_dict}


def data_sufficiency_check(
    df: pd.DataFrame,
    target: Optional[str],
    features: List[str],
    signals: Any,
    session_state: Any
) -> Dict[str, Any]:
    """Data sufficiency: n, p, n/p."""
    findings = []
    warnings = []
    figures = []
    n, p = len(df), len(features)
    ratio = n / p if p else 0
    stats_dict = {"n_rows": n, "n_features": p, "ratio": ratio}

    tbl = pd.DataFrame([{"Metric": "Samples", "Value": n}, {"Metric": "Features", "Value": p}, {"Metric": "n/p", "Value": f"{ratio:.1f}"}])
    figures.append(('table', tbl))
    findings.append(f"n={n:,}, p={p}; n/p={ratio:.1f}.")
    if ratio < 20:
        warnings.append("n/p < 20; consider more data or fewer features for stable models.")
    return {'findings': findings, 'warnings': warnings, 'figures': figures, 'stats': stats_dict}


def feature_scaling_check(
    df: pd.DataFrame,
    target: Optional[str],
    features: List[str],
    signals: Any,
    session_state: Any
) -> Dict[str, Any]:
    """Min, max, std, range, and outlier summary per feature with scaling recommendation."""
    findings = []
    warnings = []
    figures = []
    numeric = [f for f in features if f in df.columns and np.issubdtype(df[f].dtype, np.number)]
    if not numeric:
        return {'findings': ["No numeric features"], 'warnings': [], 'figures': [], 'stats': {}}

    capped = numeric[:20]
    rows = []
    ranges = []
    features_with_outliers = []
    for f in capped:
        s = df[f].dropna()
        fmin, fmax, fstd = s.min(), s.max(), s.std()
        frange = fmax - fmin
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        has_outliers = bool((s > q3 + 1.5 * iqr).any() or (s < q1 - 1.5 * iqr).any())
        if has_outliers:
            features_with_outliers.append(f)
        ranges.append(frange)
        rows.append({
            "Feature": f,
            "Min": f"{fmin:.4f}",
            "Max": f"{fmax:.4f}",
            "Std": f"{fstd:.4f}",
            "Range": f"{frange:.4f}",
            "Outliers (IQR)": "Yes" if has_outliers else "No",
        })
    figures.append(('table', pd.DataFrame(rows)))

    # Range ratio and warning
    positive_ranges = [r for r in ranges if r > 0]
    if len(positive_ranges) >= 2:
        max_range = max(positive_ranges)
        min_range = min(positive_ranges)
        range_ratio = max_range / min_range
        if range_ratio > 100:
            warnings.append(
                f"Feature range ratio is {range_ratio:.1f}x (max range / min range). "
                "Large differences in scale can hurt linear models, SVM, KNN, and neural networks. "
                "Consider scaling your features."
            )
    else:
        range_ratio = None

    # Horizontal bar chart of feature ranges (log scale)
    if ranges:
        chart_features = capped[:len(ranges)]
        fig = go.Figure(go.Bar(
            x=ranges,
            y=chart_features,
            orientation='h',
            marker_color='steelblue',
        ))
        fig.update_layout(
            title="Feature Ranges (log scale)",
            xaxis_title="Range (max − min)",
            yaxis_title="Feature",
            xaxis_type="log",
            height=max(300, 28 * len(chart_features)),
            margin=dict(l=160, r=20, t=40, b=40),
        )
        figures.append(('plotly', fig))

    # Recommendation
    if features_with_outliers:
        scaler_rec = "RobustScaler (outliers detected in: " + ", ".join(features_with_outliers[:5])
        if len(features_with_outliers) > 5:
            scaler_rec += f" +{len(features_with_outliers) - 5} more"
        scaler_rec += ")"
    else:
        scaler_rec = "StandardScaler (no significant outliers detected)"

    ratio_str = f"{range_ratio:.1f}x" if range_ratio is not None else "N/A"
    findings.append(
        f"Range ratio is {ratio_str}. "
        f"Recommended scaler: {scaler_rec}. "
        "Scaling recommended for: linear models, SVM, KNN, neural networks. "
        "Not needed for: tree-based models."
    )
    findings.append(f"Scaling summary for {len(capped)} of {len(numeric)} numeric features.")

    return {
        'findings': findings,
        'warnings': warnings,
        'figures': figures,
        'stats': {
            'range_ratio': range_ratio,
            'features_with_outliers': features_with_outliers,
            'n_numeric': len(numeric),
        },
    }


# MERGE NOTE (TurboTab <- main): main's e24a534 DELETED quick_probe_baselines,
# because it "split the full frame and never consulted the lockbox, so its
# held-out-looking scores were fit partly on sealed rows". TurboTab fixed that
# same defect at the caller instead: pages/02_EDA.py lists this function in
# _TRAIN_ONLY_ACTIONS and hands it train_row_mask()-scoped rows only, saying so
# on screen. Three TurboTab test files import it by name
# (test_the_quick_baseline_does_not_leak.py,
# test_eda_does_not_model_on_sealed_rows.py,
# test_the_manuscript_does_not_assert_an_uncorrected_count.py), so it is kept.
# It no longer has a UI entry point: main removed the recommendation-card panel
# that offered it. HUMAN REVIEW: either re-list it from a section on the page or
# take main's deletion and retire the three test files with it.


def quick_probe_baselines(
    df: pd.DataFrame,
    target: Optional[str],
    features: List[str],
    signals: Any,
    session_state: Any
) -> Dict[str, Any]:
    """Run quick baseline models (constant, simple GLM, shallow RF)."""
    findings = []
    warnings = []
    figures = []
    
    if not target or target not in df.columns:
        return {
            'findings': ["Target not available"],
            'warnings': [],
            'figures': []
        }
    
    if len(features) == 0:
        return {
            'findings': ["No features selected"],
            'warnings': [],
            'figures': []
        }
    
    # Prepare data
    X = df[features].select_dtypes(include=[np.number])
    y = df[target]

    # THE N CASCADE, REPORTED (`AUDIT-004`). This mask deletes every row with a
    # missing value in the target OR in any one of the features, and the
    # numbers below used to be presented with no statement of what they were
    # about. `research/NUTRITION_PACK.md` §06 lists *silent listwise deletion
    # with no N cascade* first among its anti-patterns, and this project
    # already has the vocabulary: an exclusion that changes N is reported in
    # participant flow, because a reported n that is not the n the reader
    # assumes is the same defect as an uncorrected count.
    #
    # On a wide table with scattered missingness listwise deletion can remove
    # most of the rows, and an MAE computed on whoever happened to be complete
    # is a number about a subset nobody named.
    n_supplied = int(len(df))
    n_features_requested = len(features)
    n_features_numeric = int(X.shape[1])
    valid_mask = ~(y.isnull() | X.isnull().any(axis=1))
    n_used = int(valid_mask.sum())
    n_dropped = n_supplied - n_used
    X = X[valid_mask]
    y = y[valid_mask]

    if n_dropped:
        findings.append(
            f"Baselines were fitted on {n_used:,} of {n_supplied:,} rows — "
            f"{n_dropped:,} were removed for a missing value in the target or "
            f"in one of the {n_features_numeric} features. Every number below "
            f"is about those {n_used:,} rows."
        )
    if n_features_numeric < n_features_requested:
        findings.append(
            f"{n_features_numeric} of {n_features_requested} selected features "
            f"are numeric and were used; the rest were left out of these "
            f"probes rather than encoded."
        )

    if len(X) < 10:
        return {
            'findings': findings + ["Insufficient data for baseline models"],
            'warnings': warnings,
            'figures': []
        }

    # THE SPLIT, THROUGH THE VETTED SPLITTER (`AUDIT-002`).
    #
    # This was `train_test_split(X, y, test_size=0.2, random_state=42)` — rows
    # divided at random with no group awareness. On a table with repeated
    # measures one person's rows land on both sides and the MAE below is
    # optimistic. `research/NUTRITION_PACK.md` §03 states it as a
    # TurboTab-specific item: *if a person contributes multiple recalls, rows
    # from the same person must never be split across train and test folds —
    # use participant-level splitting.* `METABOLOMICS_PACK.md` §10 lists
    # *repeated measures treated as independent* under Structural.
    #
    # **The answer was already recorded and this path was not reading it.**
    # `DatasetSignals` carries `cohort_type_final` and `entity_id_final` — the
    # app asked whether the cohort is longitudinal and what identifies an
    # entity, and the user answered. `ml/splits.py` has implemented the grouped
    # basis the whole time, with GroupShuffleSplit and the priority order that
    # puts grouping first because a subject spanning partitions is the worse
    # leak. Both existed; this function used neither.
    #
    # 0.70/0.10/0.20 through `make_split` and then train ∪ val as the fitting
    # set, so the 80/20 shape is unchanged and nothing here reimplements a
    # partition. **Expect these numbers to be WORSE than the leaking ones on a
    # longitudinal table.** That is the app becoming correct.
    from ml.splits import SplitError, SplitSpec, make_split

    entity_col = getattr(signals, 'entity_id_final', None)
    longitudinal = getattr(signals, 'cohort_type_final', None) == 'longitudinal'
    grouped = bool(longitudinal and entity_col and entity_col in df.columns)

    probe_frame = df.loc[valid_mask, list(X.columns)].copy()
    probe_frame[target] = y
    if grouped:
        probe_frame[entity_col] = df.loc[valid_mask, entity_col]

    spec = SplitSpec(train_size=0.70, val_size=0.10, test_size=0.20,
                     random_state=42, use_group_split=grouped,
                     entity_id_col=entity_col if grouped else None)
    try:
        split = make_split(probe_frame, list(X.columns), target,
                           signals.task_type_final or 'regression', spec)
    except SplitError as exc:
        return {
            'findings': findings,
            'warnings': warnings + [f"Baselines were not run: {exc}"],
            'figures': []
        }

    X_train = pd.concat([split.X_train, split.X_val])
    y_train = pd.Series(np.concatenate([split.y_train, split.y_val]))
    X_test, y_test = split.X_test, pd.Series(split.y_test)

    if grouped:
        findings.append(
            f"Rows were split by `{entity_col}` rather than at random, so no "
            f"{entity_col} appears in both the fitting and the held-out set. "
            f"On a table with repeated measures a random split puts one "
            f"person's rows on both sides and the numbers below come out "
            f"better than they are."
        )
    elif entity_col and entity_col in df.columns:
        findings.append(
            f"Rows were split at random. `{entity_col}` identifies entities in "
            f"this table but the cohort is not recorded as longitudinal, so "
            f"nothing here says a row repeats — if it does, answer that "
            f"question and these numbers will change."
        )

    results = []

    if signals.task_type_final == 'regression':
        # Constant predictor (mean)
        constant_pred = np.full(len(y_test), y_train.mean())
        mae_const = mean_absolute_error(y_test, constant_pred)
        rmse_const = np.sqrt(mean_squared_error(y_test, constant_pred))
        r2_const = r2_score(y_test, constant_pred)
        results.append({
            'Model': 'Constant (Mean)',
            'MAE': f"{mae_const:.3f}",
            'RMSE': f"{rmse_const:.3f}",
            'R²': f"{r2_const:.3f}"
        })
        
        # Simple GLM
        try:
            glm = LinearRegression()
            glm.fit(X_train, y_train)
            y_pred_glm = glm.predict(X_test)
            mae_glm = mean_absolute_error(y_test, y_pred_glm)
            rmse_glm = np.sqrt(mean_squared_error(y_test, y_pred_glm))
            r2_glm = r2_score(y_test, y_pred_glm)
            results.append({
                'Model': 'GLM (OLS)',
                'MAE': f"{mae_glm:.3f}",
                'RMSE': f"{rmse_glm:.3f}",
                'R²': f"{r2_glm:.3f}"
            })
        except Exception as e:
            warnings.append(f"GLM failed: {str(e)}")
        
        # Shallow RF
        try:
            rf = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)
            mae_rf = mean_absolute_error(y_test, y_pred_rf)
            rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
            r2_rf = r2_score(y_test, y_pred_rf)
            results.append({
                'Model': 'RF (10 trees, depth=3)',
                'MAE': f"{mae_rf:.3f}",
                'RMSE': f"{rmse_rf:.3f}",
                'R²': f"{r2_rf:.3f}"
            })
        except Exception as e:
            warnings.append(f"RF failed: {str(e)}")
    
    else:  # classification
        # Constant predictor (majority class)
        majority_class = y_train.mode()[0] if len(y_train.mode()) > 0 else y_train.iloc[0]
        constant_pred = np.full(len(y_test), majority_class)
        acc_const = accuracy_score(y_test, constant_pred)
        f1_const = f1_score(y_test, constant_pred, average='weighted')
        results.append({
            'Model': 'Constant (Majority)',
            'Accuracy': f"{acc_const:.3f}",
            'F1 (weighted)': f"{f1_const:.3f}"
        })
        
        # Simple Logistic
        try:
            logreg = LogisticRegression(max_iter=500, random_state=42)
            logreg.fit(X_train, y_train)
            y_pred_log = logreg.predict(X_test)
            acc_log = accuracy_score(y_test, y_pred_log)
            f1_log = f1_score(y_test, y_pred_log, average='weighted')
            results.append({
                'Model': 'Logistic Regression',
                'Accuracy': f"{acc_log:.3f}",
                'F1 (weighted)': f"{f1_log:.3f}"
            })
        except Exception as e:
            warnings.append(f"Logistic regression failed: {str(e)}")
        
        # Shallow RF
        try:
            rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)
            acc_rf = accuracy_score(y_test, y_pred_rf)
            f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
            results.append({
                'Model': 'RF (10 trees, depth=3)',
                'Accuracy': f"{acc_rf:.3f}",
                'F1 (weighted)': f"{f1_rf:.3f}"
            })
        except Exception as e:
            warnings.append(f"RF failed: {str(e)}")
    
    if results:
        results_df = pd.DataFrame(results)
        figures.append(('table', results_df))
        findings.append(f"Ran {len(results)} baseline models")
        findings.append("These are quick probes only - not saved as trained models")

    # THE BASIS, INSPECTABLE RATHER THAN ASSERTED. `overlap` is the number of
    # entities with rows on both sides — zero is the guarantee, and it is
    # counted here rather than promised in a comment, because a promise nobody
    # can check is what `AUDIT-002` was.
    split_basis = {
        'strategy': split.strategy,
        'entity_column': entity_col if grouped else None,
        'n_fitted': int(len(X_train)),
        'n_held_out': int(len(X_test)),
        'entity_overlap': None,
    }
    if entity_col and entity_col in df.columns:
        entities = df.loc[valid_mask, entity_col]
        fitted = set(entities.loc[list(split.train_labels)
                                  + list(split.val_labels)])
        held = set(entities.loc[list(split.test_labels)])
        split_basis['entity_overlap'] = int(len(fitted & held))

    return {
        'findings': findings,
        'warnings': warnings,
        'figures': figures,
        'split_basis': split_basis,
    }


# =============================================================================
# Recommendation panel → InsightLedger  (`AUDIT-032`)
# =============================================================================
#
# `AUDIT-032`, corrected in the shape `AUDIT-021` uses — the claim narrowed to
# the one the surface can keep, not deleted and not blurred.
#
# BEFORE: `pages/02_EDA.py::_resolve_insights_from_eda_result` called
#   `ledger.resolve(...)` for every insight matching the action, with
#   `resolution_details={"action_type": "diagnostic_analysis", ...}`. A resolved
#   insight is counted by `InsightLedger.narrative_for_report` under *"N were
#   addressed during the modeling workflow"* and printed under *"Addressed
#   observations:"*, and `discussion_points_for_manuscript` skips it outright
#   (`utils/insight_ledger.py:1233` — `if i.resolved: continue`). So pressing
#   **Run Leakage Detection** made the report assert the leakage blocker had
#   been addressed and dropped the caveat the app itself had authored
#   (*"…raising the possibility of information leakage; results including this
#   predictor should be interpreted with caution"*) — while the column was
#   still a model feature.
#
# AFTER: the run is recorded as what it was. The findings are attached to the
#   insight, the insight stays open, and the page says so out loud. Nothing is
#   removed: the diagnostic still reaches the ledger, it just no longer claims
#   an action it did not take.
#
# THE CLASS, not only the instance (`AGENT_ONBOARD.md` §08 check 1): **a
# read-only diagnostic recorded as a resolution.** All five actions below are
# read-only — `leakage_scan` re-reads `signals.leakage_candidate_cols` and
# returns a string, `multicollinearity_vif` computes VIFs, `missingness_scan`
# tabulates rates, `target_profile` describes a distribution,
# `data_sufficiency_check` divides n by p. Not one of them drops a column,
# fills a value or transforms a variable, so not one of them can resolve
# anything. The row named leakage; the same lens over the same function found
# the other four, which is why the whole map moved rather than one key.
#
# The map lives here rather than in `pages/02_EDA.py` because
# `tests/test_eda_ledger_bridge.py` had to keep a hand-copy of both the map and
# the function — a Streamlit page is not importable — and that copy is now the
# only place the old behavior is asserted. One importable definition is the
# fix for that class as well. `ml/` is inside `SCAN_DIRS` in
# `tests/test_insight_id_integrity.py`, and the name is unchanged, so the id
# scanner still sees these prefixes as referenced.

_ACTION_TO_INSIGHT_MAP = {
    "multicollinearity_vif": {"prefix": "eda_corr_cluster_", "category": "collinearity"},
    "leakage_scan": {"prefix": "eda_leakage_", "category": "leakage"},
    "missingness_scan": {"prefix": "eda_missing_", "category": "missing_data"},
    "target_profile": {"exact": ["eda_target_skew"], "category": "target"},
    "data_sufficiency_check": {"exact": ["eda_sufficiency_insufficient", "eda_sufficiency_borderline"], "category": "sufficiency"},
}

# Every key above. Kept as its own name so that adding an action which DOES
# change the data is a deliberate act: it must be added to the map and left out
# of this set, and `record_diagnostic_on_insights` will then refuse it rather
# than silently record it as read-only.
DIAGNOSTIC_ONLY_ACTIONS = frozenset(_ACTION_TO_INSIGHT_MAP)


def diagnostic_disclosure(title: str, n_open: int, n_closed: int = 0) -> str:
    """The sentence the EDA page shows after a recommended analysis has run.

    States the silence rather than leaving it (`AUDIT-028`): a person who has
    just watched a scan report a leakage column would otherwise reasonably read
    the green result as the problem having been handled.

    `n_closed` is the CARVE-OUT (`MISC-092`, `DRIVE-069` finding 13). One of
    these actions does answer the observations it speaks to: running VIF is the
    answer to the pairwise-correlation clusters this page raised, and
    `pages/02_EDA.py` resolves `eda_corr_cluster_*` on that run. The disclosure
    kept saying "it changes nothing. No open observation is waiting on it" —
    true about the DATA, and contradicted two pages later by a coaching panel
    crediting VIF with resolving two observations. When something was closed,
    the sentence says so and says what was closed.
    """
    if n_closed > 0:
        _cnoun = "observation" if n_closed == 1 else "observations"
        closed_clause = (
            f"{title} reads the data and reports — it removed, filled and "
            f"transformed nothing in your dataset. It IS the answer to "
            f"{n_closed} {_cnoun} this page raised, and {'that one is' if n_closed == 1 else 'those are'} "
            f"now recorded as addressed by it."
        )
        if n_open <= 0:
            return closed_clause + " Nothing else is waiting on it."
        _onoun, _overb, _othem = (
            ("observation", "stays", "it") if n_open == 1
            else ("observations", "stay", "them")
        )
        return (
            closed_clause +
            f" A further {n_open} {_onoun} it speaks to {_overb} **open** in "
            f"the report until an action on a later page addresses {_othem}."
        )
    if n_open <= 0:
        return (
            f"{title} reads the data and reports; it changes nothing. "
            f"No open observation is waiting on it."
        )
    noun, verb, them = (
        ("observation", "stays", "it") if n_open == 1
        else ("observations", "stay", "them")
    )
    return (
        f"{title} reads the data and reports — it removed, filled and "
        f"transformed nothing. The {n_open} {noun} it speaks to {verb} "
        f"**open** in the report until an action on a later page addresses "
        f"{them}."
    )


def record_diagnostic_on_insights(ledger, action_id: str, result: dict,
                                  title: str) -> List[str]:
    """Attach a completed diagnostic to the ledger insights it speaks to.

    Returns the ids of the insights that were annotated and left OPEN. The
    return value is the count the page discloses, so a caller cannot render the
    sentence without having done the recording.

    Does not resolve, does not acknowledge and does not touch severity: running
    a diagnostic is evidence about an observation, never an action on it.
    """
    mapping = _ACTION_TO_INSIGHT_MAP.get(action_id)
    if not mapping:
        return []
    # A diagnostic that declined to run is not evidence about anything. Writing
    # it into an insight's `diagnostics_run` history would leave a trail saying
    # the collinearity clusters had been investigated when the investigation was
    # refused — the same class of false credit as resolving them outright.
    if result.get("refused"):
        return []
    if action_id not in DIAGNOSTIC_ONLY_ACTIONS:
        raise ValueError(
            f"{action_id!r} is mapped to insights but is not declared "
            f"read-only; route it through whatever actually performs the "
            f"action rather than through this recorder."
        )

    findings = result.get("findings", []) or []
    warnings = result.get("warnings", []) or []
    stats_ = result.get("stats", {}) or {}

    record = {
        "action_type": "diagnostic_analysis",
        "method": action_id,
        "title": title,
        "findings": list(findings),
        "warnings": list(warnings),
        "changed_the_data": False,
    }
    if stats_:
        record["stats"] = stats_

    exact_ids = mapping.get("exact", [])
    prefix = mapping.get("prefix", "")

    touched: List[str] = []
    for insight in ledger.insights:
        if insight.resolved:
            continue
        if not (insight.id in exact_ids or (prefix and insight.id.startswith(prefix))):
            continue
        # `metadata` is carried by `to_dict`/`from_dict` and by `upsert`, so the
        # record survives a session round-trip and a later re-scan.
        history = insight.metadata.setdefault("diagnostics_run", [])
        if not any(h.get("method") == action_id for h in history):
            history.append(record)
        touched.append(insight.id)
    return touched

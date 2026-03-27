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
import streamlit as st

from ml.eval import calculate_regression_metrics, calculate_classification_metrics
from ml.clinical_units import infer_unit
from ml.physiology_reference import load_reference_bundle, match_variable_key, get_reference_interval
from ml.outliers import detect_outliers
from ml.stats_tests import (
    correlation_test,
    two_sample_location_test,
    categorical_association_test,
    normality_check,
)


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
                    unit_row['Note'] = 'Fasting assumption (reference ranges assume fasting state)'
                else:
                    unit_row['Note'] = ''
                
                unit_inferences.append(unit_row)
                
                # Empirical plausibility from NHANES reference (percentile-based)
                ref_interval = get_reference_interval(nhanes_ref, var_key)
                if inferred_unit_info.get('conversion_factor') and ref_interval:
                    ref_low, ref_high, ref_unit = ref_interval
                    converted = col_data * inferred_unit_info['conversion_factor']

                    below_min = (converted < ref_low).sum()
                    above_max = (converted > ref_high).sum()
                    total_out = below_min + above_max
                    out_rate = total_out / len(col_data)

                    if total_out > 0:
                        out_of_range.append(col)

                    empirical_ranges.append({
                        'Column': col,
                        'Reference Interval (NHANES p01–p99)': f"{ref_low}-{ref_high} {ref_unit}",
                        'Min (canonical)': f"{converted.min():.1f}",
                        'Max (canonical)': f"{converted.max():.1f}",
                        'Out of Range %': f"{out_rate:.1%}" if total_out > 0 else "0%"
                    })

                    if out_rate > 0.05:
                        warnings.append(
                            f"{col}: {out_rate:.1%} values outside NHANES reference interval "
                            f"({ref_low}-{ref_high} {ref_unit}) after conversion from {inferred_unit_info['inferred_unit']}"
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
    
    # Add unit sanity flags from signals
    if signals.physio_plausibility_flags:
        warnings.extend(signals.physio_plausibility_flags)
        findings.append(f"Found {len(signals.physio_plausibility_flags)} empirical plausibility flags")
    
    # Add note about unit overrides
    if unit_overrides:
        findings.append(f"Using {len(unit_overrides)} user-specified unit overrides")
    
    # Add insight if unit issues found
    if signals.physio_plausibility_flags or out_of_range:
        num_flags = len(signals.physio_plausibility_flags) if signals.physio_plausibility_flags else 0
        num_out_of_range = len(out_of_range) if out_of_range else 0
        insight_finding = f"Physiologic plausibility: {num_flags} empirical flags, {num_out_of_range} columns with out-of-range values"
        insight_implication = "Review units and validate values against NHANES reference intervals. Clinical thresholds are informational only."
        try:
            import streamlit as st
            from utils.storyline import add_insight
            add_insight('physio_plausibility', insight_finding, insight_implication, 'data_quality')
        except:
            pass
    
    return {
        'findings': findings,
        'warnings': warnings,
        'figures': figures
    }


def missingness_scan(
    df: pd.DataFrame,
    target: Optional[str],
    features: List[str],
    signals: Any,
    session_state: Any
) -> Dict[str, Any]:
    """Analyze missingness patterns and association with target."""
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
    
    # Missingness vs target association
    if signals.task_type_final == 'regression':
        target_vals = df[target].dropna().values
        _, norm_p, _ = normality_check(target_vals)
        parametric = not (norm_p < 0.05 if not (norm_p != norm_p) else False)
        associations = []
        for col in signals.high_missing_cols[:10]:
            if col in df.columns and col != target:
                missing_mask = df[col].isnull()
                if missing_mask.sum() > 0 and (~missing_mask).sum() > 0:
                    t_m = df.loc[missing_mask, target].dropna().values
                    t_nm = df.loc[~missing_mask, target].dropna().values
                    if len(t_m) >= 2 and len(t_nm) >= 2:
                        stat, p, name = two_sample_location_test(t_m, t_nm, parametric)
                        associations.append({
                            'Column': col,
                            'Target Mean (Missing)': float(np.mean(t_m)),
                            'Target Mean (Non-Missing)': float(np.mean(t_nm)),
                            'Difference': float(abs(np.mean(t_m) - np.mean(t_nm))),
                            'Test': name,
                            'p-value': p,
                        })
        if associations:
            assoc_df = pd.DataFrame(associations).sort_values('Difference', ascending=False)
            figures.append(('table', assoc_df))
            findings.append("Missingness may be informative (associated with target)")
            for row in assoc_df.head(3).itertuples():
                pv = getattr(row, 'p_value', None)
                if pv is not None and np.isfinite(pv):
                    findings.append(f"  {row.Column}: {row.Test} p={pv:.4f}")
            top_assoc = assoc_df.iloc[0] if len(assoc_df) > 0 else None
            if top_assoc is not None and top_assoc['Difference'] > 0:
                insight_finding = f"Missingness in {top_assoc['Column']} correlates with target (Δ={top_assoc['Difference']:.2f})"
                insight_implication = "Add missingness indicator features; consider tree/boosting models that handle missing values natively"
                try:
                    from utils.storyline import add_insight
                    add_insight('missingness_association', insight_finding, insight_implication, 'data_quality')
                except Exception:
                    pass
    elif signals.task_type_final == 'classification':
        associations = []
        for col in signals.high_missing_cols[:10]:
            if col in df.columns and col != target:
                missing_mask = df[col].isnull()
                if missing_mask.sum() > 0 and (~missing_mask).sum() > 0:
                    cont = pd.crosstab(missing_mask, df[target])
                    if cont.size >= 1:
                        stat, p, name = categorical_association_test(cont.values, use_fisher=(cont.shape == (2, 2)))
                        max_diff = 0.0
                        try:
                            mp = df.loc[missing_mask, target].value_counts(normalize=True)
                            np_ = df.loc[~missing_mask, target].value_counts(normalize=True)
                            com = mp.reindex(np_.index, fill_value=0).fillna(0)
                            max_diff = (np_.reindex(com.index, fill_value=0) - com).abs().max()
                        except Exception:
                            pass
                        associations.append({
                            'Column': col,
                            'Max Class Prop Difference': max_diff,
                            'Test': name,
                            'p-value': p,
                        })
        if associations:
            assoc_df = pd.DataFrame(associations).sort_values('Max Class Prop Difference', ascending=False)
            figures.append(('table', assoc_df))
            findings.append("Missingness may be informative (associated with class)")
            for row in assoc_df.head(3).itertuples():
                pv = getattr(row, 'p_value', None)
                if pv is not None and np.isfinite(pv):
                    findings.append(f"  {row.Column}: {row.Test} p={pv:.4f}")
    
    return {
        'findings': findings,
        'warnings': warnings,
        'figures': figures
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
        
        # Add insight for outliers
        if outlier_rate > 0.1:
            insight_finding = f"High outlier rate in target: {outlier_rate:.1%}"
            insight_implication = "Consider robust loss functions (Huber) or tree-based models (RF/ExtraTrees) which are less sensitive to outliers"
            try:
                import streamlit as st
                from utils.storyline import add_insight
                add_insight('target_outliers', insight_finding, insight_implication, 'target_characteristics')
            except:
                pass
    
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
                warnings.append(f"Class imbalance detected (ratio: {imbalance_ratio:.2f}) - consider class weighting")
    
    return {
        'findings': findings,
        'warnings': warnings,
        'figures': figures
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
    max_corr = signals.collinearity_summary.get('max_corr', 0)
    if high_corr_pairs:
        findings.append(f"Found {len(high_corr_pairs)} highly correlated pairs (>0.85)")
        warnings.append("High collinearity may cause GLM coefficient instability")
    
    # Add insight for high collinearity
    if max_corr > 0.85:
        insight_finding = f"High multicollinearity detected: max correlation = {max_corr:.2f}"
        insight_implication = "Use regularized linear models (Ridge/Lasso/ElasticNet) to stabilize coefficients; consider PCA for dimensionality reduction"
        try:
            import streamlit as st
            from utils.storyline import add_insight
            add_insight('collinearity', insight_finding, insight_implication, 'feature_relationships')
        except:
            pass
    
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


def influence_diagnostics(
    df: pd.DataFrame,
    target: Optional[str],
    features: List[str],
    signals: Any,
    session_state: Any
) -> Dict[str, Any]:
    """Leverage and Cook's distance from OLS."""
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
    stats_dict["n_high_leverage"] = int((h > 2 * k / len(X_arr)).sum())
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
    """Normality of OLS residuals (Q–Q, Shapiro–Wilk)."""
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
    """VIF table for numeric features."""
    findings = []
    warnings = []
    figures = []
    stats_dict: Dict[str, Any] = {}

    numeric = [f for f in features if f in signals.numeric_cols]
    if len(numeric) < 2:
        return {'findings': ["Need ≥2 numeric features for VIF"], 'warnings': [], 'figures': [], 'stats': {}}

    X = df[numeric].fillna(df[numeric].median())
    vifs = []
    for i, col in enumerate(numeric):
        other = [c for j, c in enumerate(numeric) if j != i]
        try:
            lm = LinearRegression().fit(X[other], X[col])
            r2 = r2_score(X[col], lm.predict(X[other]))
            vif = 1 / (1 - r2) if r2 < 1 else np.inf
        except Exception:
            vif = np.nan
        vifs.append((col, float(vif) if np.isfinite(vif) else 999.0))

    stats_dict["vif"] = vifs
    vif_df = pd.DataFrame([{"Feature": c, "VIF": v} for c, v in vifs])
    figures.append(('table', vif_df))

    high = [c for c, v in vifs if v > 10]
    findings.append(f"VIF computed for {len(numeric)} features.")
    if high:
        warnings.append(f"VIF > 10: {', '.join(high)}; consider dropping or regularizing.")
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
    
    # Remove rows with missing target
    valid_mask = ~(y.isnull() | X.isnull().any(axis=1))
    X = X[valid_mask]
    y = y[valid_mask]
    
    if len(X) < 10:
        return {
            'findings': ["Insufficient data for baseline models"],
            'warnings': [],
            'figures': []
        }
    
    # Simple train/test split (80/20)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
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
    
    return {
        'findings': findings,
        'warnings': warnings,
        'figures': figures
    }

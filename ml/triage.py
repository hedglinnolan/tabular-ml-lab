"""
Triage and detection logic for task type and cohort structure.
"""
import pandas as pd
import numpy as np

# Dtype questions are asked through pd.api.types, never by comparing a dtype
# object to a string. pandas 3 makes `str` the default dtype for text columns,
# so `dtype in ['object', ...]` stops matching a text column and every branch
# keyed on it silently takes the wrong path (T0-LIVE-004). The predicates below
# answer the question that was actually meant, on both majors.
from pandas.api import types as _pdt

from ml import name_registry as _name_registry


def _is_categorical_like(s) -> bool:
    """Text, category or boolean — anything whose values are labels."""
    return bool(_pdt.is_object_dtype(s) or _pdt.is_string_dtype(s)
                or isinstance(s.dtype, pd.CategoricalDtype) or _pdt.is_bool_dtype(s))


def _is_integer_like(s) -> bool:
    return bool(_pdt.is_integer_dtype(s) and not _pdt.is_bool_dtype(s))


def _is_float_like(s) -> bool:
    return bool(_pdt.is_float_dtype(s))
from typing import Dict, List, Optional, Literal
from datetime import datetime


def detect_task_type(df: pd.DataFrame, target: str) -> Dict:
    """
    Detect task type (regression vs classification) from target column.
    
    Args:
        df: DataFrame with data
        target: Name of target column
        
    Returns:
        Dict with keys:
            - detected: "regression" or "classification"
            - confidence: "low", "med", or "high"
            - reasons: List of explanation strings
    """
    if target not in df.columns:
        return {
            'detected': None,
            'confidence': None,
            'reasons': [f"Target column '{target}' not found in dataframe"]
        }
    
    target_series = df[target]
    n_rows = len(target_series)
    n_unique = target_series.nunique()
    unique_ratio = n_unique / n_rows if n_rows > 0 else 0
    
    reasons = []
    detected = None
    confidence = "low"
    
    # Check dtype first
    if _is_categorical_like(target_series):
        detected = 'classification'
        confidence = 'high'
        reasons.append(f"Target is {target_series.dtype} type (categorical/binary)")
    
    # Check for boolean-like (0/1) numeric
    elif _is_integer_like(target_series):
        unique_vals = sorted(target_series.dropna().unique())
        if len(unique_vals) == 2 and set(unique_vals) == {0, 1}:
            detected = 'classification'
            confidence = 'high'
            reasons.append("Target is binary (0/1) - classification")
        elif n_unique <= 10:
            # Low-cardinality integers are AMBIGUOUS: class codes read this
            # way, but so do counts and 0-10 ratings, which are regression
            # targets. Never assert this confidently — the UI prompts the
            # user to verify or override when confidence is not 'high'.
            detected = 'classification'
            confidence = 'low'
            reasons.append(
                f"Target has {n_unique} unique integer values (≤10) — this often means "
                f"classification, but counts or ordinal scores should be treated as "
                f"regression. Verify or override below."
            )
        elif unique_ratio < 0.02:
            detected = 'classification'
            confidence = 'med'
            reasons.append(f"Target has low cardinality ({n_unique} unique, {unique_ratio:.1%} ratio) - classification")
        else:
            detected = 'regression'
            confidence = 'med'
            reasons.append(f"Target is numeric with {n_unique} unique values ({unique_ratio:.1%} ratio) - regression")
    
    # Float numeric
    elif _is_float_like(target_series):
        if n_unique <= 10:
            detected = 'classification'
            confidence = 'med'
            reasons.append(f"Target has {n_unique} unique float values (≤10) - classification")
        elif unique_ratio < 0.02:
            detected = 'classification'
            confidence = 'low'
            reasons.append(f"Target has low cardinality ({n_unique} unique, {unique_ratio:.1%} ratio) - classification")
        else:
            detected = 'regression'
            confidence = 'high'
            reasons.append(f"Target is continuous numeric ({n_unique} unique values) - regression")
    
    else:
        # Fallback
        detected = 'regression'
        confidence = 'low'
        reasons.append(f"Target dtype '{target_series.dtype}' - defaulting to regression")
    
    return {
        'detected': detected,
        'confidence': confidence,
        'reasons': reasons
    }


def detect_cohort_structure(df: pd.DataFrame, sample_size: int = 1000) -> Dict:
    """
    Detect cohort structure (cross-sectional vs longitudinal).
    
    Args:
        df: DataFrame with data
        sample_size: Number of rows to sample for datetime parsing (for performance)
        
    Returns:
        Dict with keys:
            - detected: "cross_sectional" or "longitudinal"
            - confidence: "low", "med", or "high"
            - reasons: List of explanation strings
            - entity_id_candidates: List of candidate column names
            - entity_id_detected: Best candidate entity ID column (or None)
            - time_column_candidates: List of candidate time/datetime columns
    """
    reasons = []
    entity_id_candidates = []
    time_column_candidates = []
    detected = 'cross_sectional'
    confidence = 'med'
    
    # EXACT KEY OR DECLARED ALIAS; AN UNKNOWN NAME YIELDS SILENCE.
    #
    # These three were substring lists, and the class of error is the one
    # `ml/name_registry.py` records five instances of: `subjective_wellbeing`
    # contains `subject`, so a wellbeing score was a patient identifier;
    # `yearly_income` contains `year`, so it was a time column;
    # `membership_fee`, `encounter_cost` and `recordkeeping_score` were all
    # entity IDs. Each of those then reaches a real decision — whether the data
    # is longitudinal, and therefore how the held-out rows are chosen.
    #
    # Silence is the correct answer for a name nobody declared. `SEQN`,
    # `USUBJID` and `study_id` are real identifiers this will not recognize
    # until somebody adds them, and that is the cost the remedy is paid for:
    # a gap is visible to the user and costs a question, a wrong claim is
    # invisible and licenses a split.
    _ENTITY_ID = _name_registry.build({
        'patient': ['patient_id', 'patientid', 'pat_id', 'ptid'],
        'subject': ['subject_id', 'subjectid', 'subj', 'subj_id', 'usubjid'],
        'person': ['person_id', 'personid'],
        'respondent': ['respondent_id', 'respondentid'],
        'participant': ['participant_id', 'participantid', 'pid'],
        'member': ['member_id', 'memberid'],
        'mrn': ['medical_record_number'],
        'record': ['record_id', 'recordid'],
        'encounter': ['encounter_id', 'encounterid'],
        'seqn': [],
        'study_id': ['studyid'],
    })

    # Names that are measurements rather than identifiers. Kept as an explicit
    # exclusion rather than trusted to fall through, because a measurement that
    # is ALSO a declared identifier spelling is a genuine collision and the
    # registry should say so rather than pick.
    _CLINICAL_MEASUREMENT = _name_registry.build({
        'glucose': ['blood_glucose', 'serum_glucose', 'plasma_glucose'],
        'triglyceride': ['triglycerides'],
        'cholesterol': ['total_cholesterol'],
        'bmi': ['body_mass_index'],
        'waist': ['waist_circumference'],
        'weight': ['weight_kg', 'body_weight'],
        'height': ['height_cm', 'standing_height'],
        'bp': ['blood_pressure', 'bp_sys', 'bp_dia', 'sbp', 'dbp'],
        'kcal': ['energy_kcal', 'calories'],
        'hba1c': ['a1c', 'hemoglobin_a1c'],
        'insulin': [], 'ldl': [], 'hdl': [],
        'creatinine': [], 'albumin': [],
    })

    _TIME = _name_registry.build({
        'date': ['visit_date', 'assessment_date', 'exam_date', 'obs_date'],
        'time': ['visit_time', 'timestamp'],
        'visit': ['visit_number', 'visit_no', 'visitnum'],
        'wave': ['wave_number'],
        'year': ['study_year'],
        'month': ['study_month'],
        'day': ['study_day'],
        'baseline': [],
        'followup': ['follow_up'],
        'timepoint': ['time_point'],
    })

    # Find candidate entity ID columns with stricter criteria
    for col in df.columns:
        col_lower = col.lower()
        # Skip obvious index columns
        if col_lower in ['index', 'row', 'row_number', 'unnamed: 0']:
            continue
        
        # Exclude if it matches clinical measurement patterns
        if _name_registry.match(col, _CLINICAL_MEASUREMENT) is not None:
            continue
        
        # Check for entity ID patterns
        if _name_registry.match(col, _ENTITY_ID) is not None:
            # Additional checks: must be high cardinality and discrete
            n_unique = df[col].nunique()
            unique_ratio = n_unique / len(df) if len(df) > 0 else 0
            
            # Require high cardinality (>= 0.5) and discrete-looking
            is_discrete = (
                _is_integer_like(df[col]) or
                (_is_categorical_like(df[col]) and unique_ratio > 0.5)
            )
            
            # Exclude float continuous columns unless integer-like
            if _is_float_like(df[col]):
                # Check if values are integer-like (small decimal variance)
                sample = df[col].dropna().head(100)
                if len(sample) > 0:
                    decimal_parts = sample - sample.round()
                    if (decimal_parts.abs() > 0.01).any():
                        continue  # Skip if has significant decimals
            
            if unique_ratio >= 0.5 and is_discrete:
                entity_id_candidates.append(col)
    
    # Find candidate time columns
    for col in df.columns:
        col_lower = col.lower()
        if _name_registry.match(col, _TIME) is not None:
            time_column_candidates.append(col)
        # Also check if column is datetime type or can be parsed as datetime
        elif df[col].dtype == 'datetime64[ns]':
            time_column_candidates.append(col)
        elif _is_categorical_like(df[col]):
            # Try to parse a sample
            sample = df[col].dropna().head(min(sample_size, len(df)))
            if len(sample) > 0:
                try:
                    pd.to_datetime(sample.head(10), errors='raise')
                    time_column_candidates.append(col)
                except (ValueError, TypeError):
                    pass
    
    # Remove duplicates
    time_column_candidates = list(set(time_column_candidates))
    
    # Analyze entity ID candidates with evidence of repeated measures
    best_entity_id = None
    best_median_rows = 0
    best_repeat_rate = 0
    
    for entity_col in entity_id_candidates:
        if entity_col not in df.columns:
            continue
        
        # Count rows per entity
        entity_counts = df[entity_col].value_counts()
        median_rows = entity_counts.median()
        
        # Check repeat rate (% of entities with >1 row)
        n_repeated = (entity_counts > 1).sum()
        repeat_rate = n_repeated / len(entity_counts) if len(entity_counts) > 0 else 0
        
        # Require evidence of repeated measures: median >= 2 OR repeat_rate > 10%
        has_repeated_measures = median_rows >= 2 or repeat_rate > 0.1
        
        if has_repeated_measures and (median_rows > best_median_rows or repeat_rate > best_repeat_rate):
            best_median_rows = median_rows
            best_repeat_rate = repeat_rate
            best_entity_id = entity_col
    
    # Decision logic
    if best_entity_id and (best_median_rows >= 2 or best_repeat_rate > 0.1):
        detected = 'longitudinal'
        confidence = 'high' if best_median_rows >= 2 else 'med'
        reasons.append(
            f"Found entity ID column '{best_entity_id}' with median {best_median_rows:.1f} rows per entity "
            f"({best_repeat_rate:.0%} entities repeated) - longitudinal"
        )
    elif best_entity_id and best_median_rows == 1 and best_repeat_rate <= 0.1:
        detected = 'cross_sectional'
        confidence = 'high'
        reasons.append(
            f"Found entity ID column '{best_entity_id}' but only 1 row per entity - cross-sectional"
        )
    elif len(entity_id_candidates) > 0:
        # Had candidates but no repeated measures
        detected = 'cross_sectional'
        confidence = 'med'
        reasons.append(
            f"Found {len(entity_id_candidates)} ID-like columns but no evidence of repeated measures"
        )
    elif len(time_column_candidates) > 0:
        # Check for duplicates suggesting repeated measures
        # If we have time columns, check if there are duplicate keys elsewhere
        # (simple heuristic: check if any non-time column has duplicates)
        has_duplicates = False
        for col in df.columns:
            if col not in time_column_candidates and df[col].duplicated().any():
                has_duplicates = True
                break
        
        if has_duplicates:
            detected = 'longitudinal'
            confidence = 'med'
            reasons.append(
                f"Found time column(s) {time_column_candidates} and duplicate rows - likely longitudinal"
            )
        else:
            detected = 'cross_sectional'
            confidence = 'med'
            reasons.append(
                f"Found time column(s) {time_column_candidates} but no clear repeated measures pattern - cross-sectional"
            )
    else:
        detected = 'cross_sectional'
        confidence = 'med'
        reasons.append("No entity ID or time columns detected - assuming cross-sectional")
    
    return {
        'detected': detected,
        'confidence': confidence,
        'reasons': reasons,
        'entity_id_candidates': entity_id_candidates,
        'entity_id_detected': best_entity_id,
        'time_column_candidates': time_column_candidates
    }

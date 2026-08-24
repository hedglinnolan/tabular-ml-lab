"""
Publication engine: methods section generator, flow diagrams, TRIPOD tracking.

Generates publication-ready text, figures, and compliance checklists.
"""
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Sequence, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone


# ============================================================================
# TRIPOD Checklist
# ============================================================================

TRIPOD_ITEMS = [
    {"id": "1", "section": "Title", "item": "Identify the study as developing and/or validating a prediction model", "auto_key": "title"},
    {"id": "2", "section": "Abstract", "item": "Provide a summary of objectives, study design, setting, participants, sample size, predictors, outcome, statistical analysis, results, and conclusions", "auto_key": "abstract"},
    {"id": "3a", "section": "Introduction", "item": "Explain the medical context and rationale for developing/validating the prediction model", "auto_key": "background"},
    {"id": "3b", "section": "Introduction", "item": "Specify the objectives, including whether developing and/or validating", "auto_key": "objectives"},
    {"id": "4a", "section": "Methods", "item": "Describe the study design or source of data", "auto_key": "study_design"},
    {"id": "4b", "section": "Methods", "item": "Specify the key study dates", "auto_key": "study_dates"},
    {"id": "5a", "section": "Methods", "item": "Specify key elements of the study setting", "auto_key": "setting"},
    {"id": "5b", "section": "Methods", "item": "Describe eligibility criteria for participants", "auto_key": "eligibility"},
    {"id": "6a", "section": "Methods", "item": "Clearly define the outcome", "auto_key": "outcome_defined"},
    {"id": "7a", "section": "Methods", "item": "Clearly define all predictors used in the model", "auto_key": "predictors_defined"},
    {"id": "8", "section": "Methods", "item": "Explain how the study size was arrived at", "auto_key": "sample_size"},
    {"id": "9", "section": "Methods", "item": "Describe how missing data were handled", "auto_key": "missing_data"},
    {"id": "10a", "section": "Methods", "item": "Describe how predictors were handled in the analyses", "auto_key": "predictor_handling"},
    {"id": "10b", "section": "Methods", "item": "Specify type of model, all model-building procedures, and method for internal validation", "auto_key": "model_building"},
    {"id": "10d", "section": "Methods", "item": "Specify all measures used to assess model performance", "auto_key": "performance_measures"},
    {"id": "13a", "section": "Results", "item": "Describe the flow of participants through the study", "auto_key": "participant_flow"},
    {"id": "13b", "section": "Results", "item": "Describe the characteristics of participants", "auto_key": "table1"},
    {"id": "14a", "section": "Results", "item": "Specify the number of participants and outcome events in each analysis", "auto_key": "sample_counts"},
    {"id": "15a", "section": "Results", "item": "Present the full prediction model to allow predictions for individuals", "auto_key": "full_model"},
    {"id": "16", "section": "Results", "item": "Report performance measures with confidence intervals", "auto_key": "performance_ci"},
    {"id": "19a", "section": "Discussion", "item": "Give an overall interpretation of results considering objectives, limitations, and results from similar studies", "auto_key": "interpretation"},
    {"id": "19b", "section": "Discussion", "item": "Discuss any limitations of the study", "auto_key": "limitations"},
]


@dataclass
class TRIPODTracker:
    """Track TRIPOD checklist completion."""
    completed: Dict[str, bool] = field(default_factory=dict)
    notes: Dict[str, str] = field(default_factory=dict)
    page_refs: Dict[str, str] = field(default_factory=dict)

    def mark_complete(self, auto_key: str, note: str = "", page_ref: str = ""):
        self.completed[auto_key] = True
        if note:
            self.notes[auto_key] = note
        if page_ref:
            self.page_refs[auto_key] = page_ref

    def get_progress(self) -> Tuple[int, int]:
        total = len(TRIPOD_ITEMS)
        done = sum(1 for item in TRIPOD_ITEMS if self.completed.get(item["auto_key"], False))
        return done, total

    def get_checklist_df(self) -> pd.DataFrame:
        rows = []
        for item in TRIPOD_ITEMS:
            key = item["auto_key"]
            rows.append({
                "Item": item["id"],
                "Section": item["section"],
                "Description": item["item"],
                "Status": "✅" if self.completed.get(key, False) else "⬜",
                "Notes": self.notes.get(key, ""),
                "Page": self.page_refs.get(key, ""),
            })
        return pd.DataFrame(rows)


# ============================================================================
# Methods Section Generator
# ============================================================================

def _fmt_param_value(value: Any) -> str:
    """Format numeric parameter values compactly for narrative text."""
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:g}"
    return str(value)


def _calibration_field(cal: Any, name: str) -> Optional[float]:
    """One finite scalar off a calibration record, or None.

    `ml.calibration.CalibrationResult` is a dataclass, but a restored session
    hands back plain dicts; a record type this reader does not recognize must
    contribute nothing rather than a formatted `None`.
    """
    value = cal.get(name) if isinstance(cal, dict) else getattr(cal, name, None)
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _calibration_sentence(model_name: Any, cal: Any) -> Optional[str]:
    """One manuscript line per model, naming only the metrics that were computed.

    The classification and regression records carry different quantities on
    different scales (`AUDIT-017`), and the weak-calibration pair is named as
    such: one word for two meanings is how a reader trusts the wrong number.
    """
    parts: List[str] = []

    brier = _calibration_field(cal, 'brier_score')
    if brier is not None:
        parts.append(f"Brier score = {brier:.4f}")
    ece = _calibration_field(cal, 'ece')
    if ece is not None:
        parts.append(f"ECE = {ece:.4f}")
    weak_slope = _calibration_field(cal, 'weak_slope')
    weak_intercept = _calibration_field(cal, 'weak_intercept')
    if weak_slope is not None and weak_intercept is not None:
        parts.append(
            f"weak calibration slope = {weak_slope:.3f}, intercept = {weak_intercept:.3f}")
    slope = _calibration_field(cal, 'calibration_slope')
    intercept = _calibration_field(cal, 'calibration_intercept')
    if slope is not None and intercept is not None:
        parts.append(
            f"calibration slope = {slope:.3f}, intercept = {intercept:.3f}")

    if not parts:
        return None
    return f"**{_publication_model_label(model_name)}:** {'; '.join(parts)}.\n\n"


def _publication_model_label(model_key: Any) -> str:
    """Return a manuscript-friendly model label."""
    if model_key is None:
        return "Unknown model"

    try:
        from utils.insight_ledger import model_display_name

        return model_display_name(str(model_key))
    except Exception:
        return str(model_key).upper()


def _describe_outlier_handling(method: str, params: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Return a specific outlier-handling description when reliable params exist."""
    params = params or {}
    method = (method or "none").lower()

    if method == "percentile":
        lower = params.get("lower_percentile")
        upper = params.get("upper_percentile")
        # Also accept quantile-fraction keys (lower_q / upper_q) stored by the
        # preprocessing page and convert 0-1 fractions to percentile integers.
        if lower is None and "lower_q" in params:
            lower = params["lower_q"] * 100
        if upper is None and "upper_q" in params:
            upper = params["upper_q"] * 100
        if lower is not None and upper is not None:
            return (
                f"outliers clipped at the {_fmt_param_value(lower)}th and "
                f"{_fmt_param_value(upper)}th percentiles"
            )
        return "outliers addressed using percentile-based winsorization"

    if method == "mad":
        threshold = (
            params.get("threshold")
            or params.get("mad_threshold")
            or params.get("n_mad")
        )
        if threshold is not None:
            return f"outliers clipped using a MAD threshold of {_fmt_param_value(threshold)}"
        return "outliers addressed using MAD-based outlier clipping"

    if method == "iqr":
        multiplier = params.get("multiplier") or params.get("iqr_multiplier")
        if multiplier is not None:
            return f"outliers removed using an IQR multiplier of {_fmt_param_value(multiplier)}"
        return "outliers addressed using IQR-based outlier removal"

    return None


def pca_technique_sentence(pca_details: Dict[Any, Dict[str, Any]]) -> str:
    """The Methods clause for PCA, over the PER-MODEL preprocessing configs.

    `MINE-025`: this used to be `next(iter(pca_details.values()))` under the
    comment "use the first PCA config as representative". The dict is
    insertion-ordered by the Preprocess page's build sequence — checkbox order —
    so the component count the Methods stated as the study's belonged to
    whichever model the analyst happened to tick first. Per-model pipelines are
    the product's differentiator precisely BECAUSE they differ, which makes
    'representative' the one thing a member of that dict is not.

    Collapse only when the configs agree; name them per model when they do not.
    The clause also says where the number comes from: these are PREPROCESSING
    PCA settings, and the sentence they annotate sits in the feature-engineering
    subsection.
    """
    def _detail(d: Dict[str, Any]) -> str:
        mode, n_comp = d.get('mode'), d.get('n_components')
        if mode == "Fixed Components" and n_comp:
            return f"{int(n_comp)} components"
        if mode == "Variance Threshold" and n_comp:
            return f"{int(n_comp * 100)}% of variance retained"
        return "unspecified"

    if not pca_details:
        return "PCA dimensionality reduction"

    details = list(pca_details.values())
    if len({(d.get('mode'), d.get('n_components')) for d in details}) == 1:
        described = _detail(details[0])
        if described == "unspecified":
            return "PCA dimensionality reduction"
        return (f"PCA dimensionality reduction ({described} in each model's "
                f"preprocessing pipeline)")
    per_model = "; ".join(f"{key}: {_detail(d)}" for key, d in pca_details.items())
    return (f"PCA dimensionality reduction, configured per model "
            f"({per_model})")


def generate_methods_from_log() -> Dict[str, List[Dict[str, Any]]]:
    """Extract methodology actions grouped by step from session state log.

    Reads from InsightLedger first (primary), falling back to the legacy
    methodology_log session-state list during migration.

    Returns:
        Dictionary mapping step name to list of log entries for that step.
    """
    try:
        from utils.insight_ledger import get_ledger
        ledger = get_ledger()
        log = ledger.get_methodology_log()
        if not log:
            # Fallback to old format during migration
            import streamlit as st
            log = st.session_state.get('methodology_log', [])
    except (ImportError, Exception):
        try:
            import streamlit as st
            log = st.session_state.get('methodology_log', [])
        except ImportError:
            return {}

    steps = {}
    for entry in log:
        step = entry.get('step', 'Unknown')
        if step not in steps:
            steps[step] = []
        steps[step].append(entry)
    return steps


def _resolve_workflow_feature_counts(
    feature_names: Optional[List[str]],
    logged_steps: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    data_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Optional[int]]:
    """Resolve original/candidate/selected/engineered feature counts consistently.

    Priority is the explicit workflow audit trail, then known session-state artifacts,
    then the function inputs.
    """
    logged_steps = logged_steps or {}
    data_config = data_config or {}

    original_count = None
    candidate_count = None
    selected_count = None
    engineered_count = None

    base_features = data_config.get("feature_cols") or []
    if base_features:
        original_count = len(base_features)

    # `DRIVE-072`. THE SCREENING RUN AND THE APPLY ARE TWO RECORDS, AND STEP
    # NAME CANNOT TELL THEM APART. `utils.insight_ledger._page_to_step_name`
    # maps both back through page `04_Feature_Selection`, so both arrive under
    # `Feature Selection` and `Feature Selection Applied` is empty; `[-1]` was
    # therefore the APPLY entry, which carries no `n_features_before` /
    # `n_features_after`. Both counts fell through to `len(selected_features)`
    # and the Methods draft read "all 14 candidate predictors were retained"
    # over a selection that went 27 → 14. Entries are identified by the fields
    # they carry, which is what actually distinguishes them.
    fs_entries = list(logged_steps.get('Feature Selection', [])) + \
        list(logged_steps.get('Feature Selection Applied', []))
    screening_details: Dict[str, Any] = {}
    applied_details: Dict[str, Any] = {}
    for entry in fs_entries:
        details = entry.get('details') or {}
        if details.get('n_features_before') is not None or details.get('n_features_after') is not None:
            screening_details = details
        if details.get('n_features_selected') is not None:
            applied_details = details

    # ZERO IS A REAL COUNT; ABSENCE IS None. `x or y` swallowed a recorded
    # zero here exactly as it did in `ml/narrative_engine.py`, and a selection
    # that retained nothing then read back as the pre-selection list.
    if screening_details.get('n_features_before') is not None:
        candidate_count = screening_details['n_features_before']
    if screening_details.get('n_features_after') is not None:
        selected_count = screening_details['n_features_after']
    if applied_details.get('n_features_selected') is not None:
        selected_count = applied_details['n_features_selected']

    # THE SELECTION RECORD IS THE AUTHORITY (`CONTRACT-034`). Provenance holds
    # the screened set and the retained set as two separate facts, and page 04's
    # apply updates it in place, so it survives `data_config.feature_cols` being
    # overwritten with the post-apply list. The ledger scrape above is the
    # fallback for a session that never wrote a provenance record.
    _prov_fs = None
    _prov_upload = None
    try:
        from utils.workflow_provenance import get_provenance
        _provenance = get_provenance()
        _prov_fs = getattr(_provenance, 'feature_selection', None)
        _prov_upload = getattr(_provenance, 'upload', None)
    except Exception:
        pass
    if _prov_fs is not None:
        if getattr(_prov_fs, 'n_features_before', None) is not None:
            candidate_count = _prov_fs.n_features_before
        if getattr(_prov_fs, 'n_features_after', None) is not None:
            selected_count = _prov_fs.n_features_after
    if _prov_upload is not None and getattr(_prov_upload, 'n_features', None):
        # The predictor count as OFFERED, which `data_config.feature_cols` no
        # longer holds once a selection has been applied over it.
        original_count = _prov_upload.n_features

    try:
        import streamlit as st
        pre_fe = st.session_state.get('pre_fe_feature_cols') or []
        engineered_names = st.session_state.get('engineered_feature_names') or []
        selected_features = st.session_state.get('selected_features') or []
        if pre_fe:
            original_count = len(pre_fe)
        engineered_count = len(engineered_names) if engineered_names else engineered_count
        engineered_candidate_count = len(pre_fe) + len(engineered_names) if pre_fe else None
        if engineered_candidate_count is not None:
            if candidate_count is None or candidate_count < engineered_candidate_count:
                candidate_count = engineered_candidate_count
        if candidate_count is None and pre_fe:
            candidate_count = len(pre_fe) + (len(engineered_names) if engineered_names else 0)
        if selected_count is None and selected_features:
            selected_count = len(selected_features)
    except ImportError:
        pass

    if selected_count is None and feature_names:
        selected_count = len(feature_names)

    if candidate_count is None:
        candidate_count = selected_count

    if engineered_count is None and original_count is not None and candidate_count is not None:
        engineered_count = max(candidate_count - original_count, 0)

    # The SCREENING step's own two numbers, kept separate from the applied set
    # they are routinely confused with (`DRIVE-072`): the methods ranked
    # `ranked` columns and agreed on `ranked_kept` of them, while
    # `carried_unranked` predictors reached the modeling set without being
    # ranked at all. `selected` is the modeling set; these three explain how it
    # was arrived at, and a section that cannot use them omits them.
    _ranked = screening_details.get('n_features_before')
    _ranked_kept = screening_details.get('n_features_after')
    _carried = applied_details.get('carried_through_unranked')
    return {
        'original': original_count,
        'candidate': candidate_count,
        'selected': selected_count,
        'engineered': engineered_count,
        'ranked': _ranked,
        'ranked_kept': (applied_details.get('n_consensus_ranked')
                        if applied_details.get('n_consensus_ranked') is not None
                        else _ranked_kept),
        'carried_unranked': len(_carried) if isinstance(_carried, (list, tuple)) else None,
    }


def feature_engineering_ran(feature_counts: Optional[Dict[str, Any]]) -> bool:
    """Did the provenance record a feature-engineering stage that CREATED columns?

    `DRIVE8-21`. `candidate` is resolved from the feature-selection record's
    `n_features_before`, which is the count of columns selection could *rank* —
    on a classification run with eight non-numeric predictors that is smaller
    than the original set and has nothing to do with engineering. Every
    manuscript surface then read `candidate != original` as evidence of a
    stage, and the abstract asserted *"feature engineering yielded 19
    candidates"* on a session where page 03 was never opened.

    The only evidence that engineering happened is columns it created. With
    none recorded the FE clause is not written — the funnel sentence states the
    counts the record supports and claims no stage.

    Where no engineered count was resolved at all, the direction of the two
    counts still settles it: engineering only ADDS columns, so `candidate >
    original` is evidence of a stage and `candidate < original` — the drive's
    19 against 27 — never is.
    """
    counts = feature_counts or {}
    engineered = counts.get('engineered')
    if engineered is not None:
        try:
            return int(engineered) > 0
        except (TypeError, ValueError):
            return False
    original, candidate = counts.get('original'), counts.get('candidate')
    try:
        return bool(original and candidate and int(candidate) > int(original))
    except (TypeError, ValueError):
        return False


def _oxford_join(items: List[str]) -> str:
    """Join list items for manuscript prose."""
    cleaned = [str(item).strip() for item in items if str(item).strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"


def _feature_selection_method_label(method: str) -> str:
    """Render feature-selection methods with manuscript-friendly names."""
    labels = {
        'lasso': 'LASSO',
        'rfe': 'RFE-CV',
        'rfe-cv': 'RFE-CV',
        'rfecv': 'RFE-CV',
        'univariate': 'univariate screening',
        'f_regression': 'univariate screening',
        'mutual_info': 'mutual information screening',
        'stability': 'stability selection',
        'stability_selection': 'stability selection',
    }
    key = str(method or '').strip().lower()
    return labels.get(key, str(method or '').strip())


def _consensus_threshold_clause(
    threshold: Optional[int],
    n_completed: Optional[int],
    n_requested: Optional[int],
) -> str:
    """The "at least T of N methods" clause, over ONE universe of methods.

    `MISC-104`. The threshold pages/04 computes is a count of the methods that
    COMPLETED (`max(2, len(results) // 2)`); the denominator printed here was
    the count of the methods that were REQUESTED. With a method that raised,
    the sentence described an agreement rule the run never applied. The
    denominator is now the completed set, and the shortfall is stated rather
    than absorbed. With no record of what completed, the clause names the
    threshold and no denominator — an unknown universe is not a universe.
    """
    if not threshold:
        return ""
    if not n_completed:
        return (f" Features were retained if selected by at least {threshold} "
                f"of the methods that completed.")
    clause = (f" Features were retained if selected by at least {threshold} of "
              f"the {n_completed} method(s) that completed")
    if n_requested and n_requested != n_completed:
        clause += (f" ({n_requested} were requested; "
                   f"{n_requested - n_completed} did not complete)")
    return clause + "."


def _dedupe_latest_by(entries: List[Dict[str, Any]], key_fields: Tuple[str, ...]) -> List[Dict[str, Any]]:
    """Keep only the latest entry for each logical analysis key."""
    latest_by_key: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    ordered_keys: List[Tuple[Any, ...]] = []
    for entry in entries:
        key = tuple(entry.get(field) for field in key_fields)
        if key not in latest_by_key:
            ordered_keys.append(key)
        latest_by_key[key] = entry
    return [latest_by_key[key] for key in ordered_keys]


def _determine_best_model(selected_model_results: Dict[str, Dict], task_type: str) -> Optional[str]:
    """Determine the best model from actual metrics.
    
    Args:
        selected_model_results: Dict of model_name -> {"metrics": {...}}
        task_type: "regression" or "classification"
    
    Returns:
        Name of the best model, or None if no results.
    """
    if not selected_model_results:
        return None
    
    if task_type == "regression":
        # Lowest RMSE wins
        best = min(
            selected_model_results.items(),
            key=lambda x: x[1].get('metrics', {}).get('RMSE', float('inf'))
        )
        return best[0] if best[1].get('metrics', {}).get('RMSE') != float('inf') else None
    else:
        # Highest F1 (or AUC if F1 not available, or Accuracy as fallback)
        def score(item):
            metrics = item[1].get('metrics', {})
            if 'F1' in metrics:
                return metrics['F1']
            elif 'AUC' in metrics:
                return metrics['AUC']
            elif 'Accuracy' in metrics:
                return metrics['Accuracy']
            else:
                return -float('inf')
        
        best = max(selected_model_results.items(), key=score)
        return best[0] if score(best) > -float('inf') else None


def _ordered_metric_items(metrics: Dict[str, Any], task_type: str) -> List[Tuple[str, Any]]:
    """Return metrics in a stable, manuscript-friendly order."""
    preferred = ["RMSE", "MAE", "R2", "MedianAE"] if task_type == "regression" else ["Accuracy", "F1", "AUC"]
    ordered_names = [name for name in preferred if name in metrics] + [name for name in metrics if name not in preferred]
    return [(name, metrics[name]) for name in ordered_names]


def _resolve_manuscript_context(
    manuscript_context: Optional[Dict[str, Any]],
    feature_names: List[str],
    selected_model_results: Optional[Dict[str, Dict]],
    bootstrap_results: Optional[Dict[str, Dict]],
    best_model_name: Optional[str],
) -> Dict[str, Any]:
    """Resolve explicit manuscript facts, preferring export-frozen context over live state."""
    context = manuscript_context or {}
    frozen_feature_names = list(context.get('feature_names_for_manuscript') or feature_names or [])
    frozen_feature_counts = dict(context.get('feature_counts') or {})
    frozen_model_results = context.get('selected_model_results')
    frozen_bootstrap_results = context.get('selected_bootstrap_results')

    return {
        'included_models': list(context.get('included_models') or list((frozen_model_results or selected_model_results or {}).keys())),
        'feature_names_for_manuscript': frozen_feature_names,
        'feature_counts': frozen_feature_counts,
        'selected_model_results': frozen_model_results if frozen_model_results is not None else selected_model_results,
        'selected_bootstrap_results': frozen_bootstrap_results if frozen_bootstrap_results is not None else bootstrap_results,
        'manuscript_primary_model': context.get('manuscript_primary_model'),
        'best_model_by_metric': context.get('best_model_by_metric'),
        'best_metric_name': context.get('best_metric_name'),
    }


#: The exploratory limitation, verbatim. `ml/narrative_engine.py` opens its
#: limitation list with the same clause and `ml/manuscript_validator.py` looks
#: for it in the finished draft — one sentence, three readers, so the wording
#: does not drift out from under the check that gates the export.
EXPLORATORY_LIMITATION_SENTENCE = (
    "The analysis was run in exploratory mode: the held-out test set was not "
    "quarantined from feature engineering and selection, so reported "
    "performance may be optimistically biased and should not be presented as "
    "validated held-out performance."
)


def _exploratory_watermark(manuscript_context: Optional[Dict[str, Any]]) -> bool:
    """Whether this study carries the exploratory watermark.

    `MISC-103`. Frozen export context first, session state second — the same
    precedence `external_validation_record` uses, and for the same reason: the
    context is what the export froze, and a draft composed from it must not be
    re-decided by live state.

    The session read is mode-OR-used, matching pages/10: `exploratory_used` is
    set when the mode is enabled and cleared only by the downstream reset, so
    toggling the mode off cannot launder results computed with it on.
    """
    context = manuscript_context or {}
    if 'exploratory_mode' in context:
        return bool(context.get('exploratory_mode'))
    try:
        import streamlit as st
    except ImportError:
        return False
    try:
        return bool(st.session_state.get('exploratory_mode', False)
                    or st.session_state.get('exploratory_used', False))
    except Exception:
        return False


def external_validation_record() -> Optional[Dict[str, Any]]:
    """What page 07 recorded about an external cohort, or None.

    Provenance first, session state second: the provenance record is what
    survives an export and what a downstream reset nulls, so a manuscript
    composed after the data changed cannot claim an external validation whose
    results the same reset deleted.
    """
    try:
        import streamlit as st
    except ImportError:
        return None
    try:
        prov = st.session_state.get('workflow_provenance')
        ev = getattr(prov, 'external_validation', None) if prov is not None else None
        if ev is not None:
            return {
                'dataset_name': getattr(ev, 'dataset_name', ''),
                'n_rows': int(getattr(ev, 'n_rows', 0) or 0),
                'models': list(getattr(ev, 'models_validated', []) or []),
                'n_bootstrap': int(getattr(ev, 'n_bootstrap', 0) or 0),
                'repairs': list(getattr(ev, 'import_repairs', []) or []),
                'metrics': dict(getattr(ev, 'metrics', {}) or {}),
            }
        stored = st.session_state.get('external_validation_results')
        if isinstance(stored, dict) and stored.get('per_model'):
            return {
                'dataset_name': stored.get('dataset_name', ''),
                'n_rows': int(stored.get('n_rows', 0) or 0),
                'models': list(stored.get('per_model', {}).keys()),
                'n_bootstrap': int(stored.get('n_bootstrap', 0) or 0),
                'repairs': list(stored.get('import_repairs', []) or []),
            }
    except Exception:
        return None
    return None


def external_validation_sentence(record: Optional[Dict[str, Any]]) -> str:
    """The Performance Evaluation sentence for an external cohort.

    Says only what the record holds. With no record the wording stays the
    unqualified one the caller's own flag used to produce — a caller that
    passes external_validation=True is asserting it, not this function.
    """
    if not record:
        return "External validation was performed on an independent dataset. "

    n_rows = record.get('n_rows') or 0
    name = str(record.get('dataset_name') or "").strip()
    models = [m for m in (record.get('models') or [])]
    parts = ["External validation was performed on an independent dataset"]
    if n_rows:
        parts.append(f" of {n_rows:,} rows")
    if name:
        parts.append(f" ({name})")
    parts.append(".")
    if models:
        parts.append(
            f" {_oxford_join([_publication_model_label(m) for m in models])} "
            f"{'were' if len(models) != 1 else 'was'} scored on that cohort")
        n_boot = record.get('n_bootstrap') or 0
        if n_boot:
            parts.append(
                f", with 95% confidence intervals from {int(n_boot):,} "
                f"bootstrap resamples")
        parts.append(".")
    # The numbers themselves, when the record carries them: an external
    # validation the manuscript only ASSERTS happened is the same omission this
    # sentence exists to close.
    metrics = record.get('metrics') or {}
    for model_key, per_metric in metrics.items():
        if not isinstance(per_metric, dict):
            continue
        rendered = []
        for metric_name, vals in per_metric.items():
            if not isinstance(vals, dict) or vals.get('estimate') is None:
                continue
            lo, hi = vals.get('ci_lower'), vals.get('ci_upper')
            if lo is not None and hi is not None:
                rendered.append(f"{metric_name} {float(vals['estimate']):.3f} "
                                f"(95% CI {float(lo):.3f}–{float(hi):.3f})")
            else:
                rendered.append(f"{metric_name} {float(vals['estimate']):.3f}")
        if rendered:
            parts.append(f" On the external cohort, "
                         f"{_publication_model_label(model_key)} achieved "
                         f"{_oxford_join(rendered)}.")

    repairs = record.get('repairs') or []
    if repairs:
        parts.append(
            f" Before validation the external file required "
            f"{len(repairs)} structural repair{'s' if len(repairs) != 1 else ''} "
            f"at import: {'; '.join(str(r) for r in repairs)}.")
    parts.append(" ")
    return "".join(parts)


def generate_methods_section(
    data_config: Dict[str, Any],
    preprocessing_config: Dict[str, Any],
    model_configs: Dict[str, Any],
    split_config: Dict[str, Any],
    n_total: int,
    n_train: int,
    n_val: int,
    n_test: int,
    feature_names: List[str],
    target_name: str,
    task_type: str,
    metrics_used: List[str],
    cv_folds: Optional[int] = None,
    feature_selection_method: Optional[str] = None,
    missing_data_strategy: Optional[str] = None,
    external_validation: bool = False,
    # NEW: actual results for a richer methods/results section
    selected_model_results: Optional[Dict[str, Dict]] = None,
    bootstrap_results: Optional[Dict[str, Dict]] = None,
    best_model_name: Optional[str] = None,
    explainability_methods: Optional[List[str]] = None,
    calibration_results: Optional[Dict[str, Any]] = None,
    random_seed: int = 42,
    manuscript_context: Optional[Dict[str, Any]] = None,
    # NEW: methods parameter precision
    model_hyperparameters: Optional[Dict[str, Dict]] = None,
    hyperparameter_optimization: bool = False,
    split_strategy: Optional[str] = None,
    missing_data_summary: Optional[Dict] = None,
    ledger_narratives: Optional[Dict[str, str]] = None,
    # Which models actually produced cross-validation results. `AUDIT-026`.
    #
    # `use_cv`/`cv_folds` are the CHECKBOX. `model_results[*]['cv_results']` is
    # the only object that records that a fold loop RAN, and the two are
    # routinely different: pages/06:1455 skips CV for the neural network
    # outright, and :1489 catches a CV exception, warns, and leaves
    # `cv_results=None`. A Methods section composed from the checkbox therefore
    # asserts an internal validation design the run never performed.
    #
    # Three states on purpose, and `None` is not `[]` (trap 9): a list — even an
    # empty one — is a caller who KNOWS which models were cross-validated;
    # `None` is a caller who did not say, and ignorance must not be reported as
    # either outcome.
    cv_models_run: Optional[Sequence[str]] = None,
) -> str:
    """Generate a draft methods section for a publication.

    Returns formatted text with placeholders for study-specific details.
    """
    sections = []

    manuscript_facts = _resolve_manuscript_context(
        manuscript_context,
        feature_names,
        selected_model_results,
        bootstrap_results,
        best_model_name,
    )
    feature_names = manuscript_facts['feature_names_for_manuscript']
    selected_model_results = manuscript_facts['selected_model_results']
    bootstrap_results = manuscript_facts['selected_bootstrap_results']

    # `AUDIT-026`. Resolve "did cross-validation actually run, and for what?"
    # once, here, so every CV sentence below reads the same answer.
    #
    # An explicit `cv_models_run` wins. Otherwise derive it from the per-model
    # results, which carry `cv_results` for every trained model — pages/06:1502
    # writes the key unconditionally, `None` when no fold loop ran, so a
    # non-empty results dict is a POSITIVE record either way and not an absence.
    # With neither, the answer is unknown and stays unknown.
    if cv_models_run is not None:
        cv_models_run = [str(m) for m in cv_models_run]
    elif selected_model_results:
        cv_models_run = [
            str(name) for name, res in selected_model_results.items()
            if isinstance(res, dict) and res.get('cv_results')
        ]
    else:
        cv_models_run = None
    cv_scope_known = cv_models_run is not None

    # Feature counts: derive once so manuscript text stays internally consistent.
    logged_steps = generate_methods_from_log()
    feature_counts = manuscript_facts['feature_counts'] or _resolve_workflow_feature_counts(feature_names, logged_steps, data_config)
    _selected_count = feature_counts.get('selected')
    predictor_count = (_selected_count if _selected_count is not None
                       else len(feature_names))

    # Study design
    sections.append("### Study Design and Participants\n")
    sections.append(
        f"[PLACEHOLDER: Describe your study design, data source, and recruitment/selection criteria.] "
        f"This workflow-derived draft documents {n_total:,} observations carried into the modeling workflow, with {predictor_count} predictors in the final modeling set. "
        f"The modeled outcome was {target_name}"
        + (" as a continuous target (regression)." if task_type == "regression"
           else " as a categorical target (classification).")
    )

    # Data Cleaning (FIX 1)
    data_cleaning_entries = logged_steps.get('Data Cleaning', [])
    if data_cleaning_entries:
        sections.append("\n\n### Data Cleaning\n")
        cleaning_actions = []
        total_rows_removed = 0
        
        for entry in data_cleaning_entries:
            action = entry.get('action', '')
            details = entry.get('details', {})
            rows_before = details.get('rows_before', 0)
            rows_after = details.get('rows_after', 0)
            rows_removed = rows_before - rows_after
            
            if action:
                cleaning_actions.append(action)
            if rows_removed > 0:
                total_rows_removed += rows_removed
        
        if cleaning_actions:
            actions_list = ", ".join(cleaning_actions)
            sections.append(
                f"Prior to analysis, {len(data_cleaning_entries)} data cleaning operations were performed: {actions_list}. "
            )
            if total_rows_removed > 0:
                sections.append(
                    f"This resulted in {total_rows_removed:,} observations being excluded, "
                    f"yielding a final sample of {n_total:,} observations for modeling."
                )

    # Predictors
    sections.append("\n\n### Predictor Variables\n")
    if _selected_count == 0:
        # A recorded zero is a RESULT. Falling through to the branches below
        # would list the predictors the workflow started with as though they
        # were the ones selection kept.
        _screened = feature_counts.get('candidate') or feature_counts.get('original')
        sections.append(
            (f"Feature selection retained none of the {_screened} candidate "
             f"predictors it screened, so no selected predictor set was produced."
             ) if _screened else
            "Feature selection retained no predictors, so no selected "
            "predictor set was produced."
        )
    elif feature_counts.get('original') and feature_counts.get('selected') and feature_counts['original'] != feature_counts['selected']:
        original_count = feature_counts['original']
        candidate_count = feature_counts.get('candidate')
        engineered_count = feature_counts.get('engineered')
        selected_count = feature_counts['selected']
        if candidate_count and candidate_count != original_count and feature_engineering_ran(feature_counts):
            sections.append(
                f"The workflow began with {original_count} original predictors. "
                f"Feature engineering added {engineered_count if engineered_count is not None else max(candidate_count - original_count, 0)} predictors, yielding {candidate_count} candidate predictors. "
                f"{selected_count} predictors were retained for final modeling."
            )
        else:
            sections.append(
                f"The workflow began with {original_count} original predictors and retained {selected_count} predictors for final modeling."
            )
    elif predictor_count <= 15 and feature_names:
        feat_list = ", ".join(feature_names)
        sections.append(f"The following predictor variables were included: {feat_list}.")
    else:
        sections.append(
            f"A total of {predictor_count} predictor variables were included "
            f"(see Supplementary Table S1 for full list)."
        )

    # Feature selection: use logged data as source of truth
    
    # Build the feature selection narrative from the methodology log.
    # Priority: use the LAST "Feature Selection Applied" entry (consensus or manual
    # override), supplemented by the analysis run details from "Feature Selection".
    feature_selection_logged = False
    
    # 1. Check what was actually applied (may be consensus or manual override)
    applied_entries = logged_steps.get('Feature Selection Applied', [])
    analysis_entries = logged_steps.get('Feature Selection', [])
    
    if applied_entries:
        # Use the LAST applied entry (manual override supersedes consensus)
        applied = applied_entries[-1]
        details = applied.get('details', {})
        method = details.get('method', '')
        n_selected = details.get('n_features_selected')
        
        # Get the methods used from the analysis step (lasso, rfe, etc.)
        analysis_methods = []
        analysis_methods_completed = None
        n_original = None
        for ae in analysis_entries:
            ad = ae.get('details', {})
            analysis_methods = ad.get('methods', analysis_methods)
            if ad.get('methods_completed') is not None:
                analysis_methods_completed = ad['methods_completed']
            if ad.get('n_features_before'):
                n_original = ad['n_features_before']
        # `MISC-104`: the methods that RAN are the ones the sentence names. A
        # method that raised is not part of a consensus it never voted in.
        methods_named = (analysis_methods_completed
                         if analysis_methods_completed is not None
                         else analysis_methods)
        methods_str = _oxford_join(_feature_selection_method_label(method_name) for method_name in methods_named)

        if method == 'manual' and n_selected is not None:
            # Manual override — report it as such
            if methods_str and n_original:
                sections.append(
                    f" Feature selection was performed using {methods_str}. "
                    f"After review, {n_selected} features were manually selected for modeling."
                )
            else:
                sections.append(
                    f" {n_selected} features were manually selected for modeling."
                )
            feature_selection_logged = True
        elif method == 'consensus' and n_selected is not None:
            # Consensus selection
            n_before = feature_counts.get('candidate') or n_original or details.get('n_features_before')

            # Check for consensus threshold in details
            consensus_threshold = details.get('consensus_threshold') or details.get('threshold')
            threshold_clause = _consensus_threshold_clause(
                consensus_threshold,
                len(analysis_methods_completed) if analysis_methods_completed is not None else None,
                len(analysis_methods) if analysis_methods else None,
            )

            # `MISC-104`. TWO UNIVERSES, AND THE SENTENCE USED TO MIX THEM.
            # `n_before` counts the NUMERIC candidates that were ranked
            # (pages/04:385); `n_selected` counts what was APPLIED — the ranked
            # survivors PLUS the non-ranked features carried through unchanged
            # (pages/04:571). "reduced the feature set from 10 to 5" was
            # therefore a ratio between two different populations. Both are
            # stated when the log records the split.
            n_ranked = details.get('n_consensus_ranked')
            carried = list(details.get('carried_through_unranked') or [])

            if n_ranked is not None and carried:
                lead = (f" Consensus feature selection across {methods_str}"
                        if methods_str else " Consensus feature selection")
                if n_before:
                    ranking_clause = (f"{lead} ranked {n_before} numeric candidate "
                                      f"predictors and retained {n_ranked} of them")
                else:
                    ranking_clause = f"{lead} retained {n_ranked} ranked predictors"
                sections.append(
                    f"{ranking_clause}; {len(carried)} non-ranked feature(s) "
                    f"({_oxford_join(carried)}) were carried through unranked, "
                    f"giving {n_selected} predictors in the final modeling set."
                    f"{threshold_clause}"
                )
            elif n_before and n_before == n_selected:
                if methods_str:
                    sections.append(
                        f" Consensus feature selection across {methods_str} retained all {n_selected} candidate predictors."
                        f"{threshold_clause}"
                    )
                else:
                    sections.append(
                        f" Feature selection was performed; all {n_selected} features were retained."
                    )
            elif n_before:
                if methods_str:
                    sections.append(
                        f" Consensus feature selection across {methods_str} reduced the feature set "
                        f"from {n_before} to {n_selected} predictors."
                        f"{threshold_clause}"
                    )
                else:
                    sections.append(
                        f" Feature selection reduced the feature set from {n_before} to {n_selected} predictors."
                        f"{threshold_clause}"
                    )
            else:
                if methods_str:
                    sections.append(f" Consensus feature selection across {methods_str} retained {n_selected} features.{threshold_clause}")
                else:
                    sections.append(f" Feature selection retained {n_selected} features.{threshold_clause}")
            feature_selection_logged = True
    
    # 2. Fall back to analysis entries if nothing was explicitly applied
    if not feature_selection_logged and analysis_entries:
        for entry in analysis_entries:
            details = entry.get('details', {})
            n_before = feature_counts.get('candidate') or details.get('n_features_before')
            n_after = details.get('n_features_after')
            methods = details.get('methods', [])
            methods_str = _oxford_join(_feature_selection_method_label(method_name) for method_name in methods)
            if n_before and n_after:
                if n_before == n_after:
                    if methods_str:
                        sections.append(
                            f" Consensus feature selection across {methods_str} retained all {n_after} candidate predictors."
                        )
                    else:
                        sections.append(f" Feature selection was performed; all {n_after} features were retained.")
                else:
                    if methods_str:
                        sections.append(
                            f" Consensus feature selection across {methods_str} reduced the feature set "
                            f"from {n_before} to {n_after} predictors."
                        )
                    else:
                        sections.append(f" Feature selection reduced the feature set from {n_before} to {n_after} predictors.")
                feature_selection_logged = True
                break
    
    # Only fall back to parameter if no log entries exist
    if not feature_selection_logged and feature_selection_method:
        sections.append(f" Feature selection was performed using {feature_selection_method}.")

    # Feature Engineering (if applied)
    try:
        import streamlit as st
        feature_engineering_applied = st.session_state.get('feature_engineering_applied', False)
        if feature_engineering_applied:
            sections.append("\n\n### Feature Engineering\n")
            engineering_log = st.session_state.get('engineering_log', [])
            engineered_feature_names = st.session_state.get('engineered_feature_names', [])
            
            sections.append("Feature engineering was performed prior to feature selection. ")
            
            # List techniques from engineering log (FIX 2: Enhanced specificity)
            if engineering_log:
                sections.append("The following transformations were applied: ")
                techniques = []
                # Get per-model PCA configs for more precise reporting
                configs_by_model = st.session_state.get('preprocessing_config_by_model', {})
                pca_details = {}
                for model_key, cfg in configs_by_model.items():
                    if cfg.get('use_pca'):
                        pca_mode = cfg.get('pca_mode')
                        pca_n = cfg.get('pca_n_components')
                        pca_details[model_key] = {'mode': pca_mode, 'n_components': pca_n}
                
                for log_entry in engineering_log:
                    # Parse entries like "Polynomial degree 2: +45 features"
                    if ':' in log_entry:
                        technique, detail = log_entry.split(':', 1)
                        technique_name = technique.strip()
                        
                        # FIX 2: Specific parsing by transform type
                        if 'PCA' in technique_name.upper() and pca_details:
                            techniques.append(pca_technique_sentence(pca_details))
                        elif 'Polynomial' in technique_name:
                            # Extract degree and mode from "Polynomial degree 2 (full)" or "Polynomial degree 2 (interaction-only)"
                            import re
                            degree_match = re.search(r'degree (\d+)', technique_name)
                            mode_match = re.search(r'\((.*?)\)', technique_name)
                            if degree_match:
                                degree = degree_match.group(1)
                                mode = mode_match.group(1) if mode_match else 'full'
                                techniques.append(f"Polynomial features (degree {degree}, {mode})")
                            else:
                                techniques.append(f"Polynomial features")
                        elif 'Binning' in technique_name:
                            # Extract strategy and bins from "Binning (equal_width, 5 bins)"
                            import re
                            strat_match = re.search(r'Binning \(([^,]+),\s*(\d+)\s+bins?\)', technique_name)
                            if strat_match:
                                strategy = strat_match.group(1)
                                n_bins = strat_match.group(2)
                                techniques.append(f"Binning ({strategy}, {n_bins} bins)")
                            else:
                                techniques.append(f"Binning")
                        elif 'Ratio' in technique_name:
                            techniques.append(f"Ratio features")
                        elif 'Mathematical' in technique_name or 'transform' in technique_name.lower():
                            # Generic mathematical transforms (log, sqrt, etc.)
                            techniques.append(f"Mathematical transformations (log, sqrt, square, inverse)")
                        elif 'TDA' in technique_name.upper() or 'Topological' in technique_name:
                            techniques.append(f"Topological data analysis features")
                        elif 'UMAP' in technique_name.upper():
                            techniques.append(f"UMAP dimensionality reduction")
                        else:
                            # Fallback to generic parsing
                            techniques.append(f"{technique_name}")
                    else:
                        techniques.append(log_entry)
                sections.append("; ".join(techniques) + ". ")
            
            # Total features created
            n_engineered = feature_counts.get('engineered')
            if n_engineered is None:
                n_engineered = len(engineered_feature_names) if engineered_feature_names else 0
            if n_engineered > 0:
                sections.append(
                    f"In total, {n_engineered} engineered features were created and included in subsequent feature selection."
                )
            else:
                sections.append(
                    "The engineered features were included in subsequent feature selection and model training."
                )
    except ImportError:
        # Not in a Streamlit context, skip feature engineering section
        pass

    # Missing data & Preprocessing (combined when preprocessing_summary available)
    _preproc = preprocessing_config or {}
    _has_summary = bool(_preproc.get("missing_data"))

    sections.append("\n\n### Missing Data\n")
    if _has_summary:
        _md = _preproc["missing_data"]
        sections.append(f"Missing numeric values were handled using {_md['label']}.")
        if _md.get("indicators"):
            sections.append(
                " Binary indicator variables were added for features with missing values "
                "to allow models to leverage missingness patterns."
            )
    elif missing_data_strategy:
        sections.append(f"Missing data were handled using {missing_data_strategy}.")
    else:
        sections.append("[PLACEHOLDER: Describe how missing data were handled.]")
    
    # Add missing data summary if available
    if missing_data_summary:
        n_features_with_missing = missing_data_summary.get('n_features_with_missing')
        total_features = missing_data_summary.get('total_features')
        min_missing_rate = missing_data_summary.get('min_missing_rate')
        max_missing_rate = missing_data_summary.get('max_missing_rate')

        # Which participants this counted. The feature count comes from the
        # dataset profile, which is computed on training rows only once a test
        # set is sealed, while the per-feature rates are computed over the whole
        # frame. Two populations in one sentence: state them, or the reader
        # takes both as the whole study.
        _scope = missing_data_summary.get('row_scope')
        _scope_n = missing_data_summary.get('row_scope_n')
        _counted_in = ""
        if _scope == 'training' and _scope_n:
            _counted_in = f", counted on the {_scope_n:,} training rows"
        _rates_over = ""
        if (missing_data_summary.get('rates_row_scope') == 'all'
                and missing_data_summary.get('rates_n_rows')):
            _rates_over = (f" over all {missing_data_summary['rates_n_rows']:,} "
                           f"records")

        if n_features_with_missing is not None and total_features is not None:
            if min_missing_rate is not None and max_missing_rate is not None:
                sections.append(
                    f" {n_features_with_missing} of {total_features} features had missing values"
                    f"{_counted_in} (missing rates ranging from {min_missing_rate*100:.1f}% "
                    f"to {max_missing_rate*100:.1f}%{_rates_over})."
                )
            else:
                sections.append(
                    f" {n_features_with_missing} of {total_features} features had missing values"
                    f"{_counted_in}."
                )

    sections.append("\n\n### Data Preprocessing\n")
    
    # Per-model preprocessing: check session state for configs_by_model
    _per_model_configs = {}
    try:
        import streamlit as st
        _per_model_configs = st.session_state.get('preprocessing_config_by_model', {})
    except ImportError:
        pass
    
    # Helper labels
    _scale_labels = {
        "standard": "z-score standardization", "robust": "robust scaling",
        "minmax": "min-max normalization", "none": None,
    }
    _enc_labels = {
        "onehot": "one-hot encoding", "target": "target encoding",
        "ordinal": "ordinal encoding",
    }
    
    def _describe_model_preproc(cfg: Dict) -> List[str]:
        """Build list of preprocessing description sentences for one model config."""
        sents = []
        scaling = cfg.get('numeric_scaling', 'none')
        sl = _scale_labels.get(scaling)
        if sl:
            sents.append(f"scaled using {sl}")
        encoding = cfg.get('categorical_encoding', '')
        el = _enc_labels.get(encoding)
        if el:
            sents.append(f"categorical variables encoded using {el}")
        outlier = cfg.get('numeric_outlier_treatment', 'none')
        outlier_desc = _describe_outlier_handling(outlier, cfg.get('numeric_outlier_params', {}))
        if outlier_desc:
            sents.append(outlier_desc)
        # PCA
        if cfg.get('use_pca'):
            pca_n = cfg.get('pca_n_components')
            if isinstance(pca_n, float) and pca_n < 1:
                sents.append(f"PCA applied (retaining {pca_n*100:.0f}% variance)")
            elif isinstance(pca_n, int):
                sents.append(f"PCA applied ({pca_n} components)")
            else:
                sents.append("PCA dimensionality reduction applied")
        transform = cfg.get('numeric_power_transform', 'none')
        if transform and transform != 'none':
            sents.append(f"{transform} power transform applied")
        log_t = cfg.get('numeric_log_transform', False)
        if log_t:
            sents.append("log transform applied")
        return sents
    
    if _per_model_configs and len(_per_model_configs) > 1:
        # Check if all models share the same preprocessing
        config_signatures = {}
        for mk, cfg in _per_model_configs.items():
            sig = (
                cfg.get('numeric_scaling', 'none'),
                cfg.get('categorical_encoding', ''),
                cfg.get('numeric_outlier_treatment', 'none'),
                json.dumps(cfg.get('numeric_outlier_params', {}), sort_keys=True, default=str),
                cfg.get('numeric_power_transform', 'none'),
                cfg.get('numeric_log_transform', False),
                cfg.get('use_pca', False),
                cfg.get('pca_n_components'),
            )
            config_signatures[mk] = sig
        
        unique_sigs = set(config_signatures.values())
        
        if len(unique_sigs) == 1:
            # All models share the same preprocessing
            first_cfg = next(iter(_per_model_configs.values()))
            sents = _describe_model_preproc(first_cfg)
            if sents:
                sections.append("All models shared identical preprocessing: " + "; ".join(sents) + ".")
            else:
                sections.append("No additional preprocessing transformations were applied beyond imputation.")
        else:
            # Models differ — describe each model's full preprocessing
            # Check for any truly shared settings
            all_cfgs = list(_per_model_configs.values())
            shared_scaling = all_cfgs[0].get('numeric_scaling', 'none') if len(set(c.get('numeric_scaling', 'none') for c in all_cfgs)) == 1 else None
            shared_encoding = all_cfgs[0].get('categorical_encoding', '') if len(set(c.get('categorical_encoding', '') for c in all_cfgs)) == 1 else None
            
            shared_parts = []
            if shared_scaling and shared_scaling != 'none' and _scale_labels.get(shared_scaling):
                shared_parts.append(f"continuous features were scaled using {_scale_labels[shared_scaling]}")
            if shared_encoding and _enc_labels.get(shared_encoding):
                shared_parts.append(f"categorical variables were encoded using {_enc_labels[shared_encoding]}")
            
            if shared_parts:
                sections.append("Across all models, " + "; ".join(shared_parts) + ".")
            
            # Describe per-model differences
            sections.append(" Model-specific preprocessing differed as follows:")
            for mk, cfg in _per_model_configs.items():
                diffs = []
                # Scaling — mention if not shared or if this model differs
                if not shared_scaling:
                    scaling = cfg.get('numeric_scaling', 'none')
                    sl = _scale_labels.get(scaling)
                    if sl:
                        diffs.append(f"scaled using {sl}")
                    else:
                        diffs.append("no feature scaling")
                # Outlier treatment
                outlier = cfg.get('numeric_outlier_treatment', 'none')
                outlier_desc = _describe_outlier_handling(outlier, cfg.get('numeric_outlier_params', {}))
                if outlier_desc:
                    diffs.append(outlier_desc)
                # PCA
                if cfg.get('use_pca'):
                    pca_n = cfg.get('pca_n_components')
                    if isinstance(pca_n, float) and pca_n < 1:
                        diffs.append(f"PCA applied (retaining {pca_n*100:.0f}% variance)")
                    elif isinstance(pca_n, int):
                        diffs.append(f"PCA applied ({pca_n} components)")
                    else:
                        diffs.append("PCA dimensionality reduction applied")
                # Power transform
                transform = cfg.get('numeric_power_transform', 'none')
                if transform and transform != 'none':
                    diffs.append(f"{transform} power transform")
                # Log transform
                log_t = cfg.get('numeric_log_transform', False)
                if log_t:
                    diffs.append("log transform")
                
                if diffs:
                    sections.append(f" {_publication_model_label(mk)}: {'; '.join(diffs)}.")
                else:
                    sections.append(f" {_publication_model_label(mk)}: default preprocessing (no additional transformations).")
    elif 'Preprocessing' in logged_steps:
        # Fallback to logged preprocessing (single-config path)
        preprocessing_logged = False
        for entry in logged_steps['Preprocessing']:
            details = entry.get('details', {})
            if details:
                sentences = []
                
                if details.get('scaling') and details['scaling'] != 'none':
                    scale_label = _scale_labels.get(details['scaling'], details['scaling'])
                    if scale_label:
                        sentences.append(f"Continuous features were scaled using {scale_label}.")
                
                if details.get('encoding'):
                    enc_label = _enc_labels.get(details['encoding'], details['encoding'])
                    if enc_label:
                        sentences.append(f"Categorical variables were encoded using {enc_label}.")
                
                if details.get('outlier_handling') and details['outlier_handling'] != 'none':
                    outlier_params = (
                        details.get('outlier_params')
                        or details.get('numeric_outlier_params')
                        or {}
                    )
                    outlier_desc = _describe_outlier_handling(details['outlier_handling'], outlier_params)
                    if outlier_desc:
                        sentences.append(f"Outliers were {outlier_desc}.")
                
                if sentences:
                    sections.append(" ".join(sentences))
                    preprocessing_logged = True
                    break
        
        if not preprocessing_logged:
            sections.append("No additional preprocessing transformations were applied beyond imputation.")
    elif _has_summary:
        sentences = []
        _sc = _preproc.get("scaling", {})
        if _sc.get("method", "none") != "none":
            n_num = _preproc.get("n_numeric", 0)
            num_note = f" ({n_num} features)" if n_num else ""
            sentences.append(f"Continuous predictors{num_note} were scaled using {_sc['label']}.")
        _tr = _preproc.get("transforms", {})
        if _tr.get("method", "none") != "none":
            sentences.append(f"A {_tr['label']} was applied to reduce skewness.")
        _ol = _preproc.get("outliers", {})
        if _ol.get("method", "none") != "none":
            outlier_params = _ol.get("params", {})
            if not outlier_params:
                outlier_params = {
                    key: _ol.get(key)
                    for key in (
                        "lower_percentile", "upper_percentile", "threshold",
                        "mad_threshold", "n_mad", "multiplier", "iqr_multiplier"
                    )
                    if _ol.get(key) is not None
                }
            outlier_desc = _describe_outlier_handling(_ol.get("method", "none"), outlier_params)
            if outlier_desc:
                sentences.append(f"Outliers were {outlier_desc}.")
            else:
                sentences.append(f"Outliers were addressed via {_ol['label']}.")
        _en = _preproc.get("encoding", {})
        if _en.get("method"):
            n_cat = _preproc.get("n_categorical", 0)
            cat_note = f" ({n_cat} variables)" if n_cat else ""
            sentences.append(f"Categorical predictors{cat_note} were transformed using {_en['label']}.")
        if sentences:
            sections.append(" ".join(sentences))
        else:
            sections.append("No additional preprocessing transformations were applied beyond imputation.")
    elif _preproc:
        steps = []
        scaling = _preproc.get("numeric_scaling", "standard")
        if scaling != "none":
            steps.append(f"numeric features were {scaling}-scaled")
        outlier_desc = _describe_outlier_handling(
            _preproc.get("numeric_outlier_treatment", "none"),
            _preproc.get("numeric_outlier_params", {}),
        )
        if outlier_desc:
            steps.append(outlier_desc)
        # Don't mention imputation here (already in Missing Data section)
        cat_enc = _preproc.get("categorical_encoding", "onehot")
        if cat_enc:
            steps.append(f"categorical variables were encoded using {cat_enc} encoding")
        if steps:
            sections.append(f"Preprocessing included: {'; '.join(steps)}.")
    else:
        sections.append("[PLACEHOLDER: Describe preprocessing steps.]")

    # FIX 6: Preprocessing order of operations
    # Determine which optional steps were actually used
    outlier_used = False
    power_transform_used = False
    
    if _per_model_configs:
        for cfg in _per_model_configs.values():
            if cfg.get('numeric_outlier_treatment', 'none') != 'none':
                outlier_used = True
            if cfg.get('numeric_power_transform', 'none') != 'none' or cfg.get('numeric_log_transform', False):
                power_transform_used = True
    elif _preproc:
        if _preproc.get('numeric_outlier_treatment', 'none') != 'none' or (_preproc.get('outliers', {}).get('method', 'none') != 'none'):
            outlier_used = True
        if _preproc.get('numeric_power_transform', 'none') != 'none' or _preproc.get('numeric_log_transform', False):
            power_transform_used = True
    
    # Build the order sentence
    order_steps = ["missing value imputation", "feature scaling", "categorical encoding"]
    if outlier_used:
        order_steps.append("outlier treatment")
    if power_transform_used:
        order_steps.append("power transformation")
    
    sections.append(
        f" For all models, preprocessing was applied in the following order: "
        f"{', '.join(order_steps[:-1])}, and {order_steps[-1]}."
    )

    # Model development
    sections.append("\n\n### Model Development\n")
    
    # Use logged Model Training data if available
    models_str = None
    if 'Model Training' in logged_steps:
        for entry in logged_steps['Model Training']:
            details = entry.get('details', {})
            models_trained = details.get('models', [])
            best_model = details.get('best_model')
            use_cv = details.get('use_cv', False)
            cv_folds_logged = details.get('cv_folds')
            hyperopt_logged = details.get('hyperparameter_optimization', False)
            
            if models_trained:
                models_str = ', '.join(_publication_model_label(m) for m in models_trained)
                sections.append(f"The workflow trained and compared the following model candidates: {models_str}.")
            
            # Don't use logged best_model here - we'll determine it from actual results below
            
            break  # Use first logged training entry
    else:
        model_names = list(model_configs.keys()) if model_configs else []
        if model_names:
            models_str = ', '.join(_publication_model_label(n) for n in model_names)
            sections.append(
                f"The following models were developed and compared: {models_str}."
            )

    # Class weighting narration
    _class_weight_used = False
    if 'Model Training' in logged_steps:
        for entry in logged_steps['Model Training']:
            if entry.get('details', {}).get('class_weight_balanced'):
                _class_weight_used = True
                break

    if _class_weight_used:
        # `GUIDED-049`, and this is the SECOND methods generator — the one its
        # fix did not reach. `ml/narrative_engine.py` was corrected; this file
        # kept shipping the removed sentence verbatim, and
        # `pages/10_Report_Export.py` falls through to here whenever the
        # provenance singleton is empty. So the app went on endorsing
        # rebalancing in the artifact that *is* the product, by a second route.
        #
        # One sentence, one owner. `imbalance_advice.manuscript_sentence`
        # still says what was done — the reader has to know — and says it
        # without approval, with the limitation and the citation attached.
        # `model_purpose` is the key `narrative_engine` reads for the same
        # call, so the two generators resolve the purpose identically. Absent,
        # `manuscript_sentence` returns the unqualified-purpose form, which
        # still carries the limitation — the sentence is never approving.
        from ml import imbalance_advice as _imbalance
        sections.append(" " + _imbalance.manuscript_sentence(
            (manuscript_context or {}).get("model_purpose")))

    # Add hyperparameter details if available
    if model_hyperparameters:
        hp_details = []
        for model_name, params in model_hyperparameters.items():
            if not params:
                continue
            model_key = model_name.lower()
            # Format publication-relevant params by model type
            if model_key in ('ridge', 'lasso', 'elasticnet'):
                relevant = []
                if 'alpha' in params:
                    relevant.append(f"α={_fmt_param_value(params['alpha'])}")
                if 'l1_ratio' in params and model_key == 'elasticnet':
                    relevant.append(f"l1_ratio={_fmt_param_value(params['l1_ratio'])}")
                if relevant:
                    hp_details.append(f"{_publication_model_label(model_name)} ({', '.join(relevant)})")
            elif model_key in ('histgb_reg', 'histgb_clf', 'rf', 'xgb', 'lgbm'):
                relevant = []
                if 'n_estimators' in params:
                    relevant.append(f"n_estimators={_fmt_param_value(params['n_estimators'])}")
                if 'max_depth' in params:
                    relevant.append(f"max_depth={_fmt_param_value(params['max_depth'])}")
                if 'learning_rate' in params:
                    relevant.append(f"learning_rate={_fmt_param_value(params['learning_rate'])}")
                if relevant:
                    hp_details.append(f"{_publication_model_label(model_name)} ({', '.join(relevant)})")
            elif model_key == 'nn':
                relevant = []
                if 'hidden_layer_sizes' in params:
                    hls = params['hidden_layer_sizes']
                    if isinstance(hls, (list, tuple)):
                        relevant.append(f"hidden layers: {list(hls)}")
                    else:
                        relevant.append(f"hidden layers: {hls}")
                if 'learning_rate_init' in params:
                    relevant.append(f"learning rate={_fmt_param_value(params['learning_rate_init'])}")
                if 'max_iter' in params:
                    relevant.append(f"max epochs={_fmt_param_value(params['max_iter'])}")
                if relevant:
                    hp_details.append(f"{_publication_model_label(model_name)} ({', '.join(relevant)})")
            elif model_key == 'svm':
                relevant = []
                if 'C' in params:
                    relevant.append(f"C={_fmt_param_value(params['C'])}")
                if 'kernel' in params:
                    relevant.append(f"kernel={params['kernel']}")
                if 'gamma' in params:
                    relevant.append(f"gamma={_fmt_param_value(params['gamma'])}")
                if relevant:
                    hp_details.append(f"{_publication_model_label(model_name)} ({', '.join(relevant)})")
            elif model_key == 'knn':
                relevant = []
                if 'n_neighbors' in params:
                    relevant.append(f"n_neighbors={_fmt_param_value(params['n_neighbors'])}")
                if 'weights' in params:
                    relevant.append(f"weights={params['weights']}")
                if relevant:
                    hp_details.append(f"{_publication_model_label(model_name)} ({', '.join(relevant)})")
        if hp_details:
            sections.append(f" Key hyperparameters: {'; '.join(hp_details)}.")
    
    # Add hyperparameter optimization note
    if hyperparameter_optimization:
        sections.append(" Hyperparameter optimization was performed using Optuna (30 trials per model).")
    
    # FIX 5: Baseline model reporting
    if selected_model_results:
        baseline_models = [name for name in selected_model_results.keys() if 'baseline' in name.lower()]
        if baseline_models:
            if task_type == "regression":
                sections.append(" Baseline models (mean predictor and simple linear regression) were automatically generated for comparison.")
            else:
                sections.append(" Baseline models (majority class predictor and simple logistic regression) were automatically generated for comparison.")
    
    # Construct split description with strategy
    # Auto-detect stratification from split_config if split_strategy not explicitly provided
    if not split_strategy:
        _stratify = split_config.get('stratify', False) if isinstance(split_config, dict) else getattr(split_config, 'stratify', False)
        _use_time = split_config.get('use_time_split', False) if isinstance(split_config, dict) else getattr(split_config, 'use_time_split', False)
        _use_group = split_config.get('use_group_split', False) if isinstance(split_config, dict) else getattr(split_config, 'use_group_split', False)
        if _use_time:
            split_strategy = "temporal"
        elif _use_group:
            split_strategy = "group-based"
        elif _stratify:
            split_strategy = "stratified random"

    split_desc = "Data were split"
    if split_strategy:
        split_desc += f" using {split_strategy} sampling"
    split_desc += (
        f" into training ({n_train:,}, {n_train/n_total*100:.0f}%), "
        f"validation ({n_val:,}, {n_val/n_total*100:.0f}%), and "
        f"test ({n_test:,}, {n_test/n_total*100:.0f}%) sets."
    )
    sections.append(split_desc)

    # Target trimming
    _trim_enabled = split_config.get('target_trim_enabled', False) if isinstance(split_config, dict) else getattr(split_config, 'target_trim_enabled', False)
    if _trim_enabled:
        _trim_lo = split_config.get('target_trim_lower', 0.0) if isinstance(split_config, dict) else getattr(split_config, 'target_trim_lower', 0.0)
        _trim_hi = split_config.get('target_trim_upper', 1.0) if isinstance(split_config, dict) else getattr(split_config, 'target_trim_upper', 1.0)
        sections.append(
            f" Prior to splitting, target variable values below the {_trim_lo*100:.0f}th percentile "
            f"and above the {_trim_hi*100:.0f}th percentile were excluded to reduce the influence "
            f"of extreme outliers on model training."
        )

    # Target transformation
    _target_transform = split_config.get('target_transform', 'none') if isinstance(split_config, dict) else getattr(split_config, 'target_transform', 'none')
    if _target_transform and _target_transform != 'none':
        _transform_names = {'log1p': 'log(1+x)', 'yeo-johnson': 'Yeo-Johnson', 'box-cox': 'Box-Cox'}
        _tname = _transform_names.get(_target_transform, _target_transform)
        sections.append(
            f" The target variable was transformed using the {_tname} power transformation "
            f"prior to model training. All reported performance metrics reflect predictions "
            f"back-transformed to the original scale."
        )
    
    # Check for CV from logged data or parameters
    cv_to_use = None
    if 'Model Training' in logged_steps:
        for entry in logged_steps['Model Training']:
            details = entry.get('details', {})
            if details.get('use_cv'):
                cv_to_use = details.get('cv_folds', 5)
                break
    if cv_to_use is None and cv_folds:
        cv_to_use = cv_folds
    
    # `AUDIT-026`. This said "{n}-fold cross-validation was used for internal
    # validation." on the strength of the checkbox alone. The sentence is not
    # deleted — it is made to say what the run did, which is the honest
    # replacement and the only one that keeps the Methods section complete.
    if cv_to_use:
        if not cv_scope_known:
            sections.append(
                f" {cv_to_use}-fold cross-validation was enabled in the training "
                f"configuration. This draft was assembled without per-model "
                f"results, so it does not report whether cross-validation "
                f"produced scores for any model; the internal validation this "
                f"draft can attest to is the train/validation/test split "
                f"described above."
            )
        elif not cv_models_run:
            # Requested and ran for nothing — pages/06:1455 (the neural network
            # is excluded from CV) or :1489 (CV raised and was swallowed).
            sections.append(
                f" {cv_to_use}-fold cross-validation was requested but produced "
                f"results for no model, so no cross-validated estimate is "
                f"reported: internal validation rests on the single "
                f"train/validation/test split described above."
            )
        else:
            _cv_names = _oxford_join([_publication_model_label(m) for m in cv_models_run])
            sections.append(
                f" {cv_to_use}-fold cross-validation was used for internal "
                f"validation of {_cv_names}."
            )
            _not_cv = [m for m in (selected_model_results or {}) if m not in set(cv_models_run)]
            if _not_cv:
                sections.append(
                    f" It was not run for "
                    f"{_oxford_join([_publication_model_label(m) for m in _not_cv])}, "
                    f"whose reported performance comes from the held-out split alone."
                )

    # Performance evaluation
    sections.append("\n\n### Performance Evaluation\n")
    sections.append(
        f"Model performance was evaluated on the workflow's held-out data using {', '.join(metrics_used)}. "
        f"When available, 95% confidence intervals were computed from 1,000 BCa bootstrap resamples. "
    )
    # The parameter was never passed by anything: page 10 assembles this call
    # from session state, and page 07 computed external metrics, displayed them
    # and dropped them, so the flag stayed False no matter what the researcher
    # had actually done. The record is now written when the validation runs, and
    # it is what this section reads when the caller does not say.
    ext_record = external_validation_record() if not external_validation else None
    if ext_record:
        external_validation = True
    if external_validation:
        sections.append(external_validation_sentence(ext_record))

    # Explainability - check both parameter and session state
    explainability_to_mention = set(explainability_methods or [])
    
    # Check session state for SHAP and Bland-Altman
    try:
        import streamlit as st
        if st.session_state.get('shap_results'):
            explainability_to_mention.add('shap')
        if st.session_state.get('bland_altman_results'):
            explainability_to_mention.add('bland_altman')
    except ImportError:
        pass
    
    if explainability_to_mention:
        sections.append("\n\n### Model Interpretability\n")
        
        # FIX 3: Extract model list and sample size from explainability log
        explainability_models = []
        explainability_entries = logged_steps.get('Explainability', [])
        if explainability_entries:
            for entry in explainability_entries:
                details = entry.get('details', {})
                models = details.get('models', [])
                if models:
                    explainability_models.extend(models)
        # Deduplicate models
        explainability_models = list(set(explainability_models)) if explainability_models else []
        
        method_descriptions = {
            "permutation_importance": "Permutation importance was computed to assess feature contributions by measuring the decrease in model performance when each feature was randomly shuffled.",
            "shap": "SHapley Additive exPlanations (SHAP) values were computed to quantify the contribution of each feature to individual predictions.",
            "partial_dependence": "Partial dependence plots were generated to visualize the marginal effect of individual features on the predicted outcome.",
            "calibration": (
                "Model calibration was assessed using reliability diagrams, "
                "Brier score, and expected calibration error (ECE)."
                if str(task_type).lower().startswith("classif") else
                "Model calibration was assessed by regressing the observed "
                "outcome on the predicted value; the calibration slope (1.0 "
                "indicates perfect calibration), the calibration intercept "
                "(0.0 indicates no systematic bias) and R² are reported. "
                "Reliability diagrams, the Brier score and the expected "
                "calibration error are defined for predicted probabilities "
                "and were not computed for this continuous outcome."
            ),
            "subgroup": "Subgroup analysis was performed to evaluate model performance across clinically relevant subgroups.",
            "decision_curve": "Decision curve analysis was performed to assess the clinical utility of the model at various probability thresholds.",
            "bland_altman": "Bland-Altman analysis was performed to assess agreement between model predictions.",
        }
        
        for method in sorted(explainability_to_mention):
            desc = method_descriptions.get(method, f"{method} analysis was performed.")
            # FIX 3: Add model scope and sample size
            if method in ['permutation_importance', 'shap'] and explainability_models:
                model_list = ', '.join(explainability_models)
                desc = desc.rstrip('.') + f" for {model_list}"
                if n_test > 0:
                    desc += f" using {n_test:,} test observations."
                else:
                    desc += "."
            sections.append(f"{desc} ")

    # Sensitivity Analysis — check methodology log and session state
    _has_seed_sensitivity = False
    _has_feature_dropout = False
    try:
        import streamlit as st
        _has_seed_sensitivity = st.session_state.get('sensitivity_seed_results') is not None
        _has_feature_dropout = st.session_state.get('sensitivity_dropout_results') is not None
    except ImportError:
        pass
    
    # Also check methodology log
    if 'Sensitivity Analysis' in logged_steps:
        for entry in logged_steps['Sensitivity Analysis']:
            action = entry.get('action', '')
            if 'seed' in action.lower():
                _has_seed_sensitivity = True
            if 'dropout' in action.lower():
                _has_feature_dropout = True
    
    if _has_seed_sensitivity or _has_feature_dropout:
        sections.append("\n\n### Sensitivity Analysis\n")
        
        # Collect ALL seed sensitivity entries (may have multiple models)
        seed_entries = []
        dropout_entries = []
        if 'Sensitivity Analysis' in logged_steps:
            for entry in logged_steps['Sensitivity Analysis']:
                action = entry.get('action', '').lower()
                if 'seed' in action:
                    seed_entries.append(entry.get('details', {}))
                if 'dropout' in action:
                    dropout_entries.append(entry.get('details', {}))
        
        if seed_entries:
            seed_entries = _dedupe_latest_by(seed_entries, ('model', 'metric'))
            if len(seed_entries) == 1:
                d = seed_entries[0]
                sections.append(
                    f"Seed stability analysis was performed on {_publication_model_label(d.get('model', '?'))} "
                    f"using {d.get('n_seeds', 'multiple')} random seeds to assess sensitivity of "
                    f"{d.get('metric', 'the primary metric')} to random initialization. "
                )
            else:
                models_str = ", ".join(
                    f"{_publication_model_label(d.get('model', '?'))} ({d.get('metric', '?')}, {d.get('n_seeds', '?')} seeds)"
                    for d in seed_entries
                )
                sections.append(
                    f"Seed stability analysis was performed on {models_str} "
                    f"to assess sensitivity to random initialization. "
                )
            
            # Include actual results from session state for the most recent run
            try:
                seed_df = st.session_state.get('sensitivity_seed_results')
                if seed_df is not None:
                    # Report results for the last seed entry's metric
                    last_metric = seed_entries[-1].get('metric', '')
                    last_model = _publication_model_label(seed_entries[-1].get('model', '?'))
                    if last_metric and last_metric in seed_df.columns:
                        valid = seed_df[last_metric].dropna()
                        if len(valid) > 1:
                            cv = valid.std() / abs(valid.mean()) * 100 if valid.mean() != 0 else 0
                            sections.append(
                                f"For {last_model}, the coefficient of variation across seeds was {cv:.1f}%, "
                                f"with {last_metric} ranging from {valid.min():.4f} to {valid.max():.4f} "
                                f"(mean: {valid.mean():.4f}, SD: {valid.std():.4f}). "
                            )
            except Exception:
                pass
        
        if dropout_entries:
            dropout_entries = _dedupe_latest_by(dropout_entries, ('model', 'metric'))
            if len(dropout_entries) == 1:
                d = dropout_entries[0]
                sections.append(
                    f"Feature dropout analysis was performed on {_publication_model_label(d.get('model', '?'))}, "
                    f"sequentially removing each of {d.get('n_features_tested', '')} features and retraining "
                    f"to measure the impact on {d.get('metric', 'the primary metric')}. "
                )
            else:
                parts = []
                for d in dropout_entries:
                    parts.append(
                        f"{_publication_model_label(d.get('model', '?'))} "
                        f"({d.get('n_features_tested', '?')} features, {d.get('metric', '?')})"
                    )
                sections.append(
                    f"Feature dropout analysis was performed on {' and '.join(parts)}, "
                    f"sequentially removing individual features and retraining to measure impact. "
                )

    # Software - get actual versions
    try:
        import sklearn
        import numpy
        import pandas
        import scipy
        
        py_version = sys.version.split()[0]
        sklearn_version = sklearn.__version__
        numpy_version = numpy.__version__
        pandas_version = pandas.__version__
        scipy_version = scipy.__version__
        
        sections.append("\n\n### Software\n")
        sections.append(
            f"All analyses were performed using Python (version {py_version}) with "
            f"scikit-learn ({sklearn_version}), NumPy ({numpy_version}), "
            f"pandas ({pandas_version}), and SciPy ({scipy_version}). "
            f"Random seed was set to {random_seed} for reproducibility."
        )
    except Exception:
        # Fallback if imports fail
        sections.append("\n\n### Software\n")
        sections.append(
            f"All analyses were performed using Python with "
            f"scikit-learn, NumPy, pandas, and SciPy. "
            f"Random seed was set to {random_seed} for reproducibility."
        )

    # Methodological Considerations
    sections.append("\n\n### Methodological Considerations\n")
    sections.append(
        "The following methodological choices and their implications are documented "
        "for transparency and reproducibility.\n\n"
    )

    # 0. The exploratory watermark, FIRST. `MISC-103`.
    #
    # This limitation reached the manuscript only through NarrativeEngine. When
    # the engine raises — or when provenance is empty — pages/10 falls back to
    # this composer, which said nothing about it, so the one caveat that is
    # about the whole study rather than one column vanished on exactly the path
    # a failure takes. It goes first here for the same reason it goes first
    # there: a study-wide caveat outranks a per-column one.
    if _exploratory_watermark(manuscript_context):
        sections.append(f"**Exploratory mode:** {EXPLORATORY_LIMITATION_SENTENCE}\n\n")

    # 1. What the cross-validation loop enclosed. `AUDIT-029`.
    #
    # This paragraph told the reader CV was "performed on data that had already
    # been preprocessed using the full training set", and hung an optimistic-bias
    # rider on PCA. Both have been false since STATE-059:
    # pages/06_Train_and_Compare.py:1481-1487 passes make_cv_pipeline the RAW
    # training partition, and ml/eval.py:182-197 clones the preprocessing into
    # the composite, so imputer, scaler, encoder AND PCA are re-fit inside every
    # fold. The manuscript is the one artifact where the description IS the
    # deliverable; it was describing a leak the code structurally prevents, and
    # in doing so it left the real out-of-loop steps undisclosed.
    #
    # §A5.5 asks for the ENTIRE pipeline inside the loop. Preprocessing is;
    # feature selection (page 04, before training) and hyperparameter tuning
    # (pages/06:1260/:1318, on a separate validation split) are not, and no
    # optimism correction is applied. That is the caveat this paragraph owes.
    #
    # Gated on cv_models_run (AUDIT-026): a paragraph describing a fold
    # structure must not print for a run that cross-validated nothing.
    if (cv_folds or cv_to_use) and cv_models_run:
        _has_feature_selection = feature_selection_logged or feature_selection_method is not None

        cv_num = cv_to_use if cv_to_use else cv_folds
        sections.append(
            f"**What the cross-validation loop enclosed:** {cv_num}-fold cross-validation was "
            f"performed on the raw training partition, with the preprocessing pipeline "
            f"(imputation, scaling, encoding, and any dimensionality reduction) re-fit inside "
            f"each fold, so no fold's held-out rows contributed to the statistics used to "
            f"transform it. The loop did not enclose the entire model-building procedure: "
            f"feature selection and hyperparameter tuning, where used, were performed before it "
            f"rather than inside it, and no optimism correction (bootstrap or nested resampling) "
            f"was applied, so the cross-validated estimates are not corrected for the optimism "
            f"those steps introduce"
        )
        if _has_feature_selection:
            sections.append(
                " — feature selection was applied in this workflow, on the training "
                "partition, before cross-validation began"
            )
        sections.append(
            ". Held-out test set performance is computed with preprocessing fit on the training "
            "partition only and is unaffected by the fold structure described here.\n\n"
        )

    # 2. Feature dropout methodology
    try:
        import streamlit as st
        _has_dropout = st.session_state.get('sensitivity_dropout_results') is not None
    except ImportError:
        _has_dropout = False

    if _has_dropout:
        sections.append(
            "**Feature dropout analysis:** When assessing the impact of individual feature removal, "
            "models were retrained using median imputation only (without the full preprocessing pipeline) "
            "due to the complexity of dynamically reconstructing column-specific pipelines for each "
            "feature permutation. This simplification is inconsequential for tree-based models, which are "
            "invariant to monotonic feature transformations. For linear and neural network models, "
            "dropout impact estimates should be interpreted with caution, as the absence of scaling "
            "may confound the effect of feature removal. Permutation importance and SHAP values, "
            "which operate on the fully preprocessed data, provide more reliable feature importance "
            "estimates for these model types.\n\n"
        )

    # 3. Feature engineering transform detection
    try:
        import streamlit as st
        _has_eng = st.session_state.get('feature_engineering_applied', False)
        _transform_map = st.session_state.get('engineered_feature_transforms', {})
    except ImportError:
        _has_eng = False
        _transform_map = {}

    if _has_eng and _transform_map:
        sections.append(
            "**Engineered feature handling in preprocessing:** To prevent double-transformation "
            "(e.g., applying a log transform to an already log-transformed feature), engineered features "
            "were identified by naming convention and automatically excluded from redundant preprocessing "
            "transforms. These features received imputation and scaling only. "
            "This detection relies on standard naming prefixes (e.g., `log_`, `sqrt_`, `PCA_`); "
            "features with non-standard names were treated as untransformed and received full preprocessing.\n\n"
        )

    # 4. Methodology audit trail
    if logged_steps:
        step_names = sorted(logged_steps.keys())
        sections.append(
            f"**Reproducibility:** All data processing decisions were recorded in an automated "
            f"methodology log covering {len(step_names)} analysis phases "
            f"({', '.join(step_names)}). "
            f"This log captures the specific parameters used at each step and can be "
            f"exported for full reproducibility.\n\n"
        )

    # ── Results Section (if actual results provided) ──
    if selected_model_results:
        sections.append("\n\n---\n\n## Results (Workflow-Derived Draft)\n")
        sections.append(f"\n### Model Performance\n")
        sections.append("This draft reports computed model outputs from the current workflow state and leaves interpretation to the author. ")

        # Determine best model from actual metrics while preserving manuscript-primary policy
        actual_best = manuscript_facts.get('best_model_by_metric') or _determine_best_model(selected_model_results, task_type)
        manuscript_primary_model = manuscript_facts.get('manuscript_primary_model')

        if manuscript_primary_model:
            sections.append(f"The manuscript-primary model was **{_publication_model_label(manuscript_primary_model)}**. ")
            if actual_best and actual_best != manuscript_primary_model:
                metric_name = manuscript_facts.get('best_metric_name') or 'held-out metric'
                sections.append(f"The best model by {metric_name} was **{_publication_model_label(actual_best)}**. ")
        elif actual_best:
            metric_name = manuscript_facts.get('best_metric_name') or 'held-out metric'
            sections.append(
                f"The best model by {metric_name} was **{_publication_model_label(actual_best)}**. "
                "No manuscript-primary model was explicitly selected in the workflow. "
            )

        sections.append(
            f"The bullet list below summarizes held-out performance for the {len(selected_model_results)} included model(s) "
            "and can be used when drafting the Results section.\n\n"
        )

        # Build a text table
        for name, res in selected_model_results.items():
            metrics = res.get("metrics", {})
            cis = bootstrap_results.get(name, {}) if bootstrap_results else {}
            metric_strs = []
            for m, v in _ordered_metric_items(metrics, task_type):
                ci = cis.get(m)
                if ci and hasattr(ci, 'ci_lower'):
                    metric_strs.append(f"{m}: {v:.4f} (95% CI: {ci.ci_lower:.4f}–{ci.ci_upper:.4f})")
                else:
                    metric_strs.append(f"{m}: {v:.4f}")
            sections.append(f"**{_publication_model_label(name)}:** {'; '.join(metric_strs)}\n\n")

        # Calibration. `MISC-102`: this block was unreachable — no caller passed
        # `calibration_results` — so every calibration number page 06 computed
        # stopped at the app. The header is emitted only when at least one model
        # produced a readable sentence, so a records dict of empty artifacts
        # cannot open a Calibration section that then says nothing.
        if calibration_results:
            cal_sentences = [
                s for s in (
                    _calibration_sentence(name, cal)
                    for name, cal in calibration_results.items()
                ) if s
            ]
            if cal_sentences:
                sections.append("\n### Calibration\n")
                sections.append("Calibration outputs are reported only for models with computed calibration artifacts.\n\n")
                sections.extend(cal_sentences)

        # FIX 4: Statistical Validation Results
        stat_val_entries = logged_steps.get('Statistical Validation', [])
        if stat_val_entries:
            sections.append("\n### Statistical Validation\n")
            for entry in stat_val_entries:
                action = entry.get('action', '')
                details = entry.get('details', {})
                test_name = details.get('test_name') or action
                variable = details.get('variable', 'unknown variable')
                statistic = details.get('statistic')
                p_value = details.get('p_value')
                
                if statistic is not None and p_value is not None:
                    p_text = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
                    sections.append(
                        f"{test_name} was performed on {variable}: "
                        f"statistic = {statistic:.4f}, {p_text}.\n\n"
                    )
                elif action:
                    sections.append(f"{action}.\n\n")
            
            # Multiple testing caveat if >3 tests
            if len(stat_val_entries) > 3:
                sections.append(
                    "Note: Multiple statistical tests were performed; "
                    "readers should consider the increased risk of Type I error "
                    "when interpreting individual p-values.\n\n"
                )

    if ledger_narratives:
        sections.append('\n\n### Data Quality and Preprocessing Rationale\n')
        sections.append(
            'The following observations were identified during exploratory analysis '
            'and addressed during the modeling workflow:\n'
        )
        for phase, narrative in ledger_narratives.items():
            sections.append(f'**{phase}:** {narrative}\n')

    return "\n".join(sections)


# ============================================================================
# Flow Diagram (CONSORT-style)
# ============================================================================

def generate_flow_diagram_mermaid(
    n_total: int,
    n_excluded: int = 0,
    exclusion_reasons: Optional[Dict[str, int]] = None,
    n_missing_target: int = 0,
    n_analyzed: int = 0,
    n_train: int = 0,
    n_val: int = 0,
    n_test: int = 0,
) -> str:
    """Generate a Mermaid flowchart for sample selection.

    Returns Mermaid diagram text that can be rendered in Streamlit
    or exported to SVG/PNG.
    """
    lines = ["graph TD"]
    lines.append(f'    A["Total records<br/>N = {n_total:,}"]')

    if n_excluded > 0 or exclusion_reasons:
        if exclusion_reasons:
            reasons = "<br/>".join(f"• {r}: n={n}" for r, n in exclusion_reasons.items())
            lines.append(f'    B["Excluded<br/>n = {n_excluded:,}<br/>{reasons}"]')
        else:
            lines.append(f'    B["Excluded<br/>n = {n_excluded:,}"]')
        lines.append("    A --> B")
        n_after = n_total - n_excluded
        lines.append(f'    C["After exclusions<br/>n = {n_after:,}"]')
        lines.append("    A --> C")
    else:
        lines.append(f'    C["All records<br/>n = {n_total:,}"]')
        lines.append("    A --> C")

    if n_missing_target > 0:
        lines.append(f'    D["Missing target<br/>n = {n_missing_target:,}"]')
        lines.append("    C --> D")
        n_analyzed = n_analyzed or (n_total - n_excluded - n_missing_target)
        lines.append(f'    E["Analyzed<br/>n = {n_analyzed:,}"]')
        lines.append("    C --> E")
    else:
        n_analyzed = n_analyzed or (n_total - n_excluded)
        lines.append(f'    E["Analyzed<br/>n = {n_analyzed:,}"]')
        lines.append("    C --> E")

    if n_train > 0:
        lines.append(f'    F["Training set<br/>n = {n_train:,}"]')
        lines.append(f'    G["Validation set<br/>n = {n_val:,}"]')
        lines.append(f'    H["Test set<br/>n = {n_test:,}"]')
        lines.append("    E --> F")
        lines.append("    E --> G")
        lines.append("    E --> H")

    return "\n".join(lines)


# ============================================================================
# Subgroup Analysis
# ============================================================================

def subgroup_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subgroup_labels: np.ndarray,
    task_type: str = "regression",
    metric_fn=None,
    metric_name: str = "Metric",
    n_bootstrap: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute metrics stratified by subgroup with bootstrap CIs.

    Args:
        y_true: True values
        y_pred: Predicted values
        subgroup_labels: Group labels for each sample
        task_type: 'regression' or 'classification'
        metric_fn: Custom metric function (default: RMSE for regression, accuracy for classification)
        metric_name: Name of the metric
        n_bootstrap: Number of bootstrap resamples
        random_state: Random seed

    Returns:
        DataFrame with subgroup, n, metric, CI_lower, CI_upper
    """
    from ml.bootstrap import bootstrap_metric

    if metric_fn is None:
        if task_type == "regression":
            from sklearn.metrics import mean_squared_error
            metric_fn = lambda yt, yp: np.sqrt(mean_squared_error(yt, yp))
            metric_name = "RMSE"
        else:
            from sklearn.metrics import accuracy_score
            metric_fn = accuracy_score
            metric_name = "Accuracy"

    groups = np.unique(subgroup_labels)
    rows = []

    # Overall
    overall_result = bootstrap_metric(
        y_true, y_pred, metric_fn,
        n_resamples=n_bootstrap, metric_name=metric_name, random_state=random_state,
    )
    rows.append({
        "Subgroup": "Overall",
        "N": len(y_true),
        metric_name: f"{overall_result.estimate:.4f}",
        "95% CI": f"[{overall_result.ci_lower:.4f}, {overall_result.ci_upper:.4f}]",
        "_estimate": overall_result.estimate,
        "_ci_lower": overall_result.ci_lower,
        "_ci_upper": overall_result.ci_upper,
    })

    for g in groups:
        mask = subgroup_labels == g
        if mask.sum() < 5:
            continue
        result = bootstrap_metric(
            y_true[mask], y_pred[mask], metric_fn,
            n_resamples=n_bootstrap, metric_name=metric_name, random_state=random_state,
        )
        rows.append({
            "Subgroup": str(g),
            "N": int(mask.sum()),
            metric_name: f"{result.estimate:.4f}",
            "95% CI": f"[{result.ci_lower:.4f}, {result.ci_upper:.4f}]",
            "_estimate": result.estimate,
            "_ci_lower": result.ci_lower,
            "_ci_upper": result.ci_upper,
        })

    return pd.DataFrame(rows)


def plot_forest_subgroups(subgroup_df: pd.DataFrame, metric_name: str = "Metric"):
    """Generate a forest plot for subgroup analysis.

    Returns a Plotly figure.
    """
    import plotly.graph_objects as go

    df = subgroup_df.copy()
    df = df.iloc[::-1]  # Reverse for plotting

    fig = go.Figure()

    # Error bars
    fig.add_trace(go.Scatter(
        x=df["_estimate"],
        y=df["Subgroup"],
        mode="markers",
        marker=dict(size=10, color="steelblue"),
        error_x=dict(
            type="data",
            symmetric=False,
            array=df["_ci_upper"] - df["_estimate"],
            arrayminus=df["_estimate"] - df["_ci_lower"],
        ),
        name=metric_name,
    ))

    # Overall reference line
    overall = df.loc[df["Subgroup"] == "Overall", "_estimate"]
    if len(overall) > 0:
        fig.add_vline(x=overall.iloc[0], line_dash="dash", line_color="gray")

    fig.update_layout(
        title=f"Subgroup Analysis — {metric_name}",
        xaxis_title=metric_name,
        yaxis_title="",
        height=max(300, 60 * len(df)),
    )

    return fig


# ============================================================================
# FIX 7: Decision Audit Trail
# ============================================================================

_AUDIT_PHASE_MAP = {
    'Upload & Audit': 'Data Preparation',
    'Data Cleaning': 'Data Preparation',
    'EDA': 'Data Preparation',
    'Preprocessing': 'Data Preparation',
    'Feature Engineering': 'Feature Engineering',
    'Feature Selection': 'Feature Engineering',
    'Feature Selection Applied': 'Feature Engineering',
    'Model Training': 'Model Selection',
    'Explainability': 'Evaluation',
    'Sensitivity Analysis': 'Evaluation',
    'Statistical Validation': 'Evaluation',
}


def _audit_phase_for_step(step: str) -> str:
    """Map workflow steps to publication-friendly appendix phases."""
    return _AUDIT_PHASE_MAP.get(step, 'Workflow Decisions')


def _clean_audit_text(text: Any) -> str:
    """Remove formatting artifacts that should not appear in the appendix."""
    if text is None:
        return ""
    cleaned = str(text)
    cleaned = cleaned.replace("<!--", "").replace("-->", "")
    cleaned = cleaned.replace("% \\begin{figure}", "").replace("% \\end{figure}", "")
    cleaned = " ".join(cleaned.split())
    try:
        from utils.insight_ledger import _clean_for_manuscript

        cleaned = _clean_for_manuscript(cleaned)
    except Exception:
        pass
    return cleaned.strip(" .-–—:;")


def _format_audit_timestamp(timestamp: str) -> str:
    """Format timestamps consistently for the audit appendix.

    `RECORD-017`: a NAIVE timestamp is not UTC. Every entry in the Record and
    the Ledger is `datetime.now().isoformat()` — local wall clock, no tzinfo —
    and this used to stamp each one 'UTC', so every line of the exported
    Decision Audit Trail was off by the machine's offset (and near midnight, off
    by a day) with nothing to explain the skew to a reviewer reconciling it
    against lab notes or a commit log. A naive stamp is now rendered without a
    zone label: silent about a fact this function does not know, rather than
    asserting one it does not. Aware stamps are still converted and labeled.
    """
    if not timestamp:
        return "Timestamp unavailable"
    try:
        dt = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return dt.strftime("%Y-%m-%d %H:%M")
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return _clean_audit_text(timestamp)


def _summarize_audit_details(details: Dict[str, Any]) -> str:
    """Collapse structured details into a short appendix-friendly summary."""
    if not details:
        return ""

    parts: List[str] = []

    n_before = details.get('n_before') or details.get('n_features_before')
    n_after = details.get('n_after') or details.get('n_features_after')
    if n_before is not None and n_after is not None:
        parts.append(f"feature count {n_before} -> {n_after}")

    n_created = details.get('n_created') or details.get('n_features_created')
    if n_created:
        parts.append(f"{n_created} engineered features added")

    rows = details.get('rows') or details.get('n_rows') or details.get('n_observations')
    cols = details.get('cols') or details.get('n_cols') or details.get('n_features')
    target = details.get('target') or details.get('target_col')
    if rows:
        row_part = f"{rows} rows"
        if cols:
            row_part += f", {cols} columns"
        if target:
            row_part += f", target={target}"
        parts.append(row_part)

    task_type = details.get('task_type')
    if task_type:
        parts.append(f"task={task_type}")

    method = details.get('method') or details.get('imputation') or details.get('strategy')
    if method:
        parts.append(f"method={_clean_audit_text(method)}")

    models = details.get('models') or details.get('models_trained')
    if isinstance(models, list) and models:
        parts.append(f"models={', '.join(_clean_audit_text(m) for m in models[:4])}")

    analyses = details.get('analyses')
    if isinstance(analyses, list) and analyses:
        parts.append(f"analyses={', '.join(_clean_audit_text(a) for a in analyses[:4])}")

    if not parts:
        for key, value in details.items():
            if key in {'finding', 'category', 'action_type', 'timestamp'}:
                continue
            if value in (None, "", [], {}):
                continue
            if isinstance(value, list):
                rendered = ", ".join(_clean_audit_text(v) for v in value[:4])
            else:
                rendered = _clean_audit_text(value)
            if rendered:
                parts.append(f"{key}={rendered}")
            if len(parts) >= 3:
                break

    return "; ".join(parts)


def generate_decision_audit_trail() -> str:
    """Generate a grouped, deduplicated decision appendix from workflow logs."""
    try:
        import streamlit as st
        methodology_log = st.session_state.get('methodology_log', [])
        try:
            from utils.insight_ledger import get_ledger
            ledger_log = get_ledger().get_methodology_log()
        except Exception:
            ledger_log = []
    except ImportError:
        return ""

    source_entries = ledger_log or methodology_log
    if not source_entries:
        return ""

    sorted_log = sorted(source_entries, key=lambda x: x.get('timestamp', ''))
    grouped_entries: Dict[str, List[str]] = {}
    seen_signatures = set()

    for entry in sorted_log:
        step = _clean_audit_text(entry.get('step', 'Workflow Decisions')) or 'Workflow Decisions'
        action = _clean_audit_text(entry.get('action', 'Recorded workflow decision')) or 'Recorded workflow decision'
        details = entry.get('details') or {}
        finding = _clean_audit_text(details.get('finding', ''))
        detail_summary = _clean_audit_text(_summarize_audit_details(details))
        timestamp = _format_audit_timestamp(entry.get('timestamp', ''))

        if finding and finding.lower() != action.lower():
            rationale = finding
        elif detail_summary:
            rationale = f"Recorded parameters: {detail_summary}"
        else:
            rationale = "Workflow configuration recorded for reproducibility."

        signature = (
            _audit_phase_for_step(step).lower(),
            action.lower(),
            rationale.lower(),
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        grouped_entries.setdefault(_audit_phase_for_step(step), []).append(
            f"{timestamp} | Action: {action}. Rationale: {rationale}."
        )

    if not grouped_entries:
        return ""

    phase_order = ['Data Preparation', 'Feature Engineering', 'Model Selection', 'Evaluation', 'Workflow Decisions']
    lines: List[str] = []
    for phase in phase_order:
        phase_entries = grouped_entries.get(phase)
        if not phase_entries:
            continue
        lines.append(f"### {phase}")
        for idx, text in enumerate(phase_entries, 1):
            lines.append(f"{idx}. {text}")
        lines.append("")

    return "\n".join(lines).strip()

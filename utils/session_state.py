"""
Session state management for multi-page Streamlit app.
Defines schema and initialization functions.
"""
import streamlit as st
from typing import Optional, Dict, Any, List, Literal
from dataclasses import dataclass, field
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

# Headless — the key name lives with the graph that defines what a branch is.
from turbotab.cascade import BRANCH_ARCHIVE_KEY


@dataclass
class TaskTypeDetection:
    """Task type detection results and overrides."""
    detected: Optional[Literal["regression", "classification"]] = None
    confidence: Optional[Literal["low", "med", "high"]] = None
    reasons: List[str] = field(default_factory=list)
    override_enabled: bool = False
    override_value: Optional[Literal["regression", "classification"]] = None
    
    @property
    def final(self) -> Optional[Literal["regression", "classification"]]:
        """Get final task type (override if enabled, else detected)."""
        if self.override_enabled and self.override_value is not None:
            return self.override_value
        return self.detected


@dataclass
class CohortStructureDetection:
    """Cohort structure detection results and overrides."""
    detected: Optional[Literal["cross_sectional", "longitudinal"]] = None
    confidence: Optional[Literal["low", "med", "high"]] = None
    reasons: List[str] = field(default_factory=list)
    override_enabled: bool = False
    override_value: Optional[Literal["cross_sectional", "longitudinal"]] = None
    entity_id_candidates: List[str] = field(default_factory=list)
    entity_id_detected: Optional[str] = None
    entity_id_override_enabled: bool = False
    entity_id_override_value: Optional[str] = None
    time_column_candidates: List[str] = field(default_factory=list)
    
    @property
    def final(self) -> Optional[Literal["cross_sectional", "longitudinal"]]:
        """Get final cohort type (override if enabled, else detected)."""
        if self.override_enabled and self.override_value is not None:
            return self.override_value
        return self.detected
    
    @property
    def entity_id_final(self) -> Optional[str]:
        """Get final entity ID column (override if enabled, else detected)."""
        if self.entity_id_override_enabled and self.entity_id_override_value is not None:
            return self.entity_id_override_value
        return self.entity_id_detected


@dataclass
class DataConfig:
    """Configuration for dataset and target/feature selection."""
    target_col: Optional[str] = None
    feature_cols: List[str] = field(default_factory=list)
    datetime_col: Optional[str] = None  # For time-series splits
    task_type: Optional[str] = None  # 'regression' or 'classification' (DEPRECATED: use task_type_detection.final)


@dataclass
class SplitConfig:
    """Configuration for train/val/test splits."""
    train_size: float = 0.7
    val_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42
    stratify: bool = False  # For classification
    use_time_split: bool = False  # Use datetime_col for splitting
    datetime_col: Optional[str] = None  # Column to use for time-based splitting
    target_trim_enabled: bool = False  # Remove rows where target is outside quantile range (regression only)
    target_trim_lower: float = 0.0  # Lower quantile threshold (0.0–0.5)
    target_trim_upper: float = 1.0  # Upper quantile threshold (0.5–1.0)
    target_transform: str = 'none'  # 'none', 'log1p', 'yeo-johnson', 'box-cox'


@dataclass
class ModelConfig:
    """Configuration for model hyperparameters."""
    # Neural Network
    nn_epochs: int = 200
    nn_batch_size: int = 256
    nn_lr: float = 0.001
    nn_weight_decay: float = 1e-5
    nn_patience: int = 30
    nn_dropout: float = 0.1
    
    # Random Forest
    rf_n_estimators: int = 500
    rf_max_depth: Optional[int] = None
    rf_min_samples_leaf: int = 10
    
    # GLM/Huber
    huber_epsilon: float = 1.35
    huber_alpha: float = 0.0


def init_session_state():
    """Initialize the session state variables **this module declares**.

    `AUDIT-011`: this used to say *"Initialize all session state variables with
    defaults."* It does not and cannot. The dict below declares 50 keys; a sweep
    of `pages/`, `utils/` and `ml/` finds 128 more that are read from
    `st.session_state` and initialized nowhere — page-local widget state, cached
    results, and one that matters.

    **The one that matters is the purpose, and its absence is recorded here
    rather than left to be discovered.** `DOMAIN_SCIENCE.md` §01.3 names five
    decisions whose correct handling *inverts* on whether the model is for
    prediction or for estimating an association. This workflow has no field for
    that answer: `pages/06_Train_and_Compare.py` reads a `model_purpose` key
    when it composes the class-weighting advisory, and nothing in this
    repository ever writes one, so that advisory can only ever say the purpose
    is unrecorded. The Guided door asks the question and records it on
    `AnalysisProject.purpose`, which is the authoritative record for it.

    **`task_mode` is not that answer and must not be read as one.** It gates
    which pages are reachable, and `pages/06` refuses to run unless it is
    already `'prediction'` — so mapping it onto the purpose would hand every
    trained model a prediction objective nobody chose. `turbotab/purpose.py` is
    explicit that the purpose is always asked, never inferred and never
    defaulted; a slot filled from a gate would be the app deciding what the
    user's paper is about. The field stays absent until the question is asked
    on this door.
    """
    defaults = {
        # Data
        'raw_data': None,
        'df_engineered': None,  # Dataset after feature engineering
        'feature_engineering_applied': False,
        'engineered_feature_names': [],
        'selected_features': [],
        'data_config': DataConfig(),
        'data_audit': None,
        
        # Project-based dataset management
        'task_mode': None,  # 'prediction' | 'hypothesis_testing'
        'datasets_registry': {},  # Dict mapping dataset_id -> DataFrame
        'working_table': None,  # The merged/active DataFrame for analysis
        'merge_steps': [],  # List of merge operations
        'last_merge_columns': [],  # Columns from the last merge result
        
        # Detection and triage
        'task_type_detection': TaskTypeDetection(),
        'cohort_structure_detection': CohortStructureDetection(),
        
        # Preprocessing
        'preprocessing_pipeline': None,
        'preprocessing_config': None,
        'preprocessing_pipelines_by_model': {},
        'preprocessing_config_by_model': {},
        
        # Splits
        'split_config': SplitConfig(),
        'X_train': None,
        'X_val': None,
        'X_test': None,
        'y_train': None,
        'y_val': None,
        'y_test': None,
        'feature_names': None,
        'feature_names_by_model': {},
        
        # Models
        'model_config': ModelConfig(),
        'trained_models': {},  # Dict[str, Any] - model name -> model wrapper object
        'model_results': {},  # Dict[str, Dict] - model name -> metrics/history
        'fitted_estimators': {},  # Dict[str, Any] - model name -> fitted sklearn-compatible estimator/pipeline
        'fitted_preprocessing_pipelines': {},  # Dict[str, Pipeline] - model name -> preprocessing pipeline used
        
        # Evaluation
        'cv_results': None,  # For k-fold CV
        # Default ON. pages/06 reads `get('use_cv', True)` for the checkbox, but
        # this seed runs first, so the seed IS the default a user sees — a False
        # here makes that fallback unreachable and ships CV off.
        'use_cv': True,
        'cv_folds': 5,
        
        # Explainability
        'permutation_importance': {},
        'partial_dependence': {},
        'explainability_robustness': {},

        # EDA
        'eda_results': {},  # Dict[str, Dict] - recommendation_id -> results
        'eda_insights': [],  # LEGACY — backward compat, computed from insight_ledger
        
        # Report
        'report_data': None,
        
        # Global settings
        'random_seed': 42,  # Global random seed
        'data_source': None,  # Track data source (uploaded CSV, built-in dataset, etc.)
        'data_filename': None,  # Track filename for uploads or dataset label
        'dataset_id': None,  # Incrementing dataset identifier
        'dataset_history': [],  # Archive of replaced datasets (metadata only)
        'has_completed_tour': False,  # Guided tour dismissed/completed
        'show_guided_tour': False,  # Expand guided tour in sidebar
        'workflow_mode': 'quick',  # 'quick' | 'advanced' navigation emphasis only
        
        # Methodology logging for auto-generated methods section
        'methodology_log': [],  # List of methodology actions for publication
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Insight ledger — single logical layer for cross-page insight tracking.
    # Initialized separately because it's a class instance, not a plain default.
    if 'insight_ledger' not in st.session_state:
        from utils.insight_ledger import InsightLedger
        st.session_state.insight_ledger = InsightLedger()


def get_data(full_study: bool = False,
             apply_row_filter: bool = True) -> Optional[pd.DataFrame]:
    """Get active data from session state.
    Columns come from df_engineered (if feature engineering was applied), else
    raw_data. Rows are then masked by filtered_data (the preprocess row
    filter) whenever it exists — it is a filter, not a competing frame.

    When a cohort run is active ("same question, different people"), the rows of
    that cohort are ALL any page sees — the filter is applied here, once, rather
    than in each of the nine pages that would each have to remember. Pass
    full_study=True for the two things that must span the whole study: drawing
    the test lockbox (every cohort inherits its slice of ONE split) and choosing
    the target and features (fixed across runs, which is what makes the runs
    comparable at all).

    `apply_row_filter=False` returns the frame the row filter is computed FROM —
    the engineered/raw frame with the cohort applied but WITHOUT the
    filtered_data mask. Exactly one caller needs it: the code that RECOMPUTES
    filtered_data (`pages/05`). Filtering the already-filtered frame and writing
    the result back is a one-way ratchet — every rebuild can only ever remove
    more rows, and widening a plausibility bound restores none of them
    (`STATE-037`). Every other caller wants the masked frame.
    """
    # Explicitly check for None to avoid DataFrame boolean ambiguity
    df_eng = st.session_state.get('df_engineered')
    df_filt = st.session_state.get('filtered_data') if apply_row_filter else None
    if df_eng is not None:
        df = df_eng
        if df_filt is not None:
            # A row filter is a MASK, not a rival frame. df_engineered used to
            # win outright, so a plausibility filter applied after feature
            # engineering — the documented page order — was written to
            # filtered_data and then read by nobody: page 05 said the rows were
            # removed and they trained the model anyway (CONTRACT-013).
            # Applying it here keeps that sentence true whichever order the two
            # steps ran in; when FE ran second, df_engineered is already the
            # narrower frame and the mask changes nothing.
            _kept = df.index.isin(df_filt.index)
            if _kept.any():
                df = df[_kept]
            # No overlap at all means the two frames are different vintages,
            # not a filter — masking would empty the dataset. Only set_data and
            # a new cohort can produce that, and both pop filtered_data.
    else:
        df = df_filt if df_filt is not None else st.session_state.get('raw_data')

    from utils.cohorts import active_cohort, apply_cohort
    if df is None:
        return df
    if full_study:
        # "The whole study" cannot be recovered by skipping ONE filter. Feature
        # engineering and preprocess row-filters write df_engineered and
        # filtered_data from whatever get_data() handed them, so inside a run
        # those frames are themselves cohort-sized — and full_study=True was
        # returning one cohort's rows while its two callers, the test lockbox
        # and the cohort chooser, both documented that they need all of them.
        # The lockbox was then redrawn on that subset and rows sealed since
        # upload became trainable.
        #
        # Starting a run clears df_engineered, so any narrowed frame that
        # exists during a run was built inside it. The study is therefore the
        # data as uploaded (raw_data tracks cleaning actions via set_data).
        if active_cohort() is not None:
            return st.session_state.get('raw_data', df)
        return df
    return apply_cohort(df)


def get_split_rows(part: str = "test",
                   df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """The active frame's rows for a stored split partition, addressed by LABEL.

    The one way any page reads a partition back after Train & Compare drew it.
    Raises `ml.splits.SplitIdentityError` when the active frame no longer
    contains every row the split recorded — the split was drawn on a different
    set of people, and no answer computed here would be about the reported
    test set. Callers surface the refusal; none of them may fall back to
    selecting rows by position.

    `part` is "train", "val" or "test". Pass `df` to resolve against a frame
    already in hand instead of re-fetching.
    """
    from ml.splits import resolve_split_rows
    labels = st.session_state.get(f"{part}_row_labels")
    if df is None:
        df = get_data()
    return resolve_split_rows(df, labels, part=part)


def _hashable_cell(value: Any) -> Any:
    """Render one cell in a form pandas can hash, preserving its content."""
    if isinstance(value, np.ndarray):
        return repr(value.tolist())
    if isinstance(value, (set, frozenset)):
        return repr(sorted(map(repr, value)))
    if isinstance(value, (list, tuple, dict)):
        return repr(value)
    return value


def _content_fingerprint(df: pd.DataFrame) -> Optional[int]:
    """Cheap deterministic fingerprint of a DataFrame's contents AND row labels.

    Stable across .copy() (hashes values, not identity).

    `MINE-010`/`STATE-044`: this used to return None for any frame pandas could
    not hash — a parquet upload with a list/array column is enough — and the
    caller read None as 'unchanged', so a corrected re-upload with the same
    schema kept every stale model, metric and figure. Unhashable cells are now
    stringified for hashing instead. None is reserved for a frame we genuinely
    cannot fingerprint, and callers must treat it as CHANGED, never as clean.

    `STATE-044` (second half): the sum was taken with `index=False`, which makes
    it a hash of the *multiset* of rows and therefore blind to WHICH ROW IS
    WHICH. A re-upload of the same people sorted differently — a RangeIndex, so
    every label now names a different person — produced an identical
    fingerprint, and set_data took the benign-rerun branch: models, metrics and
    `test_row_labels` survived, and the sealed test set silently became 50 other
    people that `resolve_split_rows` had no reason to refuse. Each row's content
    is now paired WITH its label before summing, so a label-faithful reorder
    (the labels travel with their rows) still fingerprints the same while
    relabeled content does not.
    """
    try:
        return hash((df.shape, int(pd.util.hash_pandas_object(df, index=True).sum())))
    except Exception:
        pass
    try:
        coerced = df.copy(deep=False)
        for col in coerced.columns:
            if coerced[col].dtype == object:
                coerced[col] = coerced[col].map(_hashable_cell)
        return hash((df.shape,
                     int(pd.util.hash_pandas_object(coerced, index=True).sum())))
    except Exception:
        return None


def set_data(df: pd.DataFrame, is_schema_change: Optional[bool] = None):
    """Set raw data in session state. Clears filtered_data so it is not stale.

    Invalidation contract:
    - Column set changed (or is_schema_change=True): full reset via
      reset_data_dependent_state() — new dataset, new analysis.
    - Same columns but different values (re-upload of corrected data, cleaning
      actions): configuration (target, features, ledger, logs) is kept, but all
      downstream RESULTS (engineered features, splits, models, metrics,
      explainability, reports) are cleared via reset_downstream_results() —
      they describe data that no longer exists.
    - Identical content (benign rerun): no-op.

    is_schema_change=False suppresses only the full config reset; it does NOT
    keep results computed from different data.
    """
    if not df.index.is_unique:
        # The test-set lockbox and train/test masks identify rows by index
        # LABEL; duplicate labels would silently over-select rows into both
        # partitions (e.g. a parquet upload that preserved a non-unique index).
        df = df.reset_index(drop=True)

    old_df = st.session_state.get('raw_data')
    old_cols = frozenset(old_df.columns) if old_df is not None else None
    new_cols = frozenset(df.columns)

    old_fp = st.session_state.get('_raw_data_fingerprint')
    if old_fp is None and old_df is not None:
        # A restored session carries the frame but not the fingerprint. Recompute
        # it rather than let 'unknown' mean 'changed' below, which would clear the
        # analysis on the benign page-01 revisit that re-sets the same table.
        old_fp = _content_fingerprint(old_df)
    new_fp = _content_fingerprint(df)

    st.session_state.raw_data = df
    st.session_state['_raw_data_fingerprint'] = new_fp
    st.session_state.pop("filtered_data", None)

    if is_schema_change is None:
        is_schema_change = (old_cols is not None and old_cols != new_cols)

    # A cohort is a set of row labels in the PREVIOUS data, so genuinely new
    # data must drop it — it would either match nobody or, worse, match
    # different people. Re-setting the SAME frame must not, which is what page
    # 01 does on every visit while restoring its working table: clearing there
    # would end the run the moment the researcher looked at the upload page.
    if is_schema_change:
        reset_data_dependent_state()      # clears the cohort itself
    elif (old_df is not None
          and (old_fp is None or new_fp is None or old_fp != new_fp)):
        # `MINE-010`: 'we could not fingerprint this frame' is not evidence that
        # nothing changed. Both fingerprints had to be non-None to reach here,
        # so an unhashable frame took NEITHER branch and every model, metric and
        # figure from the previous values survived under the new dataset's name.
        # Unknown now means changed: the reset is the safe answer, a stale
        # manuscript number is not.
        #
        # Results are stale — they were computed from different values. But the
        # RUN is a set of row labels, and a cleaning action (impute, clip,
        # recode) changes values without changing who the rows are. Clearing it
        # here meant that restoring a saved one-group session and opening
        # Upload & Audit silently reverted the analysis to the whole study.
        # Clear only when the cohort's people are no longer all present.
        from utils.cohorts import active_cohort, clear_cohort
        _run = active_cohort()
        if _run is not None and set(_run["labels"]) <= set(df.index):
            _run.pop("_label_set", None)      # cached set may be stale
        else:
            if _run is not None:
                st.session_state["_cohort_cleared_by_data_change"] = {
                    "column": _run["column"], "label": _run["label"]}
            clear_cohort()
        reset_downstream_results()


# ---------------------------------------------------------------------------
# The downstream result registry
# ---------------------------------------------------------------------------
# `STATE-038`: every key below is a RESULT computed from the current data, and
# the only thing standing between a superseded result and the exported
# manuscript is its membership here. The lists used to live inline inside
# reset_downstream_results, where a key added on a page was simply never added
# here — pdp_results survived while `partial_dependence` beside it was cleared,
# and the dropout-sensitivity keys survived to let ml/publication.py assert a
# sensitivity analysis on a model the same reset had destroyed. A new result
# key goes in one of these tuples; nothing else clears them.

# Splits and the objects that describe one partition of them.
_SPLIT_KEYS: tuple = (
    "train_indices", "val_indices", "test_indices",
    # Row LABELS alongside the positions. Added with the split extraction (L6);
    # they describe the same partition, so they go stale with it.
    "train_row_labels", "val_row_labels", "test_row_labels",
    "split_config",
    "target_transformer", "target_label_encoder",
    "y_train_original", "y_val_original", "y_test_original",
    "cv_strategy", "cv_groups_train",
    # The realized target-trim (thresholds, basis, test_rows_exempt, n_trimmed)
    # that pages/06 persists from the Split it just drew (`CONTRACT-021`). It
    # describes one split and goes stale with it.
    "split_trim_record",
)

# Analysis results from pages 02–09.
_ANALYSIS_KEYS: tuple = (
    "shap_results", "shap_matplotlib_figs", "bootstrap_results",
    "baseline_results", "calibration_results",
    "sensitivity_seed_results",
    # Written by pages/08 and read by ml/publication.py to state that a
    # feature-dropout sensitivity analysis was performed.
    "sensitivity_dropout_results", "sensitivity_dropout_baseline",
    "hypothesis_test_results",
    "table1_df", "table1_metadata", "custom_table1_tests",
    "table1_custom_test_footnotes",
    "dataset_profile",
    # WHICH ROWS the profile above describes (pages/02 writes it, pages/10
    # prints it beside the p/n ratio and the sufficiency verdict). A scope note
    # that outlives its profile labels the next one's numbers.
    "dataset_profile_scope",
    # Written by pages/07 next to `partial_dependence`, read by pages/10.
    "pdp_results",
    # External-cohort metrics written by pages/07. They describe models fitted
    # on the data being replaced, and ml/publication.py states that external
    # validation was performed on the strength of them.
    "external_validation_results",
    "bland_altman_results", "preprocessing_summary",
)

_FEATURE_SELECTION_KEYS: tuple = ("feature_selection_results", "consensus_features")

# Report artifacts.
_REPORT_KEYS: tuple = (
    "methods_section", "flow_diagram", "tripod_tracker", "latex_report",
    "report_best_model", "report_model_selection", "report_explain_selection",
    "report_include_results", "report_include_llm", "manuscript_context",
    # A stale manuscript_export_context WINS over rebuilding (pages/10 only
    # rebuilds `if manuscript_context is None`), so it is the most dangerous
    # survivor in this tuple, not the least.
    "manuscript_export_context", "compiled_pdf",
    "manuscript_table1_df", "manuscript_table1_metadata",
)


def reset_downstream_results(clear_feature_engineering: bool = True,
                             restore_pre_fe_features: bool = True,
                             clear_feature_selection: bool = True,
                             preserve_branches: bool = False):
    """Clear every RESULT computed from the current data, keeping configuration.

    Single source of truth for downstream invalidation — used by
    reset_data_dependent_state() (full data change), set_data() (same-schema
    content change), and Page 01 (feature/target/task change). Any page that
    introduces a new result key must register it in the tuples above; the
    provenance sections it nulls come from the record's own schema.

    clear_feature_selection=False preserves the feature-selection results,
    consensus list, and its provenance/ledger entries. Feature Selection uses
    this when it APPLIES a new selection: the pipelines/splits/models built on
    the old feature set are stale and must go, but the selection just made (and
    its record) must survive.

    **Three idioms, on purpose — for now (`T0-STATE-001`).** This function
    clears via `pop(key, None)`, `= None`, and `= {}` / `= []`, and which one a
    key gets is not arbitrary: some keys are read by *bare attribute access*,
    which raises `AttributeError` if the key is absent. Measured across the
    tree:

        model_results          10 bare reads (pages/06)
        eda_results             5 bare reads (pages/02)
        trained_models          4 bare reads (pages/06)
        fitted_estimators       2 bare reads (pages/06)

    So normalizing everything to `pop` is not a rewrite of this function — it is
    a rewrite of ~25 call sites that must move to `.get()` first. `turbotab.cascade`
    declares the graph once and its `all_result_keys()` is the checklist for
    that work; doing it here alone would turn a stale-value bug into an
    AttributeError on the Train and EDA pages.

    Until then the rule is: a key that anything reads bare is emptied in place;
    everything else is popped.

    **`preserve_branches` (`BRANCH-001`).** Archived cohort branches are dropped
    here, by default, on every call. A branch is a set of results that answered
    one question about one group of people; every caller that reaches this
    function changed that question — the data, the target, the engineering
    recipe, the selection, the preprocessing rule, the seal. The only caller
    that has not is a cohort switch, which changes *who the rows are* and
    nothing else, and it is the only one that may pass True.

    The drop lives INSIDE the reset rather than beside its call sites so that a
    caller written without knowing branches exist cannot leak a stale one by
    omission. There are twelve such callers today and the thirteenth is the
    dangerous one.

    **Fresh objects, never `.clear()` (`BRANCH-002`).** Every line below assigns
    a new container, sets None, or pops. Nothing is emptied in place. A cohort
    switch snapshots the live objects and then calls this function, so clearing
    one in place would empty the snapshot too and lose the branch it had just
    banked. `tests/test_the_branch_archive_survives_the_reset.py` pins it.
    """
    # Archived cohort branches, unless the caller is the cohort switch. See the
    # `preserve_branches` note above: a branch is only comparable under the
    # question it answered, and every caller that reaches here without the flag
    # changed that question. Dropping it first also means the snapshot a switch
    # took moments ago is already safe in the archive before anything below
    # touches the live objects.
    if not preserve_branches:
        st.session_state.pop(BRANCH_ARCHIVE_KEY, None)

    # Feature engineering (df_engineered would otherwise keep serving stale
    # data through get_data()'s precedence)
    if clear_feature_engineering:
        st.session_state.pop("df_engineered", None)
        st.session_state.feature_engineering_applied = False
        st.session_state.engineered_feature_names = []
        st.session_state.pop("engineering_log", None)
        # FE overwrote selected_features with engineered columns; restore the
        # pre-FE selection so the config refers to columns that still exist.
        pre_fe = st.session_state.pop("pre_fe_feature_cols", None)
        if restore_pre_fe_features and pre_fe:
            st.session_state.selected_features = list(pre_fe)
            dc = st.session_state.get("data_config")
            if dc is not None and getattr(dc, "feature_cols", None):
                dc.feature_cols = list(pre_fe)

    # Pipelines
    st.session_state.preprocessing_pipeline = None
    st.session_state.preprocessing_config = None
    st.session_state.preprocessing_pipelines_by_model = {}
    st.session_state.preprocessing_config_by_model = {}
    # The list of models that HAVE pipelines has to go with the pipelines.
    # Leaving it made page 06 badge a model "Tuned for this model" when nothing
    # was left to tune it with, and page 05 report it as already built.
    st.session_state.pop("preprocess_built_model_keys", None)

    # Splits & targets
    st.session_state.X_train = None
    st.session_state.X_val = None
    st.session_state.X_test = None
    st.session_state.y_train = None
    st.session_state.y_val = None
    st.session_state.y_test = None
    st.session_state.feature_names = None
    st.session_state.feature_names_by_model = {}
    for key in _SPLIT_KEYS:
        st.session_state.pop(key, None)

    # A row filter is part of WHO the results were computed on. get_data() masks
    # every page's frame by filtered_data whenever it exists, so a filter left
    # over from a superseded preprocessing config keeps shrinking the dataset
    # across every reset that is not a full data change (`STATE-037`).
    st.session_state.pop("filtered_data", None)

    # Models & metrics
    st.session_state.trained_models = {}
    st.session_state.model_results = {}
    st.session_state.fitted_estimators = {}
    st.session_state.fitted_preprocessing_pipelines = {}
    st.session_state.cv_results = None

    # Analysis results (pages 02-09) — all computed from data values, so all
    # stale the moment the data changes, even with an unchanged schema.
    st.session_state.permutation_importance = {}
    st.session_state.partial_dependence = {}
    st.session_state.explainability_robustness = {}
    st.session_state.eda_results = {}
    st.session_state.eda_insights = []
    _analysis_keys = list(_ANALYSIS_KEYS)
    if clear_feature_selection:
        _analysis_keys += list(_FEATURE_SELECTION_KEYS)
    for key in _analysis_keys:
        st.session_state.pop(key, None)

    # Report artifacts
    st.session_state.report_data = None
    for key in _REPORT_KEYS:
        st.session_state.pop(key, None)

    # Coach evidence describes the data it was measured on — a probe verdict
    # ("learnable signal", "no signal") must never survive a data change and
    # keep steering picks for a dataset it never saw. Same for the one-shot
    # auto-select flag: new data deserves fresh auto-picks.
    st.session_state.pop("coach_probe_result", None)
    st.session_state.pop("_coach_applied", None)

    # Downstream provenance sections now describe work that no longer exists
    # — and the list of them is DERIVED from the record's own schema, because a
    # hand-typed one drifts from it silently. `sensitivity` and
    # `statistical_validation` were missing from the list this used to carry, so
    # the Methods draft kept naming the tests and printing the corrected-
    # significance count while the same reset deleted the results those tests
    # produced, and get_completeness() went on reporting both stages as done
    # (`CONTRACT-034`, `STATE-047`).
    prov = st.session_state.get("workflow_provenance")
    if prov is not None:
        from utils.workflow_provenance import downstream_sections
        for section in downstream_sections(
                clear_feature_engineering=clear_feature_engineering,
                clear_feature_selection=clear_feature_selection):
            if hasattr(prov, section):
                setattr(prov, section, None)

    # The ledger must not keep asserting actions that were just invalidated:
    # roll back resolutions earned on the cleared pages (the findings remain),
    # and drop auto-generated insights outright wherever their producer
    # re-detects on its next visit/run — EDA scans, preprocess guards
    # (high-cardinality, probe findings), and post-training diagnostics all
    # describe data or models that no longer exist. Absent is better than
    # false.
    _rollback_pages = {
        "05_Preprocess", "06_Train_and_Compare",
        "07_Explainability", "08_Sensitivity_Analysis",
        "09_Hypothesis_Testing",
    }
    if clear_feature_selection:
        _rollback_pages |= {"03_Feature_Engineering", "04_Feature_Selection"}
    _pruned_pages = {"02_EDA", "05_Preprocess", "06_Train_and_Compare"}

    ledger = st.session_state.get("insight_ledger")
    if ledger is not None:
        if hasattr(ledger, "rollback_resolutions"):
            ledger.rollback_resolutions(_rollback_pages)
        if hasattr(ledger, "prune_auto_generated"):
            ledger.prune_auto_generated(_pruned_pages)

    # `methodology_log` is the ledger's SECOND producer, and it was in neither
    # reset: pages/10 and ml/publication read `ledger_log or methodology_log`,
    # so the moment the rollback above emptied the ledger's methodology view the
    # fallback took over and re-printed the very Mann-Whitney and Shapiro-Wilk
    # rows whose results this reset had just deleted. It is pruned by the SAME
    # page set as the ledger, so the two sources cannot disagree; a step this
    # module cannot attribute to a page is kept, as everywhere else.
    _mlog = st.session_state.get("methodology_log")
    if _mlog:
        _cleared_pages = _rollback_pages | _pruned_pages
        st.session_state["methodology_log"] = [
            e for e in _mlog
            if not isinstance(e, dict)
            or _STEP_TO_PAGE.get(e.get("step"), "") not in _cleared_pages
        ]

    # A manuscript is only quarantine-clean if EVERY surviving result was
    # computed with the lockbox on, and the applied feature selection
    # (`selected_features`, `pre_fe_feature_cols`, `data_config.feature_cols`)
    # survives EVERY call here — including the full one. `STATE-040`: the pop
    # used to be unconditional, then conditional on the two clear_* flags, and
    # both versions let a same-schema set_data clear the watermark while the
    # selection chosen with the lockbox open stayed applied; the Methods then
    # claimed a clean held-out evaluation of a model fitted on features picked
    # with the test rows in view. Nothing here clears that selection, so nothing
    # here may clear its watermark. Only reset_data_dependent_state — genuinely
    # new data, which empties `selected_features` and rebuilds `data_config` —
    # drops it.


def reset_data_dependent_state():
    """Full reset for a new/replaced dataset: configuration AND results."""
    st.session_state.data_config = DataConfig()
    st.session_state.data_audit = None
    st.session_state.task_type_detection = TaskTypeDetection()
    st.session_state.cohort_structure_detection = CohortStructureDetection()
    # Note: task_mode and datasets_registry are NOT reset here
    # as they are workflow-level, not dataset-specific

    st.session_state.pop("filtered_data", None)
    from utils.cohorts import clear_cohort
    clear_cohort()
    st.session_state.pop("cohort_runs_done", None)
    # Decisions staged by a cohort switch describe the old study's columns;
    # replaying them onto a new file would rebuild features that may not
    # exist and seed settings chosen for other data.
    st.session_state.pop("cohort_replay_pending", None)
    st.session_state.pop("cohort_decisions_pending", None)
    st.session_state.selected_features = []
    # Re-seed to the same default as init_session_state, not to False: a new
    # dataset returns the user to the shipped default, and the shipped default
    # is CV on.
    st.session_state.use_cv = True
    st.session_state.cv_folds = 5

    st.session_state.pop("methodology_log", None)
    st.session_state.pop("workflow_provenance", None)
    # The honesty watermark stains the APPLIED feature selection, which only
    # this reset clears (`selected_features` above, `data_config` at the top).
    # See the note at the end of reset_downstream_results: no partial reset may
    # drop it, because no partial reset drops what it describes.
    st.session_state.pop("exploratory_used", None)
    # New dataset → a fresh test lockbox is drawn on the next config save
    st.session_state.pop("test_lockbox", None)
    st.session_state.pop("_lockbox_ledger_noted", None)

    # Reset insight ledger
    from utils.insight_ledger import InsightLedger
    st.session_state.insight_ledger = InsightLedger()

    reset_downstream_results(restore_pre_fe_features=False)


def ensure_dataset_profile(quiet: bool = True) -> Optional[Any]:
    """The dataset profile for the CURRENT feature set, recomputed if it is not.

    `DRIVE-073`. `dataset_profile` is in `_ANALYSIS_KEYS`, so applying a feature
    selection clears it — correctly, because the profile describes a feature set
    that no longer exists — and nothing on the 04 → 05 → 06 path recomputed it.
    Page 06 then dropped the class-imbalance card, the rebalancing control and
    the model-suitability badges with no word on screen, and page 10's manifest
    read "Dataset profile: Not computed" on a session that had one.

    RECOMPUTED WHERE IT IS READ, not eagerly at apply: the profile is a function
    of (rows, features, target, task), and every one of those can change after an
    apply without passing back through page 04. A value refreshed at the moment
    of use cannot be stale; a value written once at apply can, and a stale
    imbalance ratio is a wrong number on screen rather than a missing card.

    Scope follows `pages/02_EDA.py`: the profile describes the TRAINING rows,
    and `dataset_profile_scope` is rewritten beside it so the two never
    disagree about which population they are about.
    """
    profile = st.session_state.get("dataset_profile")
    data_config = st.session_state.get("data_config")
    if data_config is None:
        return profile

    feature_cols = list(st.session_state.get("selected_features")
                        or getattr(data_config, "feature_cols", None) or [])
    target_col = getattr(data_config, "target_col", None)
    if not feature_cols:
        return profile

    try:
        df = get_data()
    except Exception:
        return profile
    if df is None or df.empty:
        return profile

    try:
        from utils.test_lockbox import train_row_mask
        train_mask = train_row_mask(df.index)
        train_df = df.loc[train_mask]
        if train_df.empty:
            train_df = df
            train_mask = pd.Series(True, index=df.index)
    except Exception:
        train_df = df
        train_mask = pd.Series(True, index=df.index)

    described = set(getattr(profile, "feature_profiles", {}) or {})
    if profile is not None and described == set(feature_cols) \
            and getattr(profile, "n_rows", None) == len(train_df):
        return profile

    task_type = st.session_state.get("task_type_detection")
    task_type = getattr(task_type, "final", None) or getattr(data_config, "task_type", None)

    try:
        from ml.dataset_profile import compute_dataset_profile
        profile = compute_dataset_profile(
            train_df, target_col or feature_cols[0], feature_cols,
            task_type or "regression",
            st.session_state.get("eda_outlier_method", "iqr"),
        )
    except Exception:
        # A profile that cannot be computed leaves the previous answer alone;
        # the callers all treat absence as "these panels are unavailable".
        return st.session_state.get("dataset_profile")

    st.session_state["dataset_profile"] = profile
    _scoped = bool((~train_mask).any())
    st.session_state["dataset_profile_scope"] = {
        "rows": "training" if _scoped else "all",
        "n_rows": int(train_mask.sum()),
        "n_rows_total": int(len(df)),
        "reason": ("held-out test rows are excluded to prevent selection leakage"
                   if _scoped else "no rows are sealed in this analysis"),
    }
    if not quiet:
        st.caption(
            f"Dataset profile recomputed for the current {len(feature_cols)} "
            f"predictor(s) on n={int(train_mask.sum())} training rows."
        )
    return profile


def get_preprocessing_pipeline(model_key: Optional[str] = None) -> Optional[Pipeline]:
    """Get preprocessing pipeline from session state."""
    if model_key:
        pipelines = st.session_state.get('preprocessing_pipelines_by_model', {})
        if model_key in pipelines:
            return pipelines[model_key]
    return st.session_state.get('preprocessing_pipeline')


def set_preprocessing_pipeline(pipeline: Pipeline, config: Dict[str, Any]):
    """Set preprocessing pipeline and config."""
    st.session_state.preprocessing_pipeline = pipeline
    st.session_state.preprocessing_config = config


def set_preprocessing_pipelines(pipelines_by_model: Dict[str, Pipeline], configs_by_model: Dict[str, Any], base_config: Dict[str, Any]):
    """Set model-specific preprocessing pipelines and configs."""
    st.session_state.preprocessing_pipelines_by_model = pipelines_by_model
    st.session_state.preprocessing_config_by_model = configs_by_model
    # Preserve a default pipeline for legacy access
    default_pipeline = pipelines_by_model.get('default') or next(iter(pipelines_by_model.values()), None)
    if default_pipeline:
        st.session_state.preprocessing_pipeline = default_pipeline
    st.session_state.preprocessing_config = base_config


def get_splits() -> Optional[tuple]:
    """Get train/val/test splits from session state."""
    if st.session_state.get('X_train') is None:
        return None
    return (
        st.session_state.X_train,
        st.session_state.X_val,
        st.session_state.X_test,
        st.session_state.y_train,
        st.session_state.y_val,
        st.session_state.y_test,
    )


def set_splits(X_train, X_val, X_test, y_train, y_val, y_test, feature_names: List[str]):
    """Set train/val/test splits in session state."""
    st.session_state.X_train = X_train
    st.session_state.X_val = X_val
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_val = y_val
    st.session_state.y_test = y_test
    st.session_state.feature_names = feature_names


def add_trained_model(name: str, model: Any, results: Dict[str, Any]):
    """Add a trained model and its results to session state."""
    st.session_state.trained_models[name] = model
    st.session_state.model_results[name] = results


def log_methodology(step: str, action: str, details: Optional[Dict[str, Any]] = None):
    """Log a methodology action for the final report.

    Also writes a pre-resolved Insight to the unified ledger so that
    Report Export, TRIPOD auto-completion, and manuscript narrative
    pick up every methodology decision automatically.
    
    Args:
        step: Workflow step name (e.g., 'Feature Engineering', 'Feature Selection')
        action: Description of what was done
        details: Optional dict with additional parameters
    """
    from datetime import datetime, timezone
    
    entry = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'step': step,
        'action': action,
        'details': details or {}
    }
    
    if 'methodology_log' not in st.session_state:
        st.session_state.methodology_log = []
    
    # Steps where re-doing replaces the previous entry (user iterates on a single config)
    REPLACE_STEPS = {'Upload & Audit', 'Feature Engineering', 'Feature Selection Applied',
                     'Preprocessing', 'Model Training', 'Explainability'}
    
    log = st.session_state.methodology_log
    if step in REPLACE_STEPS:
        # Last-wins: replace previous entry with same step
        for i in range(len(log) - 1, -1, -1):
            if log[i]['step'] == step:
                log[i] = entry
                # Also update the corresponding ledger entry
                _log_to_ledger(step, action, details)
                return
    # Additive steps (EDA, Statistical Validation, Data Cleaning) — always append
    log.append(entry)
    _log_to_ledger(step, action, details)


# Mapping from log_methodology step names to ledger fields
_STEP_TO_PAGE = {
    'Upload & Audit': '01_Upload_and_Audit',
    'Data Cleaning': '01_Upload_and_Audit',
    'EDA': '02_EDA',
    'Feature Engineering': '03_Feature_Engineering',
    'Feature Selection': '04_Feature_Selection',
    'Feature Selection Applied': '04_Feature_Selection',
    'Preprocessing': '05_Preprocess',
    'Model Training': '06_Train_and_Compare',
    'Explainability': '07_Explainability',
    'Sensitivity Analysis': '08_Sensitivity_Analysis',
    'Statistical Validation': '09_Hypothesis_Testing',
}

_STEP_TO_CATEGORY = {
    'Upload & Audit': 'data_quality',
    'Data Cleaning': 'data_quality',
    'EDA': 'methodology',
    'Feature Engineering': 'distribution',
    'Feature Selection': 'methodology',
    'Feature Selection Applied': 'methodology',
    'Preprocessing': 'methodology',
    'Model Training': 'model_selection',
    'Explainability': 'explainability',
    'Sensitivity Analysis': 'sensitivity',
    'Statistical Validation': 'validation',
}



# Steps that use a fixed ledger ID (last-wins, matching methodology_log REPLACE semantics)
_REPLACE_STEP_IDS = {
    'Upload & Audit', 'Feature Engineering', 'Feature Selection Applied',
    'Preprocessing', 'Model Training', 'Explainability',
}

# Steps that are activity-only records (belong in audit trail, not narrative)
# These still write to the ledger for provenance but are marked so the
# narrative renderer can exclude them.
_AUDIT_ONLY_STEPS = {'EDA'}


def _log_to_ledger(step: str, action: str, details: Optional[Dict[str, Any]] = None):
    """Bridge: write a pre-resolved Insight entry for each methodology log call.

    Enriches details with structured action_type when inferrable from the step,
    so the narrative renderer can produce publication-quality prose.

    For REPLACE steps (Upload & Audit, Preprocessing, etc.), uses a fixed ID
    so repeated runs overwrite instead of accumulating duplicates.
    """
    try:
        from utils.insight_ledger import Insight, get_ledger
        from datetime import datetime

        ledger = get_ledger()
        page = _STEP_TO_PAGE.get(step, '02_EDA')
        category = _STEP_TO_CATEGORY.get(step, 'methodology')

        # Fixed ID for replace-steps; action-based slug for additive steps
        if step in _REPLACE_STEP_IDS:
            insight_id = f"method_{step.lower().replace(' ', '_').replace('&', 'and')}"
        else:
            slug = action.lower().replace(' ', '_')[:40]
            insight_id = f"method_{step.lower().replace(' ', '_')}_{slug}"

        # Enrich details with structured schema fields when inferrable
        enriched = dict(details) if details else {}
        if "action_type" not in enriched:
            _step_to_action_type = {
                "Preprocessing": "preprocessing",
                "Feature Engineering": "transform",
                "Feature Selection": "feature_selection",
                "Feature Selection Applied": "feature_selection",
                "Model Training": "training",
                "Upload & Audit": "data_setup",
                "Data Cleaning": "data_cleaning",
            }
            inferred = _step_to_action_type.get(step)
            if inferred:
                enriched["action_type"] = inferred
            # Infer method from details when possible
            if "method" not in enriched:
                if enriched.get("imputation"):
                    enriched["method"] = enriched["imputation"]
                elif enriched.get("scaling"):
                    enriched["method"] = enriched["scaling"]

        # Mark audit-only entries so narrative can exclude them
        is_audit_only = step in _AUDIT_ONLY_STEPS

        ledger.upsert(Insight(
            id=insight_id,
            source_page=page,
            category=category,
            severity="info",
            finding=action,
            implication="Logged methodology decision",
            recommended_action="",
            relevant_pages=["10_Report_Export"],
            resolved=True,
            resolved_by=action,
            resolved_on_page=page,
            resolution_details=enriched,
            auto_generated=True,
            metadata={"audit_only": True} if is_audit_only else {},
        ))
    except Exception:
        pass  # Never break methodology logging if ledger has issues

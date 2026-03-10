"""
Page 01: Upload and Data Audit
Project-based data management with intelligent merging.
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime

from utils.session_state import (
    init_session_state, set_data, get_data, DataConfig, reset_data_dependent_state,
    TaskTypeDetection, CohortStructureDetection
)
from utils.datasets import get_builtin_datasets
from utils.reconcile import reconcile_target_features
from utils.state_reconcile import reconcile_state_with_df
from utils.storyline import render_breadcrumb, render_page_navigation
from utils.session_projects import get_project_manager
from utils.dataset_db import detect_common_columns, suggest_join_keys, execute_merge
from utils.column_utils import make_unique_columns
from utils.theme import inject_custom_css, render_guidance, render_sidebar_workflow
from utils.table_export import table
from data_processor import (
    load_tabular_data, get_numeric_columns, get_selectable_columns,
    detect_file_type
)
from ml.triage import detect_task_type, detect_cohort_structure
from ml.eda_recommender import compute_dataset_signals

logger = logging.getLogger(__name__)


# =============================================================================
# HELPER: Visual Schema Diagram
# =============================================================================
def render_schema_diagram(dataframes: Dict[str, pd.DataFrame], common_cols: Dict[str, List[str]]):
    """
    Render a visual schema diagram showing datasets and their relationships.
    Similar to Microsoft Access relationship view.
    """
    st.markdown("#### Data Schema")
    st.caption("Visual overview of your datasets and how they can connect")
    
    # Create columns for each dataset (max 4 per row)
    n_datasets = len(dataframes)
    cols_per_row = min(n_datasets, 3)
    
    dataset_names = list(dataframes.keys())
    
    # Show datasets as "cards"
    cols = st.columns(cols_per_row)
    
    for i, (name, df) in enumerate(dataframes.items()):
        col_idx = i % cols_per_row
        
        with cols[col_idx]:
            # Dataset "card"
            st.markdown(f"""
            <div style="border: 2px solid #667eea; border-radius: 8px; padding: 10px; margin: 5px 0; background-color: #f8f9fa;">
                <div style="font-weight: bold; color: #667eea; border-bottom: 1px solid #ddd; padding-bottom: 5px; margin-bottom: 8px;">
                    {name}
                </div>
                <div style="font-size: 0.85em; color: #666;">
                    {df.shape[0]:,} rows × {df.shape[1]} columns
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show columns (highlight common ones)
            common_col_names = set(common_cols.keys())
            
            # Show first few columns
            max_cols_shown = 8
            shown_cols = list(df.columns)[:max_cols_shown]
            
            for col in shown_cols:
                if col in common_col_names:
                    # Highlight columns that appear in multiple datasets
                    st.markdown(f"<span style='color: #1a73e8; font-weight: bold;'>🔗 {col}</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span style='color: #666; font-size: 0.9em;'>○ {col}</span>", unsafe_allow_html=True)
            
            if len(df.columns) > max_cols_shown:
                st.caption(f"... +{len(df.columns) - max_cols_shown} more columns")
    
    # Show relationship summary
    if common_cols:
        st.markdown("---")
        st.markdown("**Potential Connections (🔗 = shared columns):**")
        for col, ds_list in common_cols.items():
            if len(ds_list) >= 2:
                st.markdown(f"• `{col}` connects: {' ↔ '.join(ds_list)}")
    else:
        st.warning("No shared columns detected. Consider transposing one of your datasets.")



# Initialize session state
init_session_state()

# Initialize session-only project manager (no shared disk state)
db = get_project_manager()

st.set_page_config(
    page_title="Upload & Audit",
    page_icon=None,
    layout="wide"
)
inject_custom_css()
render_sidebar_workflow(current_page="01_Upload_and_Audit")

st.title("Data Upload & Project Management")
render_breadcrumb("01_Upload_and_Audit")
render_page_navigation("01_Upload_and_Audit")

# Progress indicator

# ============================================================================
# DATA PERSISTENCE INFO & MANAGEMENT (Sidebar)
# ============================================================================
with st.sidebar:
    st.subheader("Data Management")
    
    # Get database stats
    db_stats = db.get_database_stats()
    
    # Show current state
    st.caption(f"Projects: {db_stats['n_projects']} | Datasets: {db_stats['n_datasets']}")
    
    with st.expander("About Your Data", expanded=False):
        render_guidance(
            "<strong>Your data stays private.</strong> Everything lives in your browser session only — "
            "nothing is saved to disk and no other user can see your projects or data.<br/><br/>"
            "<strong>When you refresh or close the app:</strong> All projects, data, and results are cleared. "
            "You'll need to re-upload your files.<br/><br/>"
            "<strong>Tip:</strong> Complete your analysis in one session, and use <strong>Report Export</strong> to save your results.",
            icon="🔒"
        )
    
    # Quick actions
    st.markdown("**Quick Actions:**")
    
    # Check current state
    has_working_table = st.session_state.get('working_table') is not None
    has_analysis_config = st.session_state.get('data_config') is not None and st.session_state.get('data_config').target_col is not None
    
    # Modify Data button - allows going back to change data setup
    if has_working_table or has_analysis_config:
        if st.button("Modify Data Setup", type="secondary", key="modify_data", help="Go back to change your data or merge settings"):
            # Clear analysis config but keep working table
            st.session_state.data_config = DataConfig()
            st.session_state.task_mode = None
            st.session_state.task_type_detection = TaskTypeDetection()
            # Clear trained models and preprocessing
            st.session_state.pop('trained_models', None)
            st.session_state.pop('model_results', None)
            st.session_state.pop('preprocessing_pipeline', None)
            st.session_state.pop('X_train', None)
            st.info("Analysis config cleared. You can now modify your data setup.")
            st.rerun()
        
        if st.button("Change Merge Setup", type="secondary", key="change_merge", help="Go back to re-merge your datasets"):
            st.session_state.pop('working_table', None)
            st.session_state.pop('merge_preview', None)
            st.session_state.pop('merge_config', None)
            st.session_state.pop('merge_steps', None)
            st.session_state.pop('last_merge_columns', None)
            st.session_state.pop('transposed_for_merge', None)
            st.session_state.data_config = DataConfig()
            st.session_state.task_mode = None
            reset_data_dependent_state()
            st.info("Merge cleared. You can now re-configure your data merge.")
            st.rerun()
    
    st.divider()
    
    # Data management options
    with st.expander("Reset Options", expanded=False):
        st.warning("These actions cannot be undone!")
        
        if not st.session_state.get('confirm_clear_session'):
            if st.button("Clear Current Session", type="secondary", key="clear_session", help="Clears uploaded data but keeps project structure"):
                st.session_state['confirm_clear_session'] = True
        
        if st.session_state.get('confirm_clear_session'):
            st.error("Are you sure? This will clear all uploaded data from this session (project structure is kept).")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Yes, Clear Session", type="primary", key="confirm_clear_yes"):
                    st.session_state.pop('datasets_registry', None)
                    st.session_state.pop('working_table', None)
                    st.session_state.pop('merge_steps', None)
                    st.session_state.pop('transposed_for_merge', None)
                    st.session_state.pop('confirm_clear_session', None)
                    reset_data_dependent_state()
                    st.success("Session cleared! Re-upload your files to continue.")
                    st.rerun()
            with c2:
                if st.button("Cancel", type="secondary", key="confirm_clear_no"):
                    st.session_state.pop('confirm_clear_session', None)
                    st.rerun()
        


# ============================================================================
# IMPLICIT PROJECT (auto-created per session, no UI)
# ============================================================================
active_project = db.get_active_project()
if not active_project:
    db.create_project("Session", "Auto-created session workspace")
    active_project = db.get_active_project()

# ============================================================================
# SECTION 2: UPLOAD FILES TO PROJECT
# ============================================================================
st.markdown("---")
st.header("Step 1: Upload Your Data")
st.caption("Upload one or more data files. If you have multiple files, you can merge them in the next step.")

# Initialize datasets registry for this project
if 'datasets_registry' not in st.session_state:
    st.session_state.datasets_registry = {}

# Show existing datasets in project
project_datasets = db.get_project_datasets(active_project['id'])

if project_datasets:
    st.subheader("Datasets in This Project")
    
    dataset_summary = []
    for d in project_datasets:
        in_memory = d['id'] in st.session_state.datasets_registry
        dataset_summary.append({
            'Name': d['name'],
            'Filename': d['filename'],
            'Rows': f"{d['shape_rows']:,}",
            'Columns': d['shape_cols'],
            'Status': "Ready" if in_memory else "Missing"
        })
    
    table(pd.DataFrame(dataset_summary), width="stretch", hide_index=True)
    
    # Dataset actions
    with st.expander("Manage Datasets"):
        delete_dataset = st.selectbox(
            "Select dataset to delete",
            options=[''] + [d['name'] for d in project_datasets],
            key="delete_dataset_select"
        )
        if delete_dataset and st.button("Delete Selected Dataset", type="secondary"):
            for d in project_datasets:
                if d['name'] == delete_dataset:
                    db.delete_dataset(d['id'])
                    if d['id'] in st.session_state.datasets_registry:
                        del st.session_state.datasets_registry[d['id']]
                    st.success(f"Deleted '{delete_dataset}'")
                    st.rerun()

# File upload
st.subheader("Upload New Files")

uploaded_files = st.file_uploader(
    "Upload data files (CSV, Excel, Parquet, TSV)",
    type=['csv', 'xlsx', 'xls', 'parquet', 'tsv', 'txt'],
    accept_multiple_files=True,
    key="file_uploader"
)

MAX_FILE_SIZE_MB = 50

if uploaded_files:
    for file_idx, uploaded_file in enumerate(uploaded_files):
        file_type = detect_file_type(uploaded_file.name)
        file_key = f"{uploaded_file.name.replace('.', '_').replace(' ', '_')}_{file_idx}"
        
        with st.expander(f"Configure: {uploaded_file.name}", expanded=True):
            try:
                # Large file warning
                file_size_mb = uploaded_file.size / (1024 * 1024)
                if file_size_mb > MAX_FILE_SIZE_MB:
                    st.warning(
                        f"**{uploaded_file.name}** is {file_size_mb:.1f} MB (limit: {MAX_FILE_SIZE_MB} MB). "
                        "Large files may be slow to load."
                    )
                    load_anyway = st.checkbox("Load anyway", key=f"load_large_{file_key}")
                    if not load_anyway:
                        continue

                # Excel sheet selector (for multi-sheet files)
                excel_sheet_choice = 0
                if file_type == 'excel':
                    uploaded_file.seek(0)
                    try:
                        xl = pd.ExcelFile(uploaded_file)
                        sheet_names = xl.sheet_names
                        uploaded_file.seek(0)
                        if len(sheet_names) > 1:
                            excel_sheet_choice = st.selectbox(
                                "Excel sheet to load",
                                options=range(len(sheet_names)),
                                format_func=lambda i, sn=sheet_names: sn[i],
                                key=f"excel_sheet_{file_key}",
                                help="Select which sheet to load from this Excel file"
                            )
                        else:
                            excel_sheet_choice = 0
                    except Exception:
                        excel_sheet_choice = 0
                    uploaded_file.seek(0)
                
                # Per-file transpose option
                transpose_this_file = st.checkbox(
                    "Transpose this file (rows ↔ columns)",
                    value=False,
                    key=f"transpose_{file_key}",
                    help="Use this if your features are in rows instead of columns"
                )
                
                # Load preview with transpose setting
                with st.spinner(f"Loading {uploaded_file.name}..."):
                    df_preview = load_tabular_data(
                        uploaded_file,
                        filename=uploaded_file.name,
                        transpose=transpose_this_file,
                        excel_sheet=excel_sheet_choice if file_type == 'excel' else 0
                    )
                
                # Reset file position for later
                uploaded_file.seek(0)
                
                # Ensure column names are strings for merging compatibility
                df_preview.columns = [str(c) for c in df_preview.columns]
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    preview_rows = min(5, len(df_preview))
                    table(df_preview.head(5), width="stretch")
                    st.caption(f"Shape: {df_preview.shape[0]:,} rows × {df_preview.shape[1]} columns. Showing first {preview_rows} of {len(df_preview):,} rows.")
                    if transpose_this_file:
                        st.info("Preview shows transposed data (original rows are now columns)")
                
                with col2:
                    dataset_name = st.text_input(
                        "Dataset Name",
                        value=uploaded_file.name.rsplit('.', 1)[0],
                        key=f"name_{file_key}"
                    )
                    
                    # Check if dataset with same name already exists
                    existing_names = [d['name'] for d in project_datasets] if project_datasets else []
                    name_exists = dataset_name in existing_names
                    
                    if name_exists:
                        st.warning(f"A dataset named '{dataset_name}' already exists in this project.")
                        replace_existing = st.checkbox(
                            f"Replace existing '{dataset_name}'", 
                            key=f"replace_{file_key}"
                        )
                    else:
                        replace_existing = False
                    
                    if st.button(f"Add to Project", key=f"add_{file_key}", type="primary"):
                        # Delete existing if replacing
                        if name_exists and replace_existing:
                            for d in project_datasets:
                                if d['name'] == dataset_name:
                                    db.delete_dataset(d['id'])
                                    if d['id'] in st.session_state.datasets_registry:
                                        del st.session_state.datasets_registry[d['id']]
                                    break
                        elif name_exists and not replace_existing:
                            st.error(f"Please check 'Replace existing' or change the dataset name.")
                            st.stop()
                        
                        # Reload to ensure fresh data
                        uploaded_file.seek(0)
                        sheet_param = excel_sheet_choice if file_type == 'excel' else 0
                        with st.spinner(f"Adding {dataset_name} to project..."):
                            df = load_tabular_data(
                                uploaded_file,
                                filename=uploaded_file.name,
                                transpose=transpose_this_file,
                                excel_sheet=sheet_param
                            )
                        
                        # Ensure column names are strings for merge compatibility
                        df.columns = [str(c) for c in df.columns]
                        
                        # Get column types
                        col_types = {str(col): str(df[col].dtype) for col in df.columns}
                        
                        # Add to database
                        dataset_id = db.add_dataset(
                            project_id=active_project['id'],
                            name=dataset_name,
                            filename=uploaded_file.name,
                            file_type=file_type,
                            shape_rows=df.shape[0],
                            shape_cols=df.shape[1],
                            columns=[str(c) for c in df.columns],
                            column_types=col_types,
                            is_transposed=transpose_this_file
                        )
                        
                        # Store DataFrame in registry
                        st.session_state.datasets_registry[dataset_id] = df
                        
                        st.success(f"Added '{dataset_name}' to project!")
                        st.rerun()
                        
            except Exception as e:
                st.error(f"Error loading file: {e}")
                logger.exception(e)

# Built-in datasets option
with st.expander("Or Use Built-in Dataset"):
    builtin_options = [''] + list(get_builtin_datasets().keys())
    selected_builtin = st.selectbox("Built-in Dataset", builtin_options, key="builtin_select")
    
    if selected_builtin and st.button("Add Built-in Dataset", key="add_builtin"):
        generator = get_builtin_datasets()[selected_builtin]
        df_builtin = generator(random_state=st.session_state.get('random_seed', 42))
        
        col_types = {col: str(df_builtin[col].dtype) for col in df_builtin.columns}
        
        dataset_id = db.add_dataset(
            project_id=active_project['id'],
            name=selected_builtin,
            filename=f"builtin_{selected_builtin}",
            file_type='builtin',
            shape_rows=df_builtin.shape[0],
            shape_cols=df_builtin.shape[1],
            columns=list(df_builtin.columns),
            column_types=col_types,
            is_transposed=False
        )
        
        st.session_state.datasets_registry[dataset_id] = df_builtin
        st.success(f"Added '{selected_builtin}' to project!")
        st.rerun()

# Refresh project datasets
project_datasets = db.get_project_datasets(active_project['id'])

if not project_datasets:
    st.info("Upload at least one dataset to continue.")
    st.stop()

# ============================================================================
# SECTION 3: MERGE DATASETS (if multiple)
# ============================================================================
st.markdown("---")

if len(project_datasets) > 1:
    st.header("Step 2: Combine Your Datasets")
    
    # Check how many datasets are ready in memory
    datasets_ready = sum(1 for d in project_datasets if d['id'] in st.session_state.datasets_registry)
    datasets_total = len(project_datasets)
    
    if datasets_ready < datasets_total:
        st.error(f"""
        **Cannot proceed: {datasets_total - datasets_ready} of {datasets_total} datasets not loaded.**
        
        Please scroll up to Step 2 and either:
        - Re-upload the missing files, or  
        - Clear the old dataset records and upload fresh files
        """)
        st.stop()
    
    # Load all dataframes (ensure string column names for merge compatibility)
    dataframes = {}
    for d in project_datasets:
        if d['id'] in st.session_state.datasets_registry:
            df_temp = st.session_state.datasets_registry[d['id']].copy()
            df_temp.columns = [str(c) for c in df_temp.columns]
            dataframes[d['name']] = df_temp
    
    # -------------------------------------------------------------------------
    # PLAIN LANGUAGE EXPLANATION
    # -------------------------------------------------------------------------
    st.info("""
    **You have multiple files uploaded.** Before analysis, you need to combine them into one table.
    
    Think of it like this: if you have patient demographics in one file and their lab results in another, 
    you need to connect them so each patient's info is on the same row as their results.
    """)
    
    # -------------------------------------------------------------------------
    # VISUAL SCHEMA DIAGRAM
    # -------------------------------------------------------------------------
    # Detect common columns for the visual
    common_cols = detect_common_columns(project_datasets)
    
    render_schema_diagram(dataframes, common_cols)
    
    # -------------------------------------------------------------------------
    # DATA ORIENTATION CHECK
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Check Your Data Structure")
    st.caption("""
    **Important:** For analysis, your data should usually have:
    - **Rows** = observations (patients, samples, records) 
    - **Columns** = variables/features to analyze
    
    If your data is structured the other way around, you can transpose it here.
    """)
    
    # Track transposed dataframes
    if 'transposed_for_merge' not in st.session_state:
        st.session_state.transposed_for_merge = {}
    
    orientation_issues = []
    
    for name, df in dataframes.items():
        with st.expander(f"**{name}** — {df.shape[0]:,} rows × {df.shape[1]} columns", expanded=False):
            # Show preview
            table(df.head(3), width="stretch")
            
            # Check orientation
            cols_much_larger = df.shape[1] > df.shape[0] * 2 and df.shape[1] > 10
            rows_much_larger = df.shape[0] > df.shape[1] * 10 and df.shape[0] > 100
            
            if cols_much_larger:
                st.warning(f"""
                **Possible issue:** This dataset has {df.shape[1]} columns but only {df.shape[0]} rows.
                
                If your {df.shape[1]} columns are actually observations (like patients or samples) 
                and your {df.shape[0]} rows are features, you should transpose this data.
                """)
                orientation_issues.append(name)
            
            # Transpose option for this dataset
            transpose_key = f"transpose_merge_{name}"
            currently_transposed = st.session_state.transposed_for_merge.get(name, False)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                if currently_transposed:
                    st.success(f"This dataset will be transposed (rows ↔ columns) before merging.")
                else:
                    st.caption("Data will be used as-is.")
            
            with col2:
                btn_label = "Undo Transpose" if currently_transposed else "Transpose"
                if st.button(btn_label, type="secondary", key=f"btn_transpose_{name}"):
                    st.session_state.transposed_for_merge[name] = not currently_transposed
                    st.rerun()
            
            # Show preview of transposed version and download option
            if currently_transposed:
                st.markdown("**Preview after transpose:**")
                transposed_df = df.T.reset_index()
                transposed_df.columns = ['index'] + [f"col_{i}" for i in range(len(transposed_df.columns)-1)]
                table(transposed_df.head(5), width="stretch")
                st.caption(f"After transpose: {transposed_df.shape[0]:,} rows × {transposed_df.shape[1]} columns")
                
                # Download transposed data
                csv_data = transposed_df.to_csv(index=False)
                st.download_button(
                    label=f"Download Transposed '{name}' as CSV",
                    data=csv_data,
                    file_name=f"{name}_transposed.csv",
                    mime="text/csv",
                    key=f"download_transposed_{name}"
                )
            
            # Always offer download of original/current version
            st.markdown("---")
            original_csv = df.to_csv(index=False)
            st.download_button(
                label=f"Download '{name}' as CSV (current version)",
                data=original_csv,
                file_name=f"{name}.csv",
                mime="text/csv",
                key=f"download_original_{name}"
            )
    
    # Apply transpositions to dataframes for merging
    for name in list(dataframes.keys()):
        if st.session_state.transposed_for_merge.get(name, False):
            original_df = dataframes[name]
            # Transpose and use first row as header if it looks like labels
            transposed = original_df.T.reset_index()
            # Try to use meaningful column names (deduplicate if first row has duplicates)
            if len(transposed.columns) > 0 and len(transposed) > 0:
                raw_cols = [str(c) for c in transposed.iloc[0]]
                transposed.columns = make_unique_columns(raw_cols)
                transposed = transposed.iloc[1:].reset_index(drop=True)
            transposed.columns = make_unique_columns(transposed.columns)
            dataframes[name] = transposed
    
    # Re-detect common columns after transposition
    # Build project_datasets equivalent from dataframes
    updated_project_datasets = [
        {'name': name, 'columns': list(df.columns)} 
        for name, df in dataframes.items()
    ]
    common_cols = detect_common_columns(updated_project_datasets)
    
    if orientation_issues:
        st.info(f"""
        **Tip:** {len(orientation_issues)} dataset(s) may have unusual structure. 
        Review them above and use the Transpose button if needed.
        """)
    
    st.markdown("---")
    
    # -------------------------------------------------------------------------
    # MERGE OPTIONS
    # -------------------------------------------------------------------------
    st.subheader("How would you like to combine them?")
    
    # Use the already-computed common_cols from transposed data
    # Re-compute suggestions based on potentially transposed data
    suggestions = suggest_join_keys(updated_project_datasets)
    
    # Initialize merge mode
    if 'merge_mode' not in st.session_state:
        st.session_state.merge_mode = 'guided'
    
    # Option 1: Skip merge (just use first dataset)
    # Option 2: Guided merge (we help them)
    # Option 3: Advanced merge (full control)
    
    merge_choice = st.radio(
        "Choose an option:",
        [
            "Combine datasets using a shared column (recommended)",
            "Just use one dataset (ignore the others)",
            "I know what I'm doing (advanced options)"
        ],
        key="merge_choice_radio",
        label_visibility="collapsed",
        help="Combine: merge on a shared ID column. Just use one: pick a single dataset. Advanced: full control over join type and columns."
    )
    
    # -------------------------------------------------------------------------
    # OPTION 1: GUIDED MERGE (Recommended)
    # -------------------------------------------------------------------------
    if "Combine datasets" in merge_choice:
        st.markdown("#### Connect Your Datasets")
        
        if not common_cols:
            st.warning("""
            **No matching columns found between your datasets.**
            
            To combine datasets, they need at least one column in common (like "patient_id", "date", or "record_number").
            
            Check that your files have a shared identifier column with the same name.
            """)
        else:
            st.success(f"**Good news!** We found {len(common_cols)} column(s) that appear in multiple datasets.")
            
            # Show the common columns in plain language
            st.markdown("**Shared columns that can connect your data:**")
            for col, ds_list in common_cols.items():
                st.write(f"• `{col}` - found in: {', '.join(ds_list)}")
            
            st.markdown("---")
            
            # Simple merge interface
            st.markdown("#### Set Up Your Merge")
            
            dataset_names = list(dataframes.keys())
            
            # For 2 datasets, make it simple
            if len(dataset_names) == 2:
                st.markdown(f"""
                You have **{dataset_names[0]}** and **{dataset_names[1]}**.
                
                Select the column that links them together:
                """)
                
                # Find columns that exist in both
                cols_in_both = [col for col, ds_list in common_cols.items() if len(ds_list) >= 2]
                
                if cols_in_both:
                    linking_column = st.selectbox(
                        "Linking column:",
                        options=cols_in_both,
                        key="simple_link_col",
                        help="This column should contain values that match between your two datasets (like IDs or dates)"
                    )
                    
                    # Explain what will happen
                    df1, df2 = dataframes[dataset_names[0]], dataframes[dataset_names[1]]
                    
                    st.markdown("**What will happen:**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"{dataset_names[0]}", f"{df1.shape[0]:,} rows")
                    with col2:
                        st.metric(f"{dataset_names[1]}", f"{df2.shape[0]:,} rows")
                    with col3:
                        # Estimate result size
                        matching_values = set(df1[linking_column].dropna().unique()) & set(df2[linking_column].dropna().unique())
                        st.metric("Matching values", f"{len(matching_values):,}")
                    
                    st.caption(f"Rows will be matched where `{linking_column}` values are the same in both datasets.")
                    
                    # Preview button
                    if st.button("Preview Combined Data", key="preview_merge"):
                        try:
                            with st.spinner("Merging datasets..."):
                                preview_df = pd.merge(df1, df2, on=linking_column, how='inner', suffixes=('', '_2'))
                            st.session_state.merge_preview = preview_df
                            st.session_state.merge_config = {
                                'left': dataset_names[0],
                                'right': dataset_names[1],
                                'on': linking_column,
                                'how': 'inner'
                            }
                        except Exception as e:
                            st.error(f"Preview failed: {e}")
                    
                    # Show preview if available
                    if 'merge_preview' in st.session_state and st.session_state.merge_preview is not None:
                        st.markdown("#### Preview of Combined Data")
                        preview_df = st.session_state.merge_preview
                        table(preview_df.head(10), width="stretch")
                        st.caption(f"Combined result: {preview_df.shape[0]:,} rows × {preview_df.shape[1]} columns")
                        
                        if preview_df.shape[0] == 0:
                            st.error("""
                            **No matching rows found!** 
                            
                            This means the values in your linking column don't match between the two files.
                            Check that the data in this column is formatted the same way in both files.
                            """)
                        else:
                            st.success("This looks good! Click below to use this combined data for your analysis.")
                            
                            if st.button("Use This Combined Data", type="primary", key="confirm_merge"):
                                st.session_state.working_table = preview_df
                                st.session_state.last_merge_columns = list(preview_df.columns)
                                
                                # Save merge config
                                merge_cfg = st.session_state.merge_config
                                db.save_merge_config(
                                    project_id=active_project['id'],
                                    name=f"Merge_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                    merge_steps=[{
                                        'left': merge_cfg['left'],
                                        'right': merge_cfg['right'],
                                        'left_on': merge_cfg['on'],
                                        'right_on': merge_cfg['on'],
                                        'how': merge_cfg['how']
                                    }],
                                    result_shape=(preview_df.shape[0], preview_df.shape[1]),
                                    result_columns=list(preview_df.columns),
                                    set_as_working=True
                                )
                                
                                set_data(preview_df)
                                st.session_state.pop('merge_preview', None)
                                st.session_state.pop('merge_config', None)
                                st.success("Combined data is ready for analysis!")
                                st.rerun()
                else:
                    st.warning("No columns are shared between both datasets. Check your column names.")
            
            else:
                # More than 2 datasets - comprehensive chaining workflow
                st.markdown("""
                ### Multi-Dataset Merge Workflow
                
                You have **{} datasets**. You'll combine them step by step, creating intermediate results.
                
                **Common scenario:** You might have:
                - A main data matrix (e.g., gene expression, features)
                - Outcome/label data (e.g., classifications, target variable)  
                - Metadata/annotations (e.g., row/column names, descriptions)
                """.format(len(dataset_names)))
                
                # Show current merge progress
                if 'multi_merge_result' in st.session_state and st.session_state.multi_merge_result is not None:
                    st.success(f"**Current merged result:** {st.session_state.multi_merge_result.shape[0]:,} rows × {st.session_state.multi_merge_result.shape[1]:,} columns")
                    
                    # Option to continue merging or finish
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Continue - Add Another Dataset", key="continue_merge"):
                            pass  # Continue with the merge form below
                    with col2:
                        if st.button("Finish - Use Current Result", type="primary", key="finish_multi_merge"):
                            st.session_state.working_table = st.session_state.multi_merge_result
                            st.session_state.last_merge_columns = list(st.session_state.multi_merge_result.columns)
                            set_data(st.session_state.multi_merge_result)
                            st.session_state.pop('multi_merge_result', None)
                            st.success("Merge complete!")
                            st.rerun()
                    
                    st.divider()
                
                # Step-by-step merge builder
                st.markdown("#### Add Merge Step")
                
                # Determine what can be merged
                if 'multi_merge_result' in st.session_state and st.session_state.multi_merge_result is not None:
                    left_options = ["(Previous Merge Result)"] + dataset_names
                    default_left = "(Previous Merge Result)"
                else:
                    left_options = dataset_names
                    default_left = dataset_names[0] if dataset_names else None
                
                col1, col2 = st.columns(2)
                with col1:
                    left_choice = st.selectbox(
                        "Left (base) dataset:", 
                        left_options, 
                        key="multi_left_choice"
                    )
                with col2:
                    right_options = [d for d in dataset_names if d != left_choice or left_choice == "(Previous Merge Result)"]
                    right_choice = st.selectbox(
                        "Right (to add) dataset:", 
                        right_options, 
                        key="multi_right_choice"
                    )
                
                # Get the dataframes
                if left_choice == "(Previous Merge Result)":
                    left_df = st.session_state.multi_merge_result
                    left_cols = list(left_df.columns)
                else:
                    left_df = dataframes[left_choice]
                    left_cols = list(left_df.columns)
                
                right_df = dataframes[right_choice]
                right_cols = list(right_df.columns)
                
                # Find shared columns
                shared_cols = list(set(left_cols) & set(right_cols))
                
                st.markdown("**How to connect these datasets:**")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"Left: {left_choice}")
                    left_on = st.selectbox(
                        "Left join column:", 
                        [''] + left_cols,
                        key="multi_left_on"
                    )
                with col2:
                    st.caption(f"Right: {right_choice}")
                    # Default to same column if shared
                    default_right_idx = 0
                    if left_on and left_on in right_cols:
                        default_right_idx = right_cols.index(left_on) + 1
                    right_on = st.selectbox(
                        "Right join column:", 
                        [''] + right_cols,
                        index=default_right_idx,
                        key="multi_right_on"
                    )
                
                if shared_cols:
                    st.info(f"**Shared columns detected:** {', '.join(shared_cols[:5])}{'...' if len(shared_cols) > 5 else ''}")
                else:
                    st.warning("No shared column names. Make sure you select matching ID/key columns.")
                
                # Join type
                join_type = st.radio(
                    "How to handle non-matching rows:",
                    ["Keep only matching rows (inner join)", 
                     "Keep all from left, match from right (left join)",
                     "Keep all rows from both (outer join)"],
                    key="multi_join_type"
                )
                join_how = {'inner': 'inner', 'left': 'left', 'outer': 'outer'}
                how = 'inner' if 'inner' in join_type else ('left' if 'left' in join_type else 'outer')
                
                # Preview and execute
                if left_on and right_on:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("Preview Merge", key="preview_multi_merge"):
                            try:
                                preview = pd.merge(
                                    left_df, right_df,
                                    left_on=left_on, right_on=right_on,
                                    how=how, suffixes=('', '_2')
                                )
                                st.session_state.multi_merge_preview = preview
                            except Exception as e:
                                st.error(f"Preview failed: {e}")
                    
                    with col2:
                        if st.button("Execute Merge", type="primary", key="execute_multi_merge"):
                            try:
                                merged = pd.merge(
                                    left_df, right_df,
                                    left_on=left_on, right_on=right_on,
                                    how=how, suffixes=('', '_2')
                                )
                                st.session_state.multi_merge_result = merged
                                st.session_state.pop('multi_merge_preview', None)
                                st.success(f"Merged! Result: {merged.shape[0]:,} rows × {merged.shape[1]:,} columns")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Merge failed: {e}")
                    
                    # Show preview if available
                    if 'multi_merge_preview' in st.session_state and st.session_state.multi_merge_preview is not None:
                        st.markdown("**Preview:**")
                        preview = st.session_state.multi_merge_preview
                        table(preview.head(10), width="stretch")
                        st.caption(f"Result would be: {preview.shape[0]:,} rows × {preview.shape[1]:,} columns")
                        
                        if preview.shape[0] == 0:
                            st.error("No matching rows! Check that your join columns have matching values.")
                else:
                    st.caption("Select join columns to enable merge.")
    
    # -------------------------------------------------------------------------
    # OPTION 2: USE SINGLE DATASET
    # -------------------------------------------------------------------------
    elif "Just use one" in merge_choice:
        st.markdown("#### Select Which Dataset to Use")
        st.caption("The other datasets will be ignored for this analysis.")
        
        dataset_names = list(dataframes.keys())
        selected_single = st.selectbox(
            "Dataset to use:",
            options=dataset_names,
            key="single_dataset_select"
        )
        
        selected_df = dataframes[selected_single]
        table(selected_df.head(5), width="stretch")
        st.caption(f"Shape: {selected_df.shape[0]:,} rows × {selected_df.shape[1]} columns")
        
        if st.button("Use This Dataset", type="primary", key="use_single"):
            st.session_state.working_table = selected_df
            set_data(selected_df)
            st.success(f"Using '{selected_single}' for analysis.")
            st.rerun()
    
    # -------------------------------------------------------------------------
    # OPTION 3: ADVANCED MERGE
    # -------------------------------------------------------------------------
    else:
        st.markdown("#### Advanced Merge Options")
        st.caption("For users familiar with database joins.")
        
        # Initialize merge steps in session state
        if 'merge_steps' not in st.session_state:
            st.session_state.merge_steps = []
        
        dataset_names = list(dataframes.keys())
        
        with st.expander("Join Type Reference", expanded=False):
            st.markdown("""
            - **Inner Join**: Only rows where the key exists in BOTH tables
            - **Left Join**: All rows from left table, matched rows from right (nulls if no match)
            - **Right Join**: All rows from right table, matched rows from left (nulls if no match)
            - **Outer Join**: All rows from both tables (nulls where no match)
            """)
        
        # Add merge step form
        with st.form("merge_step_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                left_options = ['(Previous Result)'] + dataset_names if st.session_state.merge_steps else dataset_names
                left_table = st.selectbox("Left Table", left_options, key="merge_left")
            
            with col2:
                right_table = st.selectbox("Right Table", dataset_names, key="merge_right")
            
            with col3:
                join_type = st.selectbox("Join Type", ['inner', 'left', 'right', 'outer'], key="merge_how")
            
            # Determine available columns for join
            if left_table == '(Previous Result)' and st.session_state.merge_steps:
                left_cols = [str(c) for c in st.session_state.get('last_merge_columns', [])]
            else:
                left_df = dataframes.get(left_table)
                left_cols = [str(c) for c in left_df.columns] if left_df is not None else []
            
            right_df = dataframes.get(right_table)
            right_cols = [str(c) for c in right_df.columns] if right_df is not None else []
            
            col1, col2 = st.columns(2)
            with col1:
                left_on = st.selectbox("Left Join Column", [''] + left_cols, key="merge_left_on")
            with col2:
                right_on = st.selectbox("Right Join Column", [''] + right_cols, key="merge_right_on")
            
            submitted = st.form_submit_button("Add Merge Step")
            
            if submitted:
                if not left_on or not right_on:
                    st.error("Please select both a Left Join Column and Right Join Column before adding a merge step.")
                elif left_table == right_table:
                    st.error("Left and Right tables must be different.")
                else:
                    step = {
                        'left': 'result' if left_table == '(Previous Result)' else left_table,
                        'right': right_table,
                        'left_on': left_on,
                        'right_on': right_on,
                        'how': join_type
                    }
                    st.session_state.merge_steps.append(step)
                    # Update last_merge_columns for next step
                    if st.session_state.merge_steps:
                        try:
                            # Preview what columns will result
                            temp_result = execute_merge(dataframes, st.session_state.merge_steps)
                            st.session_state.last_merge_columns = list(temp_result.columns)
                        except Exception as e:
                            logger.debug(f"Could not preview merge columns: {e}")
                    st.rerun()
        
        # Show current merge steps
        if st.session_state.merge_steps:
            st.markdown("**Merge Pipeline:**")
            for i, step in enumerate(st.session_state.merge_steps):
                left_name = "Previous Result" if step['left'] == 'result' else step['left']
                st.code(f"{i+1}. {left_name} {step['how'].upper()} JOIN {step['right']} ON {step['left_on']} = {step['right_on']}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Clear All", type="secondary"):
                    st.session_state.merge_steps = []
                    st.session_state.pop('working_table', None)
                    st.session_state.pop('last_merge_columns', None)
                    st.rerun()
            
            with col2:
                if st.button("Execute Merge", type="primary"):
                    try:
                        merged_df = execute_merge(dataframes, st.session_state.merge_steps)
                        st.session_state.working_table = merged_df
                        st.session_state.last_merge_columns = list(merged_df.columns)
                        
                        db.save_merge_config(
                            project_id=active_project['id'],
                            name=f"Merge_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            merge_steps=st.session_state.merge_steps,
                            result_shape=(merged_df.shape[0], merged_df.shape[1]),
                            result_columns=list(merged_df.columns),
                            set_as_working=True
                        )
                        
                        set_data(merged_df)
                        st.success(f"Merge complete! Result: {merged_df.shape[0]:,} rows × {merged_df.shape[1]} columns")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Merge failed: {e}")
                        logger.exception(e)
    
    # -------------------------------------------------------------------------
    # SHOW WORKING TABLE IF EXISTS
    # -------------------------------------------------------------------------
    if 'working_table' in st.session_state and st.session_state.working_table is not None:
        st.markdown("---")
        st.subheader("Your Combined Data (Working Table)")
        working_df = st.session_state.working_table
        table(working_df.head(10), width="stretch")
        st.caption(f"Current shape: {working_df.shape[0]:,} rows × {working_df.shape[1]} columns")
        
        # -------------------------------------------------------------------------
        # FINAL ORIENTATION CHECK FOR ANALYSIS
        # -------------------------------------------------------------------------
        st.markdown("#### Prepare for Analysis")
        
        # Check if orientation might be wrong for analysis
        cols_much_larger = working_df.shape[1] > working_df.shape[0] * 2 and working_df.shape[1] > 10
        
        st.markdown("""
        **Before proceeding:** Make sure your data is oriented correctly for analysis.
        
        For most analyses:
        - Each **row** should be one observation (patient, sample, record, etc.)
        - Each **column** should be one variable/feature you want to analyze
        """)
        
        if cols_much_larger:
            st.warning(f"""
            **Your data has {working_df.shape[1]} columns but only {working_df.shape[0]} rows.**
            
            If your observations are actually in columns (e.g., each column is a patient/sample), 
            you should transpose the data so observations become rows.
            """)
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Transpose for Analysis (rows ↔ columns)", key="transpose_final"):
                transposed = working_df.T.reset_index()
                # Try to use first row as column names (deduplicate if duplicates)
                if len(transposed) > 0:
                    raw_cols = [str(c) for c in transposed.iloc[0]]
                    transposed.columns = make_unique_columns(raw_cols)
                    transposed = transposed.iloc[1:].reset_index(drop=True)
                transposed.columns = make_unique_columns(transposed.columns)
                
                st.session_state.working_table = transposed
                st.session_state.last_merge_columns = list(transposed.columns)
                set_data(transposed)
                st.success(f"Data transposed! New shape: {transposed.shape[0]:,} rows × {transposed.shape[1]} columns")
                st.rerun()
        
        with col2:
            # Download working table
            working_csv = working_df.to_csv(index=False)
            st.download_button(
                label="Download Working Table (CSV)",
                data=working_csv,
                file_name="working_table.csv",
                mime="text/csv",
                key="download_working_table"
            )
        
        with col3:
            if st.button("Start Over (Clear Working Table)", type="secondary", key="clear_working", help="Clear the combined data and return to merge setup"):
                st.session_state['confirm_clear_working'] = True
            
            if st.session_state.get('confirm_clear_working'):
                st.warning("Are you sure? This will clear your working table and you will need to re-merge datasets.")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Yes, Clear Working Table", type="secondary", key="confirm_clear_working_yes"):
                        st.session_state.pop('working_table', None)
                        st.session_state.pop('merge_preview', None)
                        st.session_state.pop('merge_config', None)
                        st.session_state.pop('merge_steps', None)
                        st.session_state.pop('last_merge_columns', None)
                        st.session_state.pop('transposed_for_merge', None)
                        st.session_state.pop('confirm_clear_working', None)
                        reset_data_dependent_state()
                        st.rerun()
                with c2:
                    if st.button("Cancel", key="confirm_clear_working_no"):
                        st.session_state.pop('confirm_clear_working', None)
                        st.rerun()
        
        st.success(f"**Data ready for analysis!** {working_df.shape[0]:,} observations × {working_df.shape[1]} variables")

else:
    # Single dataset - use it directly
    st.header("Step 2: Working Table")
    st.caption("With a single dataset, it becomes your working table directly.")
    
    single_dataset = project_datasets[0]
    
    if single_dataset['id'] in st.session_state.datasets_registry:
        working_df = st.session_state.datasets_registry[single_dataset['id']].copy()
        # Ensure string column names
        working_df.columns = [str(c) for c in working_df.columns]
        st.session_state.working_table = working_df
        set_data(working_df)
        
        st.success(f"**Working Table:** {single_dataset['name']}")
        table(working_df.head(10), width="stretch")
        st.caption(f"Shape: {working_df.shape[0]:,} rows × {working_df.shape[1]} columns")
    else:
        st.error("""
        **Dataset not in memory.** 
        
        Please scroll up to Step 2 and either:
        - Re-upload the file, or
        - Clear the old dataset record and upload a fresh file
        """)
        st.stop()

# Get working table
df = get_data()

if df is None:
    st.warning("Please complete the merge step or load a single dataset to continue.")
    st.stop()

if len(df) == 0 or len(df.columns) == 0:
    st.warning("Your working table is empty. Please upload data with at least one row and one column.")
    st.stop()

# ============================================================================
# SECTION 4: DATA AUDIT
# ============================================================================
st.markdown("---")
st.header("Step 3: Data Audit")

# Quick summary metrics at top
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Rows", f"{df.shape[0]:,}")
with col2:
    st.metric("Columns", df.shape[1])
with col3:
    missing_total = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    missing_pct = (missing_total / total_cells) * 100 if total_cells > 0 else 0
    st.metric("Missing Values", f"{missing_total:,}", f"{missing_pct:.1f}%")
with col4:
    numeric_count = len(get_numeric_columns(df))
    st.metric("Numeric Columns", numeric_count)
with col5:
    n_duplicates = df.duplicated().sum()
    st.metric("Duplicate Rows", f"{n_duplicates:,}")

audit_results = {}

# -------------------------------------------------------------------------
# CARDINALITY ANALYSIS
# -------------------------------------------------------------------------
with st.expander("Cardinality Analysis (Unique Values per Column)", expanded=True):
    st.caption("Helps identify potential ID columns, categorical variables, and constants.")
    
    cardinality_data = []
    for col in df.columns:
        n_unique = df[col].nunique()
        n_total = len(df)
        pct_unique = (n_unique / n_total) * 100 if n_total > 0 else 0
        
        # Classify cardinality
        if n_unique == 1:
            card_type = "Constant"
            card_flag = "⚠️"
        elif n_unique == 2:
            card_type = "Binary"
            card_flag = ""
        elif n_unique == n_total:
            card_type = "Unique (potential ID)"
            card_flag = "🔑"
        elif n_unique <= 10:
            card_type = "Low cardinality"
            card_flag = ""
        elif n_unique <= 50:
            card_type = "Moderate cardinality"
            card_flag = ""
        elif pct_unique > 90:
            card_type = "High cardinality (near-unique)"
            card_flag = ""
        else:
            card_type = "High cardinality"
            card_flag = ""
        
        cardinality_data.append({
            'Column': col,
            'Unique': n_unique,
            '% Unique': f"{pct_unique:.1f}%",
            'Type': card_type,
            'Flag': card_flag
        })
    
    card_df = pd.DataFrame(cardinality_data)
    table(card_df, width="stretch", hide_index=True)
    audit_results['cardinality'] = cardinality_data
    
    # Warnings
    constants = [c['Column'] for c in cardinality_data if c['Type'] == 'Constant']
    if constants:
        st.warning(f"**Constant columns detected:** {', '.join(constants)}. These provide no information and may be removed.")
    
    # Note: Potential ID columns are flagged in the cardinality table above

# -------------------------------------------------------------------------
# DATA TYPES & VALIDITY
# -------------------------------------------------------------------------
with st.expander("Data Types & Validity Checks", expanded=False):
    st.subheader("Column Types")
    
    dtype_data = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_nonnull = df[col].count()
        n_null = df[col].isnull().sum()
        n_unique = df[col].nunique()
        
        # Sample values
        sample_vals = df[col].dropna().head(3).tolist()
        sample_str = str(sample_vals)[:50] + "..." if len(str(sample_vals)) > 50 else str(sample_vals)
        
        # Validity check
        validity_issues = []
        if n_null > 0:
            validity_issues.append(f"{n_null} missing")
        if dtype == 'object':
            # Check for mixed types
            try:
                numeric_count = pd.to_numeric(df[col], errors='coerce').notna().sum()
                if 0 < numeric_count < n_nonnull:
                    validity_issues.append("mixed types")
            except Exception:
                pass
        
        dtype_data.append({
            'Column': col,
            'Type': dtype,
            'Non-null': n_nonnull,
            'Null': n_null,
            'Unique': n_unique,
            'Sample': sample_str,
            'Issues': ', '.join(validity_issues) if validity_issues else 'OK'
        })
    
    dtype_df = pd.DataFrame(dtype_data)
    table(dtype_df, width="stretch", hide_index=True)
    audit_results['dtypes'] = dtype_data

# -------------------------------------------------------------------------
# MISSING VALUES DETAIL
# -------------------------------------------------------------------------
with st.expander("Missing Values Detail", expanded=False):
    missing_counts = df.isnull().sum()
    n_rows = len(df)
    missing_pct = (missing_counts / n_rows) * 100 if n_rows > 0 else missing_counts * 0
    missing_df = pd.DataFrame({
        'Column': missing_counts.index,
        'Missing Count': missing_counts.values,
        'Missing %': [f"{p:.1f}%" for p in missing_pct.values]
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    
    if len(missing_df) > 0:
        table(missing_df, width="stretch", hide_index=True)
        audit_results['missing'] = missing_df.to_dict('records')
        
        # Missingness patterns
        high_missing = missing_df[missing_pct[missing_df['Column']].values > 50]
        if len(high_missing) > 0:
            st.warning(f"**{len(high_missing)} column(s) have >50% missing values.** Consider removing or imputing.")
    else:
        st.success("No missing values in any column!")
        audit_results['missing'] = []

# -------------------------------------------------------------------------
# DUPLICATES
# -------------------------------------------------------------------------
with st.expander("Duplicate Rows", expanded=False):
    if n_duplicates > 0:
        dup_pct = (n_duplicates / len(df) * 100) if len(df) > 0 else 0
        st.warning(f"Found **{n_duplicates:,}** duplicate rows ({dup_pct:.1f}% of data)")
        
        # Show sample duplicates
        dup_mask = df.duplicated(keep=False)
        dup_sample = df[dup_mask].head(10)
        table(dup_sample, width="stretch")
        st.caption("Sample of duplicate rows (showing first 10)")
    else:
        st.success("No duplicate rows found!")
    audit_results['duplicates'] = n_duplicates

# -------------------------------------------------------------------------
# NUMERIC SUMMARY
# -------------------------------------------------------------------------
numeric_cols = get_numeric_columns(df)
if numeric_cols:
    with st.expander("Numeric Column Statistics", expanded=False):
        numeric_stats = df[numeric_cols].describe().T
        numeric_stats['skewness'] = df[numeric_cols].skew()
        numeric_stats['kurtosis'] = df[numeric_cols].kurtosis()
        table(numeric_stats.round(3), width="stretch")
        audit_results['numeric_stats'] = numeric_stats.to_dict()
        
        # Flag potential outliers
        n_rows = len(df)
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            outlier_count = ((df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)).sum()
            if n_rows > 0 and outlier_count > n_rows * 0.05:  # More than 5% outliers
                outlier_pct = (outlier_count / n_rows * 100)
                st.info(f"**{col}**: {outlier_count} potential outliers ({outlier_pct:.1f}%)")

# -------------------------------------------------------------------------
# SUGGESTED ACTIONS
# -------------------------------------------------------------------------
constants_cols = [c['Column'] for c in audit_results.get('cardinality', []) if c['Type'] == 'Constant']
n_rows = len(df)
high_missing_cols = [
    r['Column'] for r in audit_results.get('missing', [])
    if n_rows > 0 and (r['Missing Count'] / n_rows * 100) > 50
]
cols_with_missing = [r['Column'] for r in audit_results.get('missing', []) if r['Missing Count'] > 0]
has_duplicates = audit_results.get('duplicates', 0) > 0

suggested_actions = []
if constants_cols and len(constants_cols) < len(df.columns):
    suggested_actions.append(("Drop constant columns", constants_cols, lambda d, cols=constants_cols: d.drop(columns=cols, errors='ignore')))
if high_missing_cols and len(high_missing_cols) < len(df.columns):
    suggested_actions.append(("Drop high-missing columns (>50%)", high_missing_cols, lambda d, cols=high_missing_cols: d.drop(columns=cols, errors='ignore')))
if has_duplicates:
    suggested_actions.append(("Drop duplicate rows", [], lambda d: d.drop_duplicates()))
if cols_with_missing:
    def _impute_missing(d):
        out = d.copy()
        for col in out.columns:
            if out[col].isnull().any():
                if pd.api.types.is_numeric_dtype(out[col]):
                    out[col] = out[col].fillna(out[col].median())
                else:
                    mode_val = out[col].mode()
                    out[col] = out[col].fillna(mode_val.iloc[0] if len(mode_val) > 0 else "")
        return out
    suggested_actions.append(("Impute missing values (median/mode)", cols_with_missing, _impute_missing))

if suggested_actions:
    with st.expander("Suggested Actions", expanded=True):
        st.caption("One-click fixes based on audit findings. Each action updates your working table.")
        for i, (label, cols, apply_fn) in enumerate(suggested_actions):
            col_list = f": {', '.join(cols[:5])}{'...' if len(cols) > 5 else ''}" if cols else ""
            if st.button(f"Apply: {label}{col_list}", key=f"apply_suggested_{i}"):
                try:
                    new_df = apply_fn(df)
                    if len(new_df) == 0 or len(new_df.columns) == 0:
                        st.error("This action would result in an empty dataset. Aborted.")
                    else:
                        st.session_state.working_table = new_df
                        set_data(new_df)
                        st.success(f"Applied: {label}. New shape: {new_df.shape[0]:,} rows × {new_df.shape[1]} columns")
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to apply: {e}")
                    logger.exception(e)

st.session_state.data_audit = audit_results

# ============================================================================
# SECTION 5: TASK MODE & FIELD SELECTION
# ============================================================================
st.markdown("---")
st.header("Step 4: Configure Analysis")

# Task mode selection — styled cards, no pre-selection
current_task_mode = st.session_state.get('task_mode')

st.markdown("**What type of analysis do you want to perform?**")
_card_cols = st.columns(2)

with _card_cols[0]:
    _pred_selected = current_task_mode == 'prediction'
    _pred_border = "#667eea" if _pred_selected else "#e2e8f0"
    _pred_bg = "#f0f2ff" if _pred_selected else "#ffffff"
    _pred_check = "<div style='color:#667eea; font-weight:600; margin-top:4px;'>✓ Selected</div>" if _pred_selected else "<div style='margin-top:4px;'>&nbsp;</div>"
    st.markdown(f"<div style='border:2px solid {_pred_border}; border-radius:12px; padding:20px; background:{_pred_bg}; text-align:center; margin-bottom:8px;'><div style='font-size:2em;'>📊</div><div style='font-weight:600; font-size:1.1em;'>Prediction</div><div style='color:#64748b; font-size:0.85em;'>Build &amp; compare ML models</div>{_pred_check}</div>", unsafe_allow_html=True)
    if st.button("Select Prediction", key="btn_prediction", type="primary" if _pred_selected else "secondary", use_container_width=True):
        st.session_state.task_mode = 'prediction'
        st.rerun()

with _card_cols[1]:
    _hyp_selected = current_task_mode == 'hypothesis_testing'
    _hyp_border = "#667eea" if _hyp_selected else "#e2e8f0"
    _hyp_bg = "#f0f2ff" if _hyp_selected else "#ffffff"
    _hyp_check = "<div style='color:#667eea; font-weight:600; margin-top:4px;'>✓ Selected</div>" if _hyp_selected else "<div style='margin-top:4px;'>&nbsp;</div>"
    st.markdown(f"<div style='border:2px solid {_hyp_border}; border-radius:12px; padding:20px; background:{_hyp_bg}; text-align:center; margin-bottom:8px;'><div style='font-size:2em;'>🔬</div><div style='font-weight:600; font-size:1.1em;'>Hypothesis Testing</div><div style='color:#64748b; font-size:0.85em;'>Statistical tests without ML</div>{_hyp_check}</div>", unsafe_allow_html=True)
    if st.button("Select Hypothesis Testing", key="btn_hypothesis", type="primary" if _hyp_selected else "secondary", use_container_width=True):
        st.session_state.task_mode = 'hypothesis_testing'
        st.rerun()

task_mode = st.session_state.get('task_mode')

if task_mode is None:
    st.info("👆 Choose an analysis type above to continue.")
    st.stop()

if task_mode == "prediction":
    st.info("📊 **Prediction Mode**: Select a target variable and features to build predictive models.")
    
    # Field selection for prediction
    numeric_cols, categorical_cols = get_selectable_columns(df)
    all_cols = numeric_cols + categorical_cols
    
    if not all_cols:
        st.error("No selectable columns found in the data.")
        st.stop()
    
    # Target selection
    existing_config = st.session_state.get('data_config')
    existing_target = existing_config.target_col if existing_config else None
    
    target_idx = 0
    if existing_target and existing_target in all_cols:
        target_idx = all_cols.index(existing_target) + 1
    
    target_col = st.selectbox(
        "Target Variable (what you want to predict)",
        options=[''] + all_cols,
        index=target_idx,
        key="target_selectbox"
    )
    
    # Feature selection with "Select All" option
    if target_col:
        feature_options = [c for c in all_cols if c != target_col]
        n_available_features = len(feature_options)
        
        st.markdown(f"**Feature Variables** ({n_available_features} available)")
        
        # High-dimensional data warnings
        if n_available_features > 100:
            st.warning(f"""
            **High-dimensional data detected:** {n_available_features} potential features.
            
            Considerations:
            - EDA plots will only show the first 6-10 features in some views
            - Correlation heatmaps may be hard to read with many features
            - Some models may be slow to train
            - Consider feature selection or dimensionality reduction
            """)
        elif n_available_features > 50:
            st.info(f"""
            **Note:** {n_available_features} features available. Some EDA visualizations 
            will be limited to the first several features for readability.
            """)
        
        # Select All / Clear All buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("Select All Features", key="select_all_features"):
                # Directly set the multiselect widget state
                st.session_state.features_multiselect = feature_options
                st.rerun()
        with col2:
            if st.button("Clear Selection", key="clear_features"):
                st.session_state.features_multiselect = []
                st.rerun()
        
        # Determine default selection (only used if widget hasn't been rendered yet)
        existing_features = existing_config.feature_cols if existing_config else []
        
        if 'features_multiselect' not in st.session_state:
            # First time rendering - set initial default
            if existing_features:
                default_features = [f for f in existing_features if f in feature_options]
            else:
                default_features = feature_options[:min(10, len(feature_options))]
        else:
            # Widget already exists, use its current value
            default_features = [f for f in st.session_state.features_multiselect if f in feature_options]
        
        selected_features = st.multiselect(
            "Select features to use as predictors",
            options=feature_options,
            default=default_features,
            key="features_multiselect",
            help=f"Select from {n_available_features} available features. Use 'Select All' to include everything."
        )
        
        # Show selection summary
        if selected_features:
            st.caption(f"Selected {len(selected_features)} of {n_available_features} features")
            
            if len(selected_features) > 50:
                st.info("""
                **With many features selected:**
                - Some EDA visualizations will be limited
                - Consider if all features are necessary
                - Training time may increase
                """)
    else:
        selected_features = []
        feature_options = []
    
    if target_col and selected_features:
        # Task type detection
        task_detection = st.session_state.get('task_type_detection', TaskTypeDetection())
        existing_config = st.session_state.get('data_config')
        
        should_redetect = (
            task_detection.detected is None or
            existing_config is None or
            existing_config.target_col != target_col
        )
        
        if should_redetect:
            with st.spinner("Detecting task type..."):
                task_result = detect_task_type(df, target_col)
                task_detection = TaskTypeDetection(
                    detected=task_result['detected'],
                    confidence=task_result['confidence'],
                    reasons=task_result['reasons']
                )
                st.session_state.task_type_detection = task_detection
        
        # Show detection result
        task_det = st.session_state.task_type_detection
        if task_det.detected:
            st.success(f"Detected task type: **{task_det.detected.title()}**")
        
        # Override option
        with st.expander("Override Task Type"):
            override = st.checkbox("Override auto-detected task type", key="task_override")
            if override:
                override_value = st.radio(
                    "Task Type",
                    ['regression', 'classification'],
                    horizontal=True,
                    key="task_override_radio"
                )
                task_detection.override_enabled = True
                task_detection.override_value = override_value
                st.session_state.task_type_detection = task_detection
        
        task_type_final = task_detection.final
        
        # Save configuration
        data_config = DataConfig(
            target_col=target_col,
            feature_cols=selected_features,
            task_type=task_type_final
        )
        st.session_state.data_config = data_config
        
        st.success(f"✅ Configuration saved: **{task_type_final.title()}** task with **{len(selected_features)}** features")
        
        # Next steps
        st.markdown("---")
        st.markdown("### Next Steps")
        st.markdown("""
        1. Go to **EDA** to explore your data
        2. Go to **Preprocess** to build your preprocessing pipeline
        3. Go to **Train & Compare** to train models
        """)
    else:
        st.warning("Please select a target variable and at least one feature.")

else:
    st.info("🔬 **Hypothesis Testing Mode**: Run statistical tests on your variables.")
    
    st.session_state.data_config = DataConfig()  # Clear prediction config
    
    st.markdown("""
    ### Available Tests
    - **Correlation**: Test relationship between two numeric variables
    - **Two-Sample Comparison**: Compare means between two groups
    - **Multi-Group Comparison**: Compare means across multiple groups (ANOVA)
    - **Categorical Association**: Test association between categorical variables (Chi-square)
    - **Normality Test**: Check if a variable is normally distributed
    
    Go to the **Hypothesis Testing** page to run tests.
    """)

# ============================================================================
# WHAT HAPPENS NEXT
# ============================================================================
st.markdown("---")
st.markdown("""
### What Happens Next?

You've uploaded your data and selected a target variable. Here's your workflow:

1. **Explore Your Data (EDA)** — Distributions, correlations, missing patterns, Table 1
2. **Optional: Engineer Features** — Create polynomial, ratio, or TDA features if needed
3. **Select Features** — Identify the most predictive variables
4. **Train Models** — Compare 18 different algorithms with bootstrap CIs
5. **Validate & Export** — SHAP, calibration, sensitivity, publication-ready reports

👉 **Continue to Exploratory Data Analysis (EDA)**
""")

# ============================================================================
# STATE DEBUG
# ============================================================================
with st.expander("Debug: Session State", expanded=False):
    st.write(f"• Active Project: {active_project['name'] if active_project else 'None'}")
    st.write(f"• Datasets in Project: {len(project_datasets)}")
    st.write(f"• Working Table Shape: {df.shape if df is not None else 'None'}")
    st.write(f"• Task Mode: {st.session_state.get('task_mode', 'None')}")
    st.write(f"• Merge Steps: {len(st.session_state.get('merge_steps', []))}")
    data_config = st.session_state.get('data_config')
    st.write(f"• Target: {data_config.target_col if data_config else 'None'}")
    st.write(f"• Features: {len(data_config.feature_cols) if data_config else 0}")

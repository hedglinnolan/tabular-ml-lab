"""
Page 01: Upload and Data Audit
Project-based data management with intelligent merging.

AUDIT NOTE (Data Flow):
- Sets raw_data and working_table in session state
- Data cleaning actions modify working_table and call set_data()
- Methodology logging: Added for all suggested data cleaning actions (drop columns, drop duplicates, impute, etc.)
- Feature selection stored in data_config.feature_cols
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime

from utils.session_state import (
    init_session_state, set_data, get_data, DataConfig, reset_data_dependent_state,
    TaskTypeDetection, CohortStructureDetection, log_methodology
)
from utils.datasets import get_builtin_datasets
from utils.reconcile import reconcile_target_features
from utils.state_reconcile import reconcile_state_with_df
from utils.storyline import render_breadcrumb, render_page_navigation
from utils.session_projects import get_project_manager
from utils.column_utils import make_unique_columns
from utils.theme import inject_custom_css, render_guidance, render_sidebar_workflow
from utils.table_export import table
from utils.import_ui import render_import_doctor, repaired_frame
from data_processor import (
    load_tabular_data, get_numeric_columns, get_selectable_columns,
    detect_file_type, inspect_json
)
from utils.perf_cache import (
    cached_parse_upload, cached_audit_tables, cached_numeric_summary,
)
from ml.triage import detect_task_type, detect_cohort_structure
from ml.eda_recommender import compute_dataset_signals

logger = logging.getLogger(__name__)


# =============================================================================
# HELPER: Visual Schema Diagram
# =============================================================================
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

st.title("📂 Upload & Audit")
from utils.theme import render_flash
render_flash()
st.caption("Start here. Add your data — one file or several — confirm it looks right, then choose your analysis setup.")
render_guidance(
    "<strong>How this page works:</strong> 1) add your file or files, 2) combine them if there is more than one — "
    "the app proposes how and shows you the result first, 3) review the working table and audit, "
    "4) choose your target and continue to EDA.",
    icon="🧭"
)
render_breadcrumb("01_Upload_and_Audit")
render_page_navigation("01_Upload_and_Audit")

# Progress indicator

# ============================================================================
# DATA PERSISTENCE INFO & MANAGEMENT (Sidebar)
# ============================================================================
with st.sidebar:
    st.subheader("Session & Data")
    st.caption("Add one file or several. If you bring more than one, the app combines them for you.")
    
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
st.header("Step 1: Add Your Data")
st.caption(
    "Bring one file or bring all of them. If your study lives in several files "
    "that you have never combined — demographics here, labs there, diet in a "
    "third — that is exactly what this page is for. Add them all, and Step 2 "
    "combines them for you."
)

# Initialize datasets registry for this project
if 'datasets_registry' not in st.session_state:
    st.session_state.datasets_registry = {}

# Show existing datasets in project
project_datasets = db.get_project_datasets(active_project['id'])

if project_datasets:
    # A visible roster, not a buried table. When someone is assembling a study
    # from several files, "what have I added so far" is the question they ask
    # after every single upload — it should never take two clicks to answer.
    st.markdown(f"**In this project ({len(project_datasets)})**")
    for d in sorted(project_datasets, key=lambda x: x.get('upload_timestamp', '')):
        in_memory = d['id'] in st.session_state.datasets_registry
        c1, c2, c3 = st.columns([5, 2, 1])
        with c1:
            icon = "📄" if in_memory else "⚠️"
            st.markdown(f"{icon} **{d['name']}** — {d['shape_rows']:,} rows × "
                        f"{d['shape_cols']} columns")
            if not in_memory:
                st.caption("Not loaded in this session — re-upload the file below "
                           "to use it.")
        with c2:
            new_name = st.text_input(
                "Rename", value=d['name'], key=f"rename_{d['id']}",
                label_visibility="collapsed",
            )
            if new_name and new_name != d['name']:
                if new_name in [o['name'] for o in project_datasets]:
                    st.caption("Name already used.")
                elif st.button("Save name", key=f"rename_go_{d['id']}"):
                    db.rename_dataset(d['id'], new_name)
                    st.rerun()
        with c3:
            if st.button("Remove", key=f"del_{d['id']}", type="secondary"):
                db.delete_dataset(d['id'])
                st.session_state.datasets_registry.pop(d['id'], None)
                # The working table was built from a set of files that no
                # longer exists; forcing a recombine is the only honest move.
                st.session_state.pop("working_table", None)
                st.session_state.pop("_combine_signature", None)
                st.rerun()
    st.markdown("---")

# File upload
if project_datasets:
    st.subheader("Add another file")
    st.caption("Drop in as many as you like — they are combined in Step 2.")
else:
    st.subheader("Upload your data")
    st.caption("One file or several. If your study spans multiple files, add "
               "them all now and combine them in the next step.")

uploaded_files = st.file_uploader(
    "Upload data files (CSV, Excel, Parquet, TSV, JSON)",
    type=['csv', 'xlsx', 'xls', 'parquet', 'tsv', 'txt', 'json', 'jsonl', 'ndjson'],
    accept_multiple_files=True,
    key="file_uploader",
    help=(
        "JSON works when it holds a table: a list of records "
        '(e.g. [{"age": 40, "bmi": 22.1}, …]), a wrapped payload like '
        '{"data": [...]}, or JSON Lines (one record per line). Nested fields '
        "are flattened into dotted columns (vitals.bp)."
    ),
)

MAX_FILE_SIZE_MB = 50


def _file_key_for(name: str, idx: int) -> str:
    return f"{name.replace('.', '_').replace(' ', '_')}_{idx}"


def _log_import_repairs(file_key, dataset_name):
    """Put the Import Doctor's repairs in the record, not just on the screen.

    The audit's Suggested Actions already call log_methodology and
    record_cleaning; the Doctor's fixes went nowhere, so a manuscript said
    nothing about a column whose sentinel codes had been recoded or whose type
    had been changed before anything else saw the data. ml/publication.py
    builds its data-preparation paragraph from exactly those records.
    """
    try:
        from utils.import_ui import applied_fixes
        fixes = applied_fixes(file_key)
        if not fixes:
            return
        for fix in fixes:
            log_methodology(step='Data Cleaning',
                            action=f"{dataset_name}: {fix}",
                            details={'source': 'import_doctor',
                                     'dataset': dataset_name, 'fix': fix})
        from utils.workflow_provenance import get_provenance
        prov = get_provenance()
        for fix in fixes:
            prov.record_cleaning(action=f"Import repair — {fix}",
                                 rows_before=0, rows_after=0,
                                 details={'dataset': dataset_name,
                                          'source': 'import_doctor'})
    except Exception:
        pass          # recording must never block the upload


def _commit_dataset(df, dataset_name, filename, file_type, transposed, replace=True):
    """Register one frame as a dataset in the active project.

    Shared by the per-file 'Add' button and the 'Add all' button so both paths
    commit exactly the frame the user reviewed — including any Import Doctor
    fixes — rather than a fresh re-parse of the original file.
    """
    existing = db.get_project_datasets(active_project['id'])
    for d in existing:
        if d['name'] == dataset_name:
            if not replace:
                return False, f"A dataset named '{dataset_name}' already exists."
            db.delete_dataset(d['id'])
            st.session_state.datasets_registry.pop(d['id'], None)
            break

    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    dataset_id = db.add_dataset(
        project_id=active_project['id'],
        name=dataset_name,
        filename=filename,
        file_type=file_type,
        shape_rows=df.shape[0],
        shape_cols=df.shape[1],
        columns=[str(c) for c in df.columns],
        column_types={str(c): str(df[c].dtype) for c in df.columns},
        is_transposed=transposed,
    )
    st.session_state.datasets_registry[dataset_id] = df
    # A new file changes what "combined" means, so any table built from the
    # previous set of files must not survive as the working table.
    st.session_state.pop("working_table", None)
    st.session_state.pop("_combine_signature", None)
    return True, dataset_name


if uploaded_files and len(uploaded_files) > 1:
    # Four files should not cost four trips to a button. Someone assembling a
    # study from separate exports wants them all in, then wants Step 2.
    st.info(f"**{len(uploaded_files)} files ready.** Add them all at once, or "
            f"open any file below to rename it, transpose it, or fix structural "
            f"problems first.")
    if st.button(f"Add all {len(uploaded_files)} files to project",
                 type="primary", key="add_all_files"):
        added, failed = [], []
        with st.spinner("Adding files…"):
            for _idx, _uf in enumerate(uploaded_files):
                _fk = _file_key_for(_uf.name, _idx)
                try:
                    _ft = detect_file_type(_uf.name)
                    _frame = cached_parse_upload(
                        _uf.getvalue(), _uf.name,
                        st.session_state.get(f"transpose_{_fk}", False),
                        st.session_state.get(f"excel_sheet_{_fk}", 0) if _ft == 'excel' else 0,
                        st.session_state.get(f"records_key_{_fk}", "") or "",
                    )
                    _frame.columns = [str(c) for c in _frame.columns]
                    # Honor fixes already applied in the review below.
                    _frame = repaired_frame(_frame, _fk)
                    _name = st.session_state.get(f"name_{_fk}") or _uf.name.rsplit('.', 1)[0]
                    _log_import_repairs(_fk, _name)
                    ok, msg = _commit_dataset(_frame, _name, _uf.name, _ft,
                                              st.session_state.get(f"transpose_{_fk}", False))
                    (added if ok else failed).append(msg)
                except Exception as exc:
                    failed.append(f"{_uf.name}: {exc}")
        if added:
            st.success(f"Added {len(added)} file{'s' if len(added) != 1 else ''}: "
                       + ", ".join(added))
        for msg in failed:
            st.error(msg)
        if added:
            st.rerun()

if uploaded_files:
    for file_idx, uploaded_file in enumerate(uploaded_files):
        file_type = detect_file_type(uploaded_file.name)
        file_key = _file_key_for(uploaded_file.name, file_idx)

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

                # JSON: say where the rows are being read from, and let the
                # user correct it. The loader used to raise "pick which key
                # holds your rows" with no way to pick, and to resolve a
                # payload holding several wrapper keys by iteration order.
                records_key_choice = ""
                if file_type in ('json', 'jsonl'):
                    uploaded_file.seek(0)
                    layout = inspect_json(uploaded_file, lines=(file_type == 'jsonl'))
                    uploaded_file.seek(0)
                    if layout.error:
                        st.error(layout.error)
                        continue
                    if layout.candidates:
                        default_idx = (layout.candidates.index(layout.chosen_key)
                                       if layout.chosen_key in layout.candidates else 0)
                        records_key_choice = st.selectbox(
                            "Which part of this file holds your rows?",
                            layout.candidates, index=default_idx,
                            key=f"records_key_{file_key}",
                            help="This JSON wraps its table inside a key. Pick the "
                                 "one holding your records.",
                        )
                    if layout.note:
                        st.caption(f"ℹ️ {layout.note}")
                
                # Per-file transpose option
                transpose_this_file = st.checkbox(
                    "Transpose this file (rows ↔ columns)",
                    value=False,
                    key=f"transpose_{file_key}",
                    help="Use this if your features are in rows instead of columns"
                )
                
                # Load preview with transpose setting. Cached on file content:
                # this block re-executes on every rerun while the file sits in
                # the uploader, and re-parsing a wide file each click costs
                # seconds.
                with st.spinner(f"Loading {uploaded_file.name}..."):
                    df_preview = cached_parse_upload(
                        uploaded_file.getvalue(),
                        uploaded_file.name,
                        transpose_this_file,
                        excel_sheet_choice if file_type == 'excel' else 0,
                        records_key_choice,
                    )
                
                # Reset file position for later
                uploaded_file.seek(0)
                
                # Ensure column names are strings for merging compatibility
                df_preview.columns = [str(c) for c in df_preview.columns]

                # Structural review, before anything is committed. The doctor
                # returns the frame with the user's applied fixes, and that —
                # not a re-parse — is what gets added to the project.
                df_preview = render_import_doctor(df_preview, file_key)

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
                    
                    # A file stays in the uploader after it has been added, so
                    # this collision fired for every file the user had just
                    # successfully added — four alarming warnings for an action
                    # that worked. Same name AND same shape means it IS this
                    # file, so confirm it instead of warning about it.
                    already_added = next(
                        (d for d in (project_datasets or [])
                         if d['name'] == dataset_name
                         and d['shape_rows'] == df_preview.shape[0]
                         and d['shape_cols'] == df_preview.shape[1]), None)
                    if already_added:
                        st.success("✓ Added to your project.")
                        replace_existing = True
                    elif name_exists:
                        st.warning(
                            f"A different dataset is already called "
                            f"'{dataset_name}'. Rename this one, or replace it.")
                        replace_existing = st.checkbox(
                            f"Replace existing '{dataset_name}'",
                            key=f"replace_{file_key}"
                        )
                    else:
                        replace_existing = False
                    
                    if st.button(f"Add to Project", key=f"add_{file_key}", type="primary"):
                        if name_exists and not replace_existing:
                            st.error("Please check 'Replace existing' or change the dataset name.")
                            st.stop()

                        # df_preview already carries any Import Doctor fixes,
                        # so committing it is what keeps "what I reviewed" and
                        # "what got added" the same frame.
                        with st.spinner(f"Adding {dataset_name} to project..."):
                            _log_import_repairs(file_key, dataset_name)
                            ok, msg = _commit_dataset(
                                df_preview, dataset_name, uploaded_file.name,
                                file_type, transpose_this_file, replace=True,
                            )
                        if not ok:
                            st.error(msg)
                        else:
                            st.success(f"Added '{dataset_name}' to project!")
                            st.rerun()
                        
            except Exception as e:
                st.error(f"Error loading file: {e}")
                logger.exception(e)

# Built-in datasets option
with st.expander("Need a practice dataset instead?", expanded=False):
    st.caption("Useful for exploration or demos. Real projects will usually start with your own dataset.")
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
# SECTION 3: COMBINE FILES INTO ONE WORKING TABLE
# ============================================================================
st.markdown("---")

if len(project_datasets) > 1:
    # Load every dataset that is actually in memory, oldest first so the
    # defaults in the combine UI refer to the file the user uploaded first.
    dataframes = {}
    # get_project_datasets returns newest-first; combining reads far better in
    # upload order, so the first file the user added is the one others attach to.
    for d in sorted(project_datasets, key=lambda x: x.get('upload_timestamp', '')):
        if d['id'] in st.session_state.datasets_registry:
            _tmp = st.session_state.datasets_registry[d['id']].copy()
            _tmp.columns = [str(c) for c in _tmp.columns]
            dataframes[d['name']] = _tmp

    if len(dataframes) < len(project_datasets):
        st.error(
            f"{len(project_datasets) - len(dataframes)} of your {len(project_datasets)} "
            f"files are no longer loaded. Re-upload them above, or remove their records, "
            f"to continue."
        )
        st.stop()

    from utils.combine_ui import render_combine_step, render_combined_summary

    # The set of contributing files is part of the working table's identity:
    # adding a file must not leave the analysis quietly running on the old
    # combination while the page reports "ready".
    _combo_signature = "|".join(sorted(dataframes)) + f"|{len(dataframes)}"
    if st.session_state.get("_combine_signature") != _combo_signature:
        st.session_state.pop("working_table", None)
        st.session_state["_combine_signature"] = _combo_signature

    _combined = render_combine_step(dataframes)
    if _combined is not None:
        st.session_state.working_table = _combined
        st.session_state.last_merge_columns = list(_combined.columns)
        set_data(_combined)
        st.rerun()

    if st.session_state.get("working_table") is not None:
        working_df = st.session_state.working_table
        set_data(working_df)
        render_combined_summary(working_df)
        table(working_df.head(10), width="stretch")
    else:
        st.stop()


else:
    # Single dataset - use it directly
    st.header("Step 2: Working Table")
    st.caption("With a single dataset, it becomes your working table directly.")
    
    single_dataset = project_datasets[0]
    
    if single_dataset['id'] in st.session_state.datasets_registry:
        # Rebuild the working table from the registry only when the source
        # dataset changes — rebuilding on every rerun silently reverted any
        # cleaning actions applied to the working table.
        if (st.session_state.get('working_table') is None
                or st.session_state.get('_working_table_source_id') != single_dataset['id']):
            working_df = st.session_state.datasets_registry[single_dataset['id']].copy()
            # Ensure string column names
            working_df.columns = [str(c) for c in working_df.columns]
            st.session_state.working_table = working_df
            st.session_state['_working_table_source_id'] = single_dataset['id']
        else:
            working_df = st.session_state.working_table
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

# Get working table. This page deliberately works on the WHOLE study even when
# a cohort run is active: the audit describes the data, the lockbox must be
# drawn across all groups, and the target and feature list are fixed across runs
# — that is what makes two runs the same question asked of different people.
df = get_data(full_study=True)

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

from utils.cohort_ui import render_cohort_note as _cohort_note
_cohort_note("The audit below covers the whole study, which is what it should "
             "describe — the run filter applies from the EDA page onward.")

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
_cardinality_data, _dtype_data = cached_audit_tables(df)

with st.expander("Cardinality Analysis (Unique Values per Column)", expanded=True):
    st.caption("Helps identify potential ID columns, categorical variables, and constants.")

    cardinality_data = _cardinality_data
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

    dtype_data = _dtype_data
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
        numeric_stats = cached_numeric_summary(df, tuple(numeric_cols)).rename(
            columns={"skew": "skewness"})
        numeric_stats.index.name = 'Feature'
        table(numeric_stats.round(3).reset_index(), width="stretch")
        audit_results['numeric_stats'] = numeric_stats.to_dict()

        # Flag potential outliers — vectorized IQR scan across all columns
        n_rows = len(df)
        if n_rows > 0:
            _num = df[numeric_cols]
            _q1 = _num.quantile(0.25)
            _q3 = _num.quantile(0.75)
            _iqr = _q3 - _q1
            _outlier_counts = ((_num.lt(_q1 - 1.5 * _iqr, axis=1))
                               | (_num.gt(_q3 + 1.5 * _iqr, axis=1))).sum()
            _flagged = _outlier_counts[_outlier_counts > n_rows * 0.05]
            for col, cnt in _flagged.head(10).items():
                st.info(f"**{col}**: {int(cnt)} potential outliers ({cnt / n_rows * 100:.1f}%)")
            if len(_flagged) > 10:
                st.caption(f"…and {len(_flagged) - 10} more columns with >5% potential outliers.")

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
# High-missing column drop removed from suggestions — feature selection handles this downstream
if has_duplicates:
    suggested_actions.append(("Drop duplicate rows", [], lambda d: d.drop_duplicates()))
# Missing data imputation removed from suggestions — Preprocessing handles this properly
# with method selection, per-model pipelines, and MICE support

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
                        # Content changed → set_data clears downstream results
                        # (config is kept); reconcile drops any config references
                        # to columns this action removed.
                        set_data(new_df, is_schema_change=False)
                        reconcile_state_with_df(new_df, st.session_state)
                        log_methodology(step='Data Cleaning', action=label, details={
                            'affected_columns': cols if cols else 'all',
                            'rows_before': df.shape[0],
                            'rows_after': new_df.shape[0],
                            'cols_before': df.shape[1],
                            'cols_after': new_df.shape[1]
                        })
                        try:
                            from utils.workflow_provenance import get_provenance
                            get_provenance().record_cleaning(
                                action=label,
                                rows_before=df.shape[0],
                                rows_after=new_df.shape[0],
                                details={
                                    'affected_columns': cols if cols else 'all',
                                    'rows_before': df.shape[0],
                                    'rows_after': new_df.shape[0],
                                },
                            )
                        except Exception:
                            pass  # Provenance recording should never break the workflow
                        from utils.theme import flash
                        flash("success",
                              f"Applied: {label}. New shape: {new_df.shape[0]:,} rows × {new_df.shape[1]} columns. "
                              f"Downstream results (splits, models, reports) were reset — they described the pre-cleaning data.")
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
    if st.button("Select Prediction", key="btn_prediction", type="primary" if _pred_selected else "secondary"):
        st.session_state.task_mode = 'prediction'
        st.rerun()

with _card_cols[1]:
    _hyp_selected = current_task_mode == 'hypothesis_testing'
    _hyp_border = "#667eea" if _hyp_selected else "#e2e8f0"
    _hyp_bg = "#f0f2ff" if _hyp_selected else "#ffffff"
    _hyp_check = "<div style='color:#667eea; font-weight:600; margin-top:4px;'>✓ Selected</div>" if _hyp_selected else "<div style='margin-top:4px;'>&nbsp;</div>"
    st.markdown(f"<div style='border:2px solid {_hyp_border}; border-radius:12px; padding:20px; background:{_hyp_bg}; text-align:center; margin-bottom:8px;'><div style='font-size:2em;'>🔬</div><div style='font-weight:600; font-size:1.1em;'>Hypothesis Testing</div><div style='color:#64748b; font-size:0.85em;'>Statistical tests without ML</div>{_hyp_check}</div>", unsafe_allow_html=True)
    if st.button("Select Hypothesis Testing", key="btn_hypothesis", type="primary" if _hyp_selected else "secondary"):
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
        # Bookkeeping columns added when files are combined (e.g. which file a
        # row came from) must never be offered as predictors: a model would
        # happily "predict" the source file, which is batch leakage, not science.
        from utils.combine import is_reserved_column as _is_reserved
        feature_options = [c for c in all_cols
                           if c != target_col and not _is_reserved(c)]
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
                # Default to ALL candidate features. An arbitrary first-N cap
                # silently drops predictors on wide data (column order is not
                # relevance order) and makes the EDA tiles describe a subset
                # the user never chose. Narrowing is the Feature Selection
                # page's job, on training rows only.
                default_features = list(feature_options)
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
        
        # Show detection result — confidence-aware. A confident green banner
        # for an ambiguous detection (e.g. low-cardinality integer targets:
        # class codes vs counts vs ratings) silently steers users into the
        # wrong task type.
        task_det = st.session_state.task_type_detection
        if task_det.detected:
            _det_conf = getattr(task_det, 'confidence', None) or 'high'
            if _det_conf == 'high':
                st.success(f"Detected task type: **{task_det.detected.title()}**")
            else:
                _det_reason = ""
                _det_reasons = getattr(task_det, 'reasons', None)
                if _det_reasons:
                    _det_reason = f" — {_det_reasons[-1]}"
                st.info(
                    f"Best guess for task type: **{task_det.detected.title()}** "
                    f"(confidence: {_det_conf}){_det_reason} "
                    f"Please verify, or use the override below."
                )
        
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
        st.session_state.selected_features = list(selected_features)

        import hashlib
        # Invalidation must key on everything that defines the modeling
        # problem: the feature set, the target, and the task type. A target
        # swap with unchanged features previously kept the old target's
        # models/metrics/SHAP alive under the new outcome's name.
        _config_sig = (
            ','.join(sorted(data_config.feature_cols))
            + f"|target={data_config.target_col}|task={data_config.task_type}"
        )
        _new_hash = hashlib.md5(_config_sig.encode()).hexdigest()[:8]
        _old_hash = st.session_state.get('_data_config_features_hash', '')
        if _new_hash != _old_hash:
            st.session_state['_data_config_features_hash'] = _new_hash
            from utils.session_state import reset_downstream_results
            # Keep feature engineering: page-01 selectors operate on the
            # engineered frame, so a re-selection does not invalidate it.
            reset_downstream_results(clear_feature_engineering=False)
            if _old_hash:  # Only warn if this isn't the first save
                st.info('Feature configuration changed — downstream preprocessing, splits, and models have been reset.')

        # Quarantine the test set NOW, before any target-aware analysis can
        # see it. EDA target views, feature-engineering fits, and feature
        # selection all scope to training rows via this lockbox.
        from utils.test_lockbox import (
            ensure_lockbox, render_lockbox_status, get_lockbox,
            DEFAULT_TEST_FRACTION, is_exploratory,
        )
        # A declared subject/entity ID always wins over auto-detection: with
        # repeated measures the split must be by SUBJECT, or the same person
        # lands in both training and the sealed test set.
        _cohort = st.session_state.get('cohort_structure_detection')
        _entity_col = getattr(_cohort, 'entity_id_final', None) if _cohort else None
        _lb = ensure_lockbox(df, target_col, task_type_final, group_col=_entity_col)
        if _lb is not None and _lb.get('group_col'):
            _noun = _lb.get('group_noun') or 'subjects'
            _one = _noun.rstrip('s') if _noun.endswith('s') else _noun
            st.info(
                f"🔒 Rows repeat per {_one} (`{_lb['group_col']}`), so the held-out set was "
                f"drawn by **{_one}**, not by row — {_lb['n_test']:,} rows from "
                f"{_lb.get('n_test_groups', '?')} {_noun}. Splitting by row would put the "
                f"same {_one} in both training and testing."
            )
        if _lb is not None and get_lockbox() is not None:
            _prev_ledger_note = st.session_state.get('_lockbox_ledger_noted')
            if _prev_ledger_note != _lb['signature']:
                st.session_state['_lockbox_ledger_noted'] = _lb['signature']
                try:
                    from utils.insight_ledger import get_ledger, Insight
                    get_ledger().upsert(Insight(
                        id="upload_test_lockbox",
                        source_page="01_Upload_and_Audit",
                        category="study_design",
                        severity="info",
                        finding=(f"A {_lb['fraction']:.0%} test set (n={_lb['n_test']}"
                                 f"{', stratified' if _lb.get('stratified') else ''}) was "
                                 f"held out at upload, before feature engineering or selection."),
                        implication="Held-out evaluation is protected from selection and preprocessing leakage.",
                        recommended_action="",
                        relevant_pages=["06_Train_and_Compare"],
                        tripod_keys=["study_design", "model_building"],
                        resolved=True,
                        resolved_by="Quarantined automatically at upload (seed "
                                    f"{_lb['seed']})",
                        resolved_on_page="01_Upload_and_Audit",
                        resolution_details={"action_type": "test_lockbox",
                                            "params": {"fraction": _lb['fraction'],
                                                       "seed": _lb['seed'],
                                                       "n_test": _lb['n_test']}},
                    ))
                except Exception:
                    pass
        def _on_exploratory_toggle():
            # Both directions invalidate: results computed under one quarantine
            # regime must not survive the flip — otherwise toggling exploratory
            # off would launder full-data feature selection into an
            # unwatermarked manuscript.
            from utils.session_state import reset_downstream_results as _rdr
            _rdr(clear_feature_engineering=False)
            if st.session_state.get("exploratory_mode"):
                st.session_state["exploratory_used"] = True

        with st.expander("🔒 Test holdout settings", expanded=False):
            st.caption(
                "The test fraction is drawn once, here, so that no downstream "
                "step can peek at it. Changing it (or toggling exploratory "
                "mode in either direction) resets downstream results."
            )
            # Session restore defers widget keys via _pending_widget_state_restore;
            # each widget's owner claims its key before instantiation. This page
            # owns exploratory_mode — without this, a restored exploratory flag
            # would sit in the pending dict forever and silently never apply.
            _pending_restore = st.session_state.get("_pending_widget_state_restore", {})
            if "exploratory_mode" in _pending_restore:
                st.session_state["exploratory_mode"] = bool(
                    _pending_restore.pop("exploratory_mode")
                )
                if _pending_restore:
                    st.session_state["_pending_widget_state_restore"] = _pending_restore
                else:
                    st.session_state.pop("_pending_widget_state_restore", None)
                if st.session_state["exploratory_mode"]:
                    # Restored sessions in exploratory mode keep their honesty
                    # watermark; quarantine-off must never arrive silently.
                    st.session_state["exploratory_used"] = True
                    st.warning(
                        "🔓 This restored session was saved in **exploratory mode** — "
                        "the test-set quarantine is OFF, as it was when saved."
                    )
            _lb_frac = st.slider(
                "Held-out test fraction", 0.05, 0.40,
                float(st.session_state.get("test_lockbox_fraction", DEFAULT_TEST_FRACTION)),
                0.05, key="lockbox_fraction_slider",
            )
            if _lb_frac != st.session_state.get("test_lockbox_fraction", DEFAULT_TEST_FRACTION):
                # During a one-group run the re-draw is refused, and committing
                # the fraction anyway left the slider reading 30% beside a chip
                # reading 15% with nothing to reconcile them — and the stored
                # 30% was then picked up and acted on later, invalidating runs
                # banked against the 15% set. Commit only what took effect.
                from utils.cohorts import active_cohort as _frac_run
                if _frac_run() is not None:
                    st.warning(
                        f"🔒 The held-out fraction stays at "
                        f"{st.session_state.get('test_lockbox_fraction', DEFAULT_TEST_FRACTION):.0%} "
                        f"while you are working in one group — every run shares "
                        f"the split made before the study was divided, so it "
                        f"cannot be re-drawn now. Go back to analyzing everyone "
                        f"to change it."
                    )
                else:
                    st.session_state["test_lockbox_fraction"] = _lb_frac
                    # Same arguments as the steady-state call above. Omitting
                    # group_col here silently downgraded a subject-level split to a
                    # row-wise one — subjects landed on both sides, the chip lost
                    # its "no subject appears on both sides" clause and gained
                    # ", stratified", which reads as an upgrade, and the redraw
                    # notice said only that the fraction had changed.
                    ensure_lockbox(df, target_col, task_type_final,
                                   fraction=_lb_frac, group_col=_entity_col)
            st.checkbox(
                "Exploratory mode (disable test-set quarantine)",
                key="exploratory_mode",
                on_change=_on_exploratory_toggle,
                help="Target-aware steps see ALL rows, including the test set. "
                     "Useful for hypothesis generation; downstream metrics and "
                     "the manuscript are watermarked as exploratory and are not "
                     "publishable as held-out performance. Toggling in either "
                     "direction resets downstream results.",
            )
        _ended = st.session_state.pop("_cohort_cleared_by_data_change", None)
        if _ended:
            st.warning(
                f"👥 Your one-group run (**{_ended['column']} = {_ended['label']}**) "
                f"ended because the data changed in a way that removed some of "
                f"those rows. This analysis now covers everyone again — "
                f"re-select the group in Step 5 if you still want it."
            )
        _refused = st.session_state.pop("_lockbox_redraw_refused", None)
        if _refused:
            if _refused.get("target_changed"):
                # Changing the outcome mid-run is the one refusal that leaves the
                # sealed set partly unusable: it was drawn among the rows that
                # had a value for the OLD outcome.
                st.warning(
                    f"⚠️ You changed the outcome to **{_refused['target']}**, but "
                    f"the held-out set was sealed for **{_refused['drawn_for']}** "
                    f"and is **not** re-drawn during a one-group run "
                    f"(**{_refused['column']} = {_refused['label']}**) — every run "
                    f"shares one split, which is what lets them be compared. "
                    f"{_refused['n_scoreable']:,} of its {_refused['n_sealed']:,} "
                    f"rows have a value for **{_refused['target']}**, so that is "
                    f"how many you can score against. To hold out a set drawn for "
                    f"this outcome, go back to analyzing everyone first."
                )
            else:
                st.info(
                    f"🔒 The held-out set was **not** re-drawn. You are working in "
                    f"one group (**{_refused['column']} = {_refused['label']}**), and "
                    f"every run shares the single split made before the study was "
                    f"divided — that is what lets your runs be compared. To draw a "
                    f"new one, go back to analyzing everyone first."
                )
        if st.session_state.pop("_lockbox_redrawn", False):
            st.info("🔒 Test lockbox redrawn (data, target, fraction, or seed changed) — "
                    "downstream results were reset so nothing is evaluated against the old test set.")
        # Chip rendered AFTER the settings expander so it always reflects the
        # just-applied fraction rather than the pre-interaction value.
        render_lockbox_status()

        # Log methodology
        log_methodology(
            step='Upload & Audit',
            action=f"Configured {task_type_final} task with {len(selected_features)} features, target: {target_col}",
            details={
                'target': target_col,
                'task_type': task_type_final,
                'n_features': len(selected_features),
                'features': selected_features,
                'n_samples': len(df),
                'data_source': st.session_state.get('data_source', 'unknown'),
            }
        )
        try:
            from utils.workflow_provenance import get_provenance
            get_provenance().record_upload(
                target_col=target_col,
                task_type=task_type_final,
                feature_cols=selected_features,
                n_samples=len(df),
                data_source=st.session_state.get('data_source', 'unknown'),
            )
        except Exception:
            pass  # Provenance recording should never break the workflow

        st.success(f"✅ Configuration saved: **{task_type_final.title()}** task with **{len(selected_features)}** features")

        # Cohort runs come LAST, after the lockbox exists and the configuration
        # is saved: every run inherits its slice of that ONE split, and the
        # target and features above stay fixed across runs. Both facts depend on
        # this ordering, and the reading order should match it.
        from utils.cohort_ui import render_cohort_chooser
        render_cohort_chooser(df, target_col, task_type_final, selected_features,
                              group_col=_entity_col)
        # (Next-step guidance renders once, in the consolidated "What Happens
        # Next?" section below — two adjacent, slightly different step lists
        # read as contradictory.)
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
# WHAT HAPPENS NEXT — only once configuration is actually saved; the previous
# unconditional copy claimed "You've ... selected a target variable" while the
# warning above it said no target was selected.
# ============================================================================
_dc_next = st.session_state.get('data_config')
if (st.session_state.get('task_mode') == 'prediction'
        and _dc_next is not None and getattr(_dc_next, 'target_col', None)):
    st.markdown("---")
    st.markdown("""
### What Happens Next?

You've uploaded your data and selected a target variable. Here's your workflow:

1. **Explore Your Data (EDA)** — Distributions, correlations, missing patterns, Table 1
2. **Optional: Engineer Features** — Create polynomial, ratio, or TDA features if needed
3. **Select Features** — Identify the most predictive variables
4. **Train Models** — Compare up to 22 models with bootstrap CIs
5. **Validate & Export** — SHAP, calibration, sensitivity, publication-ready reports

👉 **Continue to Exploratory Data Analysis (EDA)**
""")

# ============================================================================
# STATE DEBUG — developer tooling; never shown on the golden path
# ============================================================================
if st.session_state.get("show_debug_panel"):
  with st.expander("Debug: Session State", expanded=False):
    st.write(f"• Active Project: {active_project['name'] if active_project else 'None'}")
    st.write(f"• Datasets in Project: {len(project_datasets)}")
    st.write(f"• Working Table Shape: {df.shape if df is not None else 'None'}")
    st.write(f"• Task Mode: {st.session_state.get('task_mode', 'None')}")
    st.write(f"• Merge Steps: {len(st.session_state.get('merge_steps', []))}")
    data_config = st.session_state.get('data_config')
    st.write(f"• Target: {data_config.target_col if data_config else 'None'}")
    st.write(f"• Features: {len(data_config.feature_cols) if data_config else 0}")

"""
Session-only project and dataset management.

All state lives in st.session_state — nothing written to disk.
Each browser session is fully isolated. Data does not persist across page refreshes.
"""
import streamlit as st
import pandas as pd
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime
import json


def _projects() -> Dict[int, dict]:
    """Get the projects dict from session state, initializing if needed."""
    if "sp_projects" not in st.session_state:
        st.session_state.sp_projects = {}
    return st.session_state.sp_projects


def _next_id(key: str) -> int:
    counter_key = f"sp_counter_{key}"
    val = st.session_state.get(counter_key, 0) + 1
    st.session_state[counter_key] = val
    return val


class SessionProjectManager:
    """Drop-in replacement for DatasetDB using only session state."""

    # ── Projects ─────────────────────────────────────────────────────

    def create_project(self, name: str, description: str = "") -> int:
        projects = _projects()
        pid = _next_id("project")
        now = datetime.now(datetime.UTC).isoformat()
        # Deactivate others
        for p in projects.values():
            p["active"] = False
        projects[pid] = {
            "id": pid,
            "name": name,
            "description": description,
            "created_at": now,
            "updated_at": now,
            "active": True,
            "datasets": {},
            "merge_configs": {},
        }
        return pid

    def get_project(self, project_id: int) -> Optional[Dict[str, Any]]:
        return _projects().get(project_id)

    def get_all_projects(self) -> List[Dict[str, Any]]:
        return sorted(_projects().values(), key=lambda p: p["updated_at"], reverse=True)

    def get_active_project(self) -> Optional[Dict[str, Any]]:
        for p in _projects().values():
            if p.get("active"):
                return p
        return None

    def set_active_project(self, project_id: int) -> bool:
        projects = _projects()
        if project_id not in projects:
            return False
        for p in projects.values():
            p["active"] = False
        projects[project_id]["active"] = True
        return True

    def delete_project(self, project_id: int) -> bool:
        projects = _projects()
        if project_id in projects:
            del projects[project_id]
            return True
        return False

    # ── Datasets ─────────────────────────────────────────────────────

    def add_dataset(
        self,
        project_id: int,
        name: str,
        filename: str,
        file_type: str,
        shape_rows: int,
        shape_cols: int,
        columns: List[str],
        column_types: Optional[Dict[str, str]] = None,
        is_transposed: bool = False,
    ) -> int:
        project = _projects().get(project_id)
        if not project:
            return -1
        did = _next_id("dataset")
        project["datasets"][did] = {
            "id": did,
            "project_id": project_id,
            "name": name,
            "filename": filename,
            "file_type": file_type,
            "shape_rows": shape_rows,
            "shape_cols": shape_cols,
            "columns": columns,
            "column_types": column_types,
            "upload_timestamp": datetime.now(datetime.UTC).isoformat(),
            "is_transposed": is_transposed,
        }
        project["updated_at"] = datetime.now(datetime.UTC).isoformat()
        return did

    def get_dataset(self, dataset_id: int) -> Optional[Dict[str, Any]]:
        for p in _projects().values():
            if dataset_id in p["datasets"]:
                return p["datasets"][dataset_id]
        return None

    def get_project_datasets(self, project_id: int) -> List[Dict[str, Any]]:
        project = _projects().get(project_id)
        if not project:
            return []
        return sorted(
            project["datasets"].values(),
            key=lambda d: d["upload_timestamp"],
            reverse=True,
        )

    def delete_dataset(self, dataset_id: int) -> bool:
        for p in _projects().values():
            if dataset_id in p["datasets"]:
                del p["datasets"][dataset_id]
                return True
        return False

    # ── Merge configs ────────────────────────────────────────────────

    def save_merge_config(
        self,
        project_id: int,
        name: str,
        merge_steps: List[Dict[str, Any]],
        result_shape: Tuple[int, int],
        result_columns: List[str],
        set_as_working: bool = True,
    ) -> int:
        project = _projects().get(project_id)
        if not project:
            return -1
        if set_as_working:
            for mc in project["merge_configs"].values():
                mc["is_working_table"] = False
        mid = _next_id("merge_config")
        project["merge_configs"][mid] = {
            "id": mid,
            "project_id": project_id,
            "name": name,
            "merge_steps": merge_steps,
            "result_shape_rows": result_shape[0],
            "result_shape_cols": result_shape[1],
            "result_columns": result_columns,
            "created_at": datetime.now(datetime.UTC).isoformat(),
            "is_working_table": set_as_working,
        }
        return mid

    def get_working_table_config(self, project_id: int) -> Optional[Dict[str, Any]]:
        project = _projects().get(project_id)
        if not project:
            return None
        for mc in project["merge_configs"].values():
            if mc.get("is_working_table"):
                return mc
        return None

    # ── Stats / reset ────────────────────────────────────────────────

    def get_database_stats(self) -> Dict[str, Any]:
        projects = _projects()
        n_datasets = sum(len(p["datasets"]) for p in projects.values())
        n_merges = sum(len(p["merge_configs"]) for p in projects.values())
        return {
            "n_projects": len(projects),
            "n_datasets": n_datasets,
            "n_merge_configs": n_merges,
            "db_path": "(session only)",
            "db_exists": True,
            "db_size_kb": 0,
        }

    def reset_all_data(self) -> bool:
        st.session_state.sp_projects = {}
        st.session_state.sp_counter_project = 0
        st.session_state.sp_counter_dataset = 0
        st.session_state.sp_counter_merge_config = 0
        return True


# ── Singleton ────────────────────────────────────────────────────────

def get_project_manager() -> SessionProjectManager:
    """Get or create the session project manager."""
    if "sp_manager" not in st.session_state:
        st.session_state.sp_manager = SessionProjectManager()
    return st.session_state.sp_manager

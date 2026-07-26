"""
Session save/load for Tabular ML Lab.

SECURITY MODEL
==============
This module never deserializes executable content. User-supplied session
bytes are parsed only as ZIP archives containing JSON (via json.loads) and
Parquet (via pandas.read_parquet). No pickle, joblib, dill, cloudpickle,
or any code-bearing format is ever loaded from user input.

Trained models, fitted pipelines, and split arrays are NOT persisted. The
user re-runs Page 06 (Train) to regenerate these from the saved raw data
and configuration. This is by design: it eliminates any need to load
arbitrary Python objects from user-supplied bytes, and it enforces the
app's reproducibility contract (retraining from saved config must
reproduce the original result).

Save format
-----------
File extension: .tmllab  (a ZIP archive)

Members:
    manifest.json              -- schema version, timestamps, app version
    config.json                -- dataclasses, preferences, methodology log
    widget_state.json          -- safe widget keys (NO API keys)
    coaching.json              -- InsightLedger.to_list()
    provenance.json            -- WorkflowProvenance.to_dict() (if present)
    lockbox.json               -- the sealed test-set labels + fraction/seed
                                  (schema >= 2.1; restoring the exact holdout
                                  keeps results comparable across sessions)
    coach_probe.json           -- ProbeResult numbers (schema >= 2.1)
    data/<name>.parquet        -- DataFrames, one per entry
    datasets/<id>.parquet      -- entries from datasets_registry
    datasets/index.json        -- name mapping for datasets_registry

Rejected on load:
    - Anything that is not a valid ZIP
    - Archives >100 MB uploaded or >500 MB uncompressed
    - Archives with >50 members (sanity cap)
    - Archive members with absolute paths or "../" in name
    - Archives whose manifest.json schema_version is unknown
    - Legacy .pkl files (detected by magic byte 0x80) -- error with migration note
"""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Set, Tuple

import pandas as pd
import streamlit as st


SAVE_SCHEMA_VERSION = "2.1"
# Older schemas this app still loads. 2.0 files simply lack the newer members
# (lockbox.json, coach_probe.json); everything present is restored as before.
_ACCEPTED_SCHEMA_VERSIONS = {"2.0", "2.1"}
SAVE_EXTENSION = "tmllab"

# Upper bounds on input size -- zip-bomb defense-in-depth beyond Streamlit's
# own maxUploadSize. Set generously above the typical 5-50 MB real session.
_MAX_UPLOAD_BYTES = 100 * 1024 * 1024       # 100 MB over-the-wire
_MAX_EXTRACTED_BYTES = 500 * 1024 * 1024    # 500 MB uncompressed total
_MAX_MEMBERS = 64                           # hard cap on archive entries

# -- Session-state key taxonomy --------------------------------------------
# Keys are sorted into exactly one of these buckets. Anything not listed is
# dropped on save (not persisted) and ignored on load (not restored).

_DATAFRAME_KEYS: Tuple[str, ...] = (
    "raw_data",
    "df_engineered",
    "working_table",
    "filtered_data",
)

# Dataclass keys with their fully-qualified constructor path.
# On load, we re-import and instantiate via from_dict semantics.
_DATACLASS_KEYS: Dict[str, Tuple[str, str]] = {
    "data_config":               ("utils.session_state", "DataConfig"),
    "split_config":              ("utils.session_state", "SplitConfig"),
    "model_config":              ("utils.session_state", "ModelConfig"),
    "task_type_detection":       ("utils.session_state", "TaskTypeDetection"),
    "cohort_structure_detection":("utils.session_state", "CohortStructureDetection"),
}

# Plain JSON-safe keys (primitives, lists, dicts of JSON-safe content).
_PLAIN_KEYS: Tuple[str, ...] = (
    "task_mode",
    "merge_steps",
    "last_merge_columns",
    "feature_engineering_applied",
    "engineered_feature_names",
    "selected_features",
    "preprocessing_config",
    "preprocessing_config_by_model",
    "use_cv",
    "cv_folds",
    "random_seed",
    "data_source",
    "data_filename",
    "dataset_id",
    "dataset_history",
    "has_completed_tour",
    "show_guided_tour",
    "workflow_mode",
    "methodology_log",
    "current_page",
    # -- schema 2.1 additions: recipe state a restore previously lost --
    "test_lockbox_fraction",       # non-default holdout % must survive restore
    "pre_fe_feature_cols",         # lets FE Reset/Skip restore the original list
    "preprocess_built_model_keys", # which models had pipelines configured
    "engineered_feature_transforms",  # double-transform guard's map
)

# Deferred widget keys -- stored separately because Streamlit requires setting
# widget keys via _pending_widget_state_restore on the next rerun.
# SECURITY: API key fields are intentionally absent. Keys must never travel
# with a save file, even if the file never leaves the user's machine.
_SAFE_WIDGET_KEYS: Set[str] = {
    "llm_backend",
    "ollama_model",
    "openai_model",
    "anthropic_model",
    "workflow_mode_selector",
    # Widget-keyed checkbox on Upload & Audit; restored via the deferred
    # mechanism so we never assign an already-instantiated widget's key.
    "exploratory_mode",
}

_PENDING_WIDGET_RESTORE_KEY = "_pending_widget_state_restore"

# Keys that are explicitly dropped on save. Some are unsafe (sklearn/torch
# pickles); others are ephemeral (derived metrics, UI caches).
_NEVER_PERSIST: Set[str] = {
    # Trained artifacts -- regenerated by re-running Page 06
    "trained_models", "fitted_estimators",
    "fitted_preprocessing_pipelines", "preprocessing_pipelines_by_model",
    "preprocessing_pipeline",
    # Splits -- regenerated from raw_data + split_config
    "X_train", "X_val", "X_test", "y_train", "y_val", "y_test",
    "feature_names", "feature_names_by_model",
    "train_indices", "val_indices", "test_indices", "target_label_encoder",
    # Derived metrics and evaluations
    "model_results", "cv_results",
    "permutation_importance", "partial_dependence", "explainability_robustness",
    "eda_results", "eda_insights",
    # Profiles & tables -- recomputed on Page 02 / Page 09
    "dataset_profile", "table1_df", "table1_metadata",
    "table1_custom_test_footnotes",
    # Export artifacts -- regenerated on Page 10
    "methods_section", "flow_diagram", "tripod_tracker", "latex_report",
    "report_data", "report_best_model", "report_model_selection",
    "report_explain_selection", "report_include_results", "report_include_llm",
    "shap_results", "shap_matplotlib_figs",
    "hypothesis_test_results", "sensitivity_seed_results",
    # Privacy-sensitive
    "openai_api_key", "anthropic_api_key",
}


# ---------------------------------------------------------------------------
# Save path
# ---------------------------------------------------------------------------

def _df_to_parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def _encode_dataclass(obj: Any) -> Dict[str, Any]:
    """Convert a dataclass instance to a JSON-safe dict via asdict()."""
    return asdict(obj) if is_dataclass(obj) else dict(obj)


def _json_safe(obj: Any) -> Any:
    """Coerce a value to something json.dumps can handle.

    Unknown objects become strings so a single non-JSON-safe item doesn't
    block the entire save.
    """
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        pass
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return str(obj)


def _build_archive_members() -> Dict[str, bytes]:
    """Build {filename: bytes} for every archive member to write.

    Only reads from st.session_state. Does not mutate any state.
    """
    members: Dict[str, bytes] = {}
    saved_keys: list = []
    skipped_keys: list = []

    # --- DataFrames ---
    for key in _DATAFRAME_KEYS:
        value = st.session_state.get(key)
        if isinstance(value, pd.DataFrame) and len(value) > 0:
            members[f"data/{key}.parquet"] = _df_to_parquet_bytes(value)
            saved_keys.append(key)

    # --- datasets_registry (Dict[Any, DataFrame]) ---
    registry = st.session_state.get("datasets_registry") or {}
    if isinstance(registry, dict) and registry:
        index = {}
        for i, (dataset_id, df) in enumerate(registry.items()):
            if isinstance(df, pd.DataFrame) and len(df) > 0:
                member_name = f"datasets/entry_{i}.parquet"
                members[member_name] = _df_to_parquet_bytes(df)
                index[str(dataset_id)] = member_name
        if index:
            members["datasets/index.json"] = json.dumps(index, indent=2).encode("utf-8")
            saved_keys.append("datasets_registry")

    # --- Dataclass configs ---
    config: Dict[str, Any] = {}
    for key in _DATACLASS_KEYS:
        value = st.session_state.get(key)
        if value is None:
            continue
        try:
            config[key] = _encode_dataclass(value)
            saved_keys.append(key)
        except Exception:
            skipped_keys.append(key)

    # --- Plain config keys ---
    for key in _PLAIN_KEYS:
        if key not in st.session_state:
            continue
        value = st.session_state[key]
        config[key] = _json_safe(value)
        saved_keys.append(key)

    members["config.json"] = json.dumps(config, indent=2, default=str).encode("utf-8")

    # --- Safe widget state (LLM model selection, but NOT API keys) ---
    widget_state: Dict[str, Any] = {}
    for key in _SAFE_WIDGET_KEYS:
        if key in st.session_state:
            widget_state[key] = _json_safe(st.session_state[key])
    members["widget_state.json"] = json.dumps(widget_state, indent=2).encode("utf-8")

    # --- Coaching ledger ---
    ledger = st.session_state.get("insight_ledger")
    if ledger is not None and hasattr(ledger, "to_list"):
        try:
            members["coaching.json"] = json.dumps(
                ledger.to_list(), indent=2, default=str
            ).encode("utf-8")
            saved_keys.append("insight_ledger")
        except Exception:
            skipped_keys.append("insight_ledger")

    # --- Workflow provenance ---
    provenance = st.session_state.get("workflow_provenance")
    if provenance is not None and hasattr(provenance, "to_dict"):
        try:
            members["provenance.json"] = json.dumps(
                provenance.to_dict(), indent=2, default=str
            ).encode("utf-8")
            saved_keys.append("workflow_provenance")
        except Exception:
            skipped_keys.append("workflow_provenance")

    def _json_metric(x: Any) -> Any:
        """Keep numbers numeric; stringify only what JSON cannot hold."""
        if isinstance(x, bool):
            return bool(x)
        if isinstance(x, (int, float)):
            return x
        if hasattr(x, "item"):
            v = x.item()
            return v if isinstance(v, (int, float, bool)) else str(v)
        return str(x)

    def _json_scalar(x: Any) -> Any:
        if isinstance(x, bool):
            return str(x)
        if isinstance(x, int):
            return x
        if hasattr(x, "item"):              # numpy scalar -> python scalar
            v = x.item()
            return v if isinstance(v, int) else str(v)
        return str(x)

    # --- Test-set lockbox (schema 2.1) ---
    # The sealed holdout must survive a save/restore VERBATIM: re-drawing it
    # with a lost non-default fraction would silently change the test set that
    # every prior result was scored on. Labels are coerced to plain int/str
    # here because json.dumps(default=str) would turn numpy ints into strings,
    # and string labels would no longer match the dataframe's integer index on
    # restore (a silent all-False membership).
    lockbox = st.session_state.get("test_lockbox")
    if isinstance(lockbox, dict) and lockbox.get("labels") is not None:
        try:
            def _plain_label(x: Any) -> Any:
                if isinstance(x, bool):
                    return str(x)
                if isinstance(x, int):
                    return x
                if hasattr(x, "item"):          # numpy scalar -> python scalar
                    v = x.item()
                    return v if isinstance(v, int) else str(v)
                return str(x)

            encoded = {
                "labels": [_plain_label(lbl) for lbl in lockbox["labels"]],
                "fraction": float(lockbox.get("fraction", 0.15)),
                "seed": int(lockbox.get("seed", 42)),
                "n_total": int(lockbox.get("n_total", 0)),
                "n_test": int(lockbox.get("n_test", len(lockbox["labels"]))),
                "signature": str(lockbox.get("signature", "")),
                "stratified": bool(lockbox.get("stratified", False)),
            }
            members["lockbox.json"] = json.dumps(encoded, indent=2).encode("utf-8")
            saved_keys.append("test_lockbox")
        except Exception:
            skipped_keys.append("test_lockbox")

    # --- Active cohort run (schema 2.1) ---
    # A run is "same question, different people", and the people are held as
    # index labels. Dropping it on restore does not merely lose a setting: the
    # restored session silently reports the WHOLE study to a researcher who
    # saved a one-group analysis, and the banked comparison runs go with it.
    run = st.session_state.get("cohort_run")
    if isinstance(run, dict) and run.get("labels"):
        try:
            members["cohort.json"] = json.dumps({
                "column": str(run["column"]),
                "value": _json_scalar(run.get("value")),
                "label": str(run["label"]),
                "labels": [_json_scalar(x) for x in run["labels"]],
                "n_rows": int(run.get("n_rows", len(run["labels"]))),
                "n_total": int(run.get("n_total", 0)),
                "position": int(run.get("position", 1)),
                "of": int(run.get("of", 1)),
                "order": [str(x) for x in (run.get("order") or [])],
                "target_col": str(run.get("target_col") or ""),
                "dropped_features": [str(x) for x in (run.get("dropped_features") or [])],
            }, indent=2).encode("utf-8")
            saved_keys.append("cohort_run")
        except Exception:
            skipped_keys.append("cohort_run")

    # Banked comparison runs. Without these the restored session can show one
    # group's result with nothing to compare it to, and the researcher re-runs
    # a cohort they had already finished.
    # Read from session_manager's own view of state rather than through
    # utils.cohorts, which holds its own streamlit reference — the saver must
    # serialize the state it is actually looking at.
    _runs = [r for r in (st.session_state.get("cohort_runs_done") or [])
             if is_dataclass(r)]
    if _runs:
        try:
            members["cohort_runs.json"] = json.dumps([{
                "column": r.column, "label": r.label,
                "n_train": int(r.n_train), "n_test": int(r.n_test),
                "dropped_features": list(r.dropped_features),
                "completed": bool(r.completed),
                # NOT _json_scalar: that coerces to str to keep index LABELS
                # matching an integer index, and a metric must stay numeric or
                # the comparison table formats "0.71" as text and f"{v:.3f}"
                # raises on it.
                "metrics": {str(k): _json_metric(v) for k, v in r.metrics.items()},
                "target_col": r.target_col, "task_type": r.task_type,
                "data_fingerprint": str(r.data_fingerprint),
            } for r in _runs], indent=2).encode("utf-8")
            saved_keys.append("cohort_runs_done")
        except Exception:
            skipped_keys.append("cohort_runs_done")

    # The engineering recipe. Without it a restored session's cohort switch
    # reverts to dropping every engineered feature while the button still
    # promises a rebuild.
    _steps = [x for x in (st.session_state.get("fe_recipe") or []) if is_dataclass(x)]
    if _steps:
        try:
            members["fe_recipe.json"] = json.dumps([{
                "kind": x.kind, "params": x.params,
                "produced": list(x.produced), "mode": x.mode,
            } for x in _steps], indent=2, default=str).encode("utf-8")
            saved_keys.append("fe_recipe")
        except Exception:
            skipped_keys.append("fe_recipe")

    # --- Coach evidence probe (schema 2.1) ---
    probe = st.session_state.get("coach_probe_result")
    if probe is not None and is_dataclass(probe):
        try:
            members["coach_probe.json"] = json.dumps(
                _json_safe(asdict(probe)), indent=2
            ).encode("utf-8")
            saved_keys.append("coach_probe_result")
        except Exception:
            skipped_keys.append("coach_probe_result")

    # --- Manifest (written last so it can reference what landed) ---
    manifest = {
        "schema_version": SAVE_SCHEMA_VERSION,
        "saved_at": datetime.now().isoformat(),
        "workflow_step": str(st.session_state.get("current_page", "Unknown")),
        "saved_keys": sorted(set(saved_keys)),
        "skipped_keys": sorted(set(skipped_keys)),
        "members": sorted(members.keys()),
    }
    members["manifest.json"] = json.dumps(manifest, indent=2).encode("utf-8")

    return members


def _pack_zip(members: Dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Write manifest first so readers can stream-validate it.
        ordered = sorted(members.keys(), key=lambda n: (n != "manifest.json", n))
        for name in ordered:
            zf.writestr(name, members[name])
    return buf.getvalue()


def _collect_session_data() -> Tuple[bytes, Dict[str, Any]]:
    """Build the archive. Returns (bytes, manifest_dict)."""
    members = _build_archive_members()
    manifest = json.loads(members["manifest.json"].decode("utf-8"))
    return _pack_zip(members), manifest


# ---------------------------------------------------------------------------
# Load path
# ---------------------------------------------------------------------------

class SessionLoadError(Exception):
    """Raised when an uploaded session file is invalid or unsafe."""


def _looks_like_pickle(head: bytes) -> bool:
    """Heuristic: legacy pickle files start with opcode 0x80 (PROTO).

    Used only to show a helpful migration error, never to trigger any
    pickle machinery.
    """
    return len(head) >= 2 and head[0] == 0x80 and head[1] in (0x02, 0x03, 0x04, 0x05)


def _validate_zip(zf: zipfile.ZipFile) -> None:
    """Fail fast on path traversal, zip bombs, or sketchy member counts."""
    infos = zf.infolist()
    if len(infos) > _MAX_MEMBERS:
        raise SessionLoadError(
            f"Archive has {len(infos)} members (max {_MAX_MEMBERS})."
        )
    total_uncompressed = 0
    for info in infos:
        name = info.filename
        # Reject absolute paths, parent-dir traversal, and NUL bytes.
        if name.startswith("/") or ".." in name.replace("\\", "/").split("/") or "\x00" in name:
            raise SessionLoadError(f"Unsafe member path: {name!r}")
        if info.file_size < 0:
            raise SessionLoadError(f"Member {name!r} reports negative size.")
        total_uncompressed += info.file_size
        if total_uncompressed > _MAX_EXTRACTED_BYTES:
            raise SessionLoadError(
                "Archive uncompressed size exceeds safety cap "
                f"({_MAX_EXTRACTED_BYTES // (1024 * 1024)} MB)."
            )


def _read_json(zf: zipfile.ZipFile, name: str) -> Any:
    with zf.open(name) as f:
        return json.loads(f.read().decode("utf-8"))


def _read_parquet(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    with zf.open(name) as f:
        return pd.read_parquet(io.BytesIO(f.read()))


def _reconstruct_dataclass(key: str, data: Dict[str, Any]) -> Any:
    """Rebuild a known dataclass from a dict, ignoring unknown keys."""
    module_name, class_name = _DATACLASS_KEYS[key]
    import importlib
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    valid = set(cls.__dataclass_fields__.keys())
    filtered = {k: v for k, v in data.items() if k in valid}
    return cls(**filtered)


def _clear_downstream_state() -> None:
    """Clear derived/trained state so a restored session starts clean.

    Called before applying restored values. Uses existing reset helper
    then also removes any of our never-persist keys that weren't covered.
    """
    try:
        from utils.session_state import reset_data_dependent_state
        reset_data_dependent_state()
    except Exception:
        pass
    for key in _NEVER_PERSIST:
        st.session_state.pop(key, None)
    # Invalidation bookkeeping from the pre-load session must not survive into
    # the restored one: a stale fingerprint could trigger (or suppress) a
    # spurious downstream reset on the next set_data call.
    st.session_state.pop("_raw_data_fingerprint", None)   # recomputed below
    st.session_state.pop("_working_table_source_id", None)


def _restore_session_data(archive_bytes: bytes) -> Tuple[int, Dict[str, Any], list]:
    """Parse archive, validate, apply to st.session_state.

    Returns (restored_count, manifest, warnings) — warnings is a list of
    human-readable strings for members that were present but could not be
    restored (surfaced in the sidebar; a partial restore must never be
    silent).
    Raises SessionLoadError on any validation failure.
    """
    if len(archive_bytes) > _MAX_UPLOAD_BYTES:
        raise SessionLoadError(
            f"File too large ({len(archive_bytes) // (1024 * 1024)} MB; "
            f"max {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB)."
        )

    head = archive_bytes[:4]
    if _looks_like_pickle(head):
        raise SessionLoadError(
            "This looks like an old .pkl session from a previous app version. "
            "For security reasons the app no longer loads pickle files. "
            "Re-run your analysis and use Save Progress to create a new "
            f".{SAVE_EXTENSION} file."
        )

    try:
        zf = zipfile.ZipFile(io.BytesIO(archive_bytes))
    except zipfile.BadZipFile as exc:
        raise SessionLoadError(f"Not a valid session archive: {exc}") from exc

    with zf:
        _validate_zip(zf)
        names = set(zf.namelist())

        if "manifest.json" not in names:
            raise SessionLoadError("Archive is missing manifest.json.")
        manifest = _read_json(zf, "manifest.json")
        version = str(manifest.get("schema_version", ""))
        if version not in _ACCEPTED_SCHEMA_VERSIONS:
            raise SessionLoadError(
                f"Unsupported session schema version {version!r} "
                f"(this app writes {SAVE_SCHEMA_VERSION!r} and accepts "
                f"{sorted(_ACCEPTED_SCHEMA_VERSIONS)})."
            )

        # Clear derived/trained state up front.
        _clear_downstream_state()
        restored = 0
        warnings: list = []

        # --- DataFrames ---
        for key in _DATAFRAME_KEYS:
            member = f"data/{key}.parquet"
            if member in names:
                st.session_state[key] = _read_parquet(zf, member)
                restored += 1
        # Banked cohort runs are keyed on the data fingerprint, so it has to be
        # rebuilt here or every restored run reads as belonging to other data
        # and is filtered out of the comparison table.
        if st.session_state.get("raw_data") is not None:
            try:
                from utils.session_state import _content_fingerprint
                st.session_state["_raw_data_fingerprint"] = _content_fingerprint(
                    st.session_state["raw_data"])
            except Exception:
                pass

        # --- datasets_registry ---
        if "datasets/index.json" in names:
            index = _read_json(zf, "datasets/index.json")
            registry: Dict[str, pd.DataFrame] = {}
            if isinstance(index, dict):
                for dataset_id, member in index.items():
                    # Defense-in-depth: only read members that actually exist
                    # and are inside the datasets/ prefix.
                    if (
                        isinstance(member, str)
                        and member.startswith("datasets/")
                        and member in names
                    ):
                        registry[str(dataset_id)] = _read_parquet(zf, member)
            if registry:
                st.session_state["datasets_registry"] = registry
                restored += 1

        # --- Config (dataclasses + plain) ---
        if "config.json" in names:
            config = _read_json(zf, "config.json")
            if isinstance(config, dict):
                for key, value in config.items():
                    if key in _DATACLASS_KEYS and isinstance(value, dict):
                        try:
                            st.session_state[key] = _reconstruct_dataclass(key, value)
                            restored += 1
                        except Exception:
                            pass
                    elif key in _PLAIN_KEYS:
                        st.session_state[key] = value
                        restored += 1
                    # unknown keys are silently ignored

        # --- Widget state (deferred) ---
        if "widget_state.json" in names:
            widgets = _read_json(zf, "widget_state.json")
            if isinstance(widgets, dict):
                safe_widgets = {
                    k: v for k, v in widgets.items() if k in _SAFE_WIDGET_KEYS
                }
                if safe_widgets:
                    st.session_state[_PENDING_WIDGET_RESTORE_KEY] = safe_widgets
                    restored += len(safe_widgets)

        # --- Coaching ledger ---
        if "coaching.json" in names:
            items = _read_json(zf, "coaching.json")
            if isinstance(items, list):
                from utils.insight_ledger import InsightLedger
                st.session_state["insight_ledger"] = InsightLedger.from_list(items)
                restored += 1

        # --- Workflow provenance ---
        if "provenance.json" in names:
            data = _read_json(zf, "provenance.json")
            if isinstance(data, dict):
                try:
                    from utils.workflow_provenance import WorkflowProvenance
                    st.session_state["workflow_provenance"] = WorkflowProvenance.from_dict(data)
                    restored += 1
                except Exception:
                    warnings.append(
                        "Workflow record could not be restored — the manuscript "
                        "compiler will start from what you re-run."
                    )

        # --- Active cohort run (absent in files saved before cohort runs) ---
        if "cohort.json" in names:
            try:
                data = _read_json(zf, "cohort.json")
                labels = data.get("labels")
                if not isinstance(labels, list) or not labels:
                    raise ValueError("cohort has no rows")
                st.session_state["cohort_run"] = {
                    "column": str(data["column"]),
                    "value": data.get("value"),
                    "label": str(data.get("label", "")),
                    "labels": list(labels),
                    "n_rows": int(data.get("n_rows", len(labels))),
                    "n_total": int(data.get("n_total", 0)),
                    "position": int(data.get("position", 1)),
                    "of": int(data.get("of", 1)),
                    "order": list(data.get("order") or []),
                    "target_col": str(data.get("target_col") or ""),
                    "dropped_features": list(data.get("dropped_features") or []),
                }
                restored += 1
            except Exception as exc:
                warnings.append(
                    f"The saved one-group run could not be restored ({exc}). "
                    "This session now covers the WHOLE study — re-select your "
                    "group on Upload & Audit before reading any result."
                )

        if "cohort_runs.json" in names:
            try:
                from utils.cohorts import CohortRun
                st.session_state["cohort_runs_done"] = [
                    CohortRun(
                        column=str(d["column"]), label=str(d["label"]),
                        n_train=int(d.get("n_train", 0)),
                        n_test=int(d.get("n_test", 0)),
                        dropped_features=list(d.get("dropped_features") or []),
                        completed=bool(d.get("completed", True)),
                        metrics=dict(d.get("metrics") or {}),
                        target_col=str(d.get("target_col") or ""),
                        task_type=str(d.get("task_type") or ""),
                        data_fingerprint=str(d.get("data_fingerprint") or ""),
                    )
                    for d in _read_json(zf, "cohort_runs.json")
                ]
                restored += 1
            except Exception as exc:
                warnings.append(
                    f"Saved comparison runs could not be restored ({exc}). "
                    "Any group you had already analyzed will need re-running."
                )

        if "fe_recipe.json" in names:
            try:
                from utils.replay import Step
                st.session_state["fe_recipe"] = [
                    Step(kind=str(d["kind"]), params=dict(d.get("params") or {}),
                         produced=list(d.get("produced") or []),
                         mode=str(d.get("mode") or "pure"))
                    for d in _read_json(zf, "fe_recipe.json")
                ]
                restored += 1
            except Exception as exc:
                warnings.append(
                    f"The feature-engineering recipe could not be restored "
                    f"({exc}). Switching cohorts will not rebuild your "
                    f"engineered features until you recreate them.")

        # --- Test-set lockbox (schema 2.1; absent in 2.0 files) ---
        if "lockbox.json" in names:
            try:
                data = _read_json(zf, "lockbox.json")
                labels = data.get("labels")
                if not isinstance(labels, list) or not labels:
                    raise ValueError("lockbox has no labels")
                st.session_state["test_lockbox"] = {
                    "labels": list(labels),
                    "fraction": float(data.get("fraction", 0.15)),
                    "seed": int(data.get("seed", 42)),
                    "n_total": int(data.get("n_total", 0)),
                    "n_test": int(data.get("n_test", len(labels))),
                    "signature": str(data.get("signature", "")),
                    "stratified": bool(data.get("stratified", False)),
                }
                # Keep the fraction key coherent with the restored lockbox even
                # if config.json predates it or disagrees.
                st.session_state["test_lockbox_fraction"] = float(
                    data.get("fraction", 0.15)
                )
                restored += 1
            except Exception as exc:
                warnings.append(
                    f"Sealed test set could not be restored ({exc}) — it will "
                    "be re-drawn from the saved seed and fraction on Upload & "
                    "Audit."
                )

        # --- Coach evidence probe (schema 2.1) ---
        if "coach_probe.json" in names:
            try:
                data = _read_json(zf, "coach_probe.json")
                from ml.coach_probe import ProbeResult
                valid = set(ProbeResult.__dataclass_fields__.keys())
                st.session_state["coach_probe_result"] = ProbeResult(
                    **{k: v for k, v in data.items() if k in valid}
                )
                restored += 1
            except Exception:
                warnings.append(
                    "Coach probe evidence could not be restored — re-run the "
                    "evidence probe on Preprocess (~10 s)."
                )

    return restored, manifest, warnings


# ---------------------------------------------------------------------------
# Public UI
# ---------------------------------------------------------------------------

def _format_size(n_bytes: int) -> str:
    if n_bytes < 1024:
        return f"{n_bytes} B"
    if n_bytes < 1024 * 1024:
        return f"{n_bytes / 1024:.1f} KB"
    return f"{n_bytes / (1024 * 1024):.1f} MB"


def render_session_controls() -> None:
    """Render save/load session controls in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💾 Session Management")
    st.sidebar.markdown("""
    <style>
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button,
    section[data-testid="stSidebar"] div[data-testid="stDownloadButton"] > button,
    section[data-testid="stSidebar"] button[kind],
    section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],
    section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"],
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background: #f8fafc !important;
        border: 1px solid rgba(148, 163, 184, 0.55) !important;
        color: #0f172a !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button *,
    section[data-testid="stSidebar"] div[data-testid="stDownloadButton"] > button *,
    section[data-testid="stSidebar"] button[kind] *,
    section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] *,
    section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] *,
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] *,
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small,
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] span,
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] p,
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] label,
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] div {
        color: #0f172a !important;
        fill: #0f172a !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover,
    section[data-testid="stSidebar"] div[data-testid="stDownloadButton"] > button:hover,
    section[data-testid="stSidebar"] button[kind]:hover,
    section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover,
    section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover,
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]:hover {
        background: #e2e8f0 !important;
        color: #020617 !important;
        border-color: rgba(100, 116, 139, 0.75) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    restore_notice = st.session_state.pop("session_restore_notice", None)
    if restore_notice:
        saved_at = restore_notice.get("saved_at", "Unknown")
        saved_date = saved_at[:10] if saved_at != "Unknown" else "Unknown"
        workflow_step = restore_notice.get("workflow_step", "Unknown")
        restored_count = restore_notice.get("restored_count", 0)

        # Resume checklist: say exactly what came back and which clicks
        # regenerate the rest. Results are re-computed, never deserialized —
        # that is the format's safety contract.
        lb = st.session_state.get("test_lockbox") or {}
        lockbox_line = (
            f"- 🔒 **Test set restored:** the same {lb.get('fraction', 0):.0%} "
            f"holdout (n={lb.get('n_test', 0)}) — results stay comparable\n"
            if lb.get("labels") else
            "- 🔒 Test set will be re-drawn from the saved seed on Upload & Audit\n"
        )
        built = st.session_state.get("preprocess_built_model_keys") or []
        built_line = (
            f"- ⚙️ Pipeline configs ready for: **{', '.join(map(str, built))}**\n"
            if built else ""
        )
        st.sidebar.success(
            f"✅ **Session Restored** (saved {saved_date}, "
            f"{restored_count} items, last step: {workflow_step})\n\n"
            f"{lockbox_line}"
            f"{built_line}"
            f"\n**To regenerate results (recomputed, not deserialized):**\n"
            f"1. **Preprocess** → Build Pipelines (settings pre-filled)\n"
            f"2. **Train & Compare** → Prepare Splits → Train Models\n"
        )
        for w in restore_notice.get("warnings", []):
            st.sidebar.warning(f"⚠️ {w}")

    # --- Save ---
    if st.sidebar.button("📥 Save Progress", help="Download your current workflow state"):
        try:
            archive_bytes, manifest = _collect_session_data()
            if not manifest.get("saved_keys"):
                st.sidebar.warning("⚠️ No session data to save yet. Start your analysis first!")
                return

            size_str = _format_size(len(archive_bytes))
            if len(archive_bytes) > _MAX_UPLOAD_BYTES:
                st.sidebar.warning(
                    f"⚠️ Session is very large ({size_str}). "
                    "Consider completing your analysis before saving."
                )

            filename = f"tabular_ml_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{SAVE_EXTENSION}"
            st.sidebar.download_button(
                label=f"⬇️ Download Session ({size_str})",
                data=archive_bytes,
                file_name=filename,
                mime="application/zip",
                help="Save this file to resume your work later",
                key="download_session_button",
            )
            skipped = manifest.get("skipped_keys", [])
            success_msg = (
                f"✅ **Session Ready for Download!**\n\n"
                f"- **Items saved:** {len(manifest.get('saved_keys', []))}\n"
                f"- **File size:** {size_str}\n"
                f"- **Current step:** {manifest.get('workflow_step', 'Unknown')}"
            )
            if skipped:
                success_msg += f"\n- **Note:** {len(skipped)} items skipped (non-serializable)"
            st.sidebar.success(success_msg)
        except Exception as exc:
            st.sidebar.error(f"❌ **Error saving session:**\n\n{exc}")

    # --- Load ---
    st.sidebar.markdown("**Or resume previous session:**")
    uploader_nonce = st.session_state.get("upload_session_file_nonce", 0)
    uploaded_session = st.sidebar.file_uploader(
        "📂 Upload Session File",
        type=[SAVE_EXTENSION, "zip"],
        help=f"Upload a .{SAVE_EXTENSION} file created by this app",
        key=f"upload_session_file_{uploader_nonce}",
    )

    if uploaded_session is not None:
        try:
            archive_bytes = uploaded_session.read()
            restored_count, manifest, load_warnings = _restore_session_data(archive_bytes)
            st.session_state["upload_session_file_nonce"] = uploader_nonce + 1
            st.session_state["session_restore_notice"] = {
                "saved_at": manifest.get("saved_at", "Unknown"),
                "restored_count": restored_count,
                "workflow_step": manifest.get("workflow_step", "Unknown"),
                "warnings": load_warnings,
            }
            st.rerun()
        except SessionLoadError as exc:
            st.sidebar.error(f"❌ **Could not load session:**\n\n{exc}")
        except Exception as exc:
            st.sidebar.error(f"❌ **Error loading session:**\n\n{exc}")

    # --- Footer ---
    st.sidebar.info(
        "⚠️ **Privacy Note**\n\n"
        "Session files contain your data and analysis results. "
        "Store them securely and do not share if data is sensitive. "
        "API keys are never included in saved files."
    )


def get_session_summary() -> Dict[str, Any]:
    """Summary of current session state for debugging."""
    return {
        "total_keys": len(st.session_state),
        "has_data": st.session_state.get("raw_data") is not None,
        "has_models": len(st.session_state.get("trained_models", {})) > 0,
        "has_provenance": "workflow_provenance" in st.session_state,
        "has_ledger": "insight_ledger" in st.session_state,
    }

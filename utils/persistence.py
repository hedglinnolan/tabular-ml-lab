"""
Data persistence: save/load DataFrames and models across sessions.

Stores data in a cache directory keyed by session ID.
"""
import os
import json
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

CACHE_DIR = Path(__file__).parent.parent / ".cache"


def _get_session_dir(session_id: Optional[str] = None) -> Path:
    """Get or create session cache directory."""
    if session_id is None:
        session_id = "default"
    session_dir = CACHE_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def save_dataframe(df: pd.DataFrame, name: str, session_id: Optional[str] = None) -> str:
    """Save DataFrame as Parquet.

    Returns the path to the saved file.
    """
    session_dir = _get_session_dir(session_id)
    path = session_dir / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return str(path)


def load_dataframe(name: str, session_id: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Load DataFrame from Parquet cache.

    Returns None if not found.
    """
    session_dir = _get_session_dir(session_id)
    path = session_dir / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


def save_model(model, name: str, session_id: Optional[str] = None) -> str:
    """Save a trained model using joblib.

    Returns path to saved model.
    """
    import joblib
    session_dir = _get_session_dir(session_id)
    path = session_dir / f"{name}.joblib"
    joblib.dump(model, path)
    return str(path)


def load_model(name: str, session_id: Optional[str] = None):
    """Load a trained model from joblib cache.

    Returns None if not found.
    """
    import joblib
    session_dir = _get_session_dir(session_id)
    path = session_dir / f"{name}.joblib"
    if path.exists():
        return joblib.load(path)
    return None


def save_session_metadata(metadata: Dict[str, Any], session_id: Optional[str] = None):
    """Save session metadata (config, choices, etc.)."""
    session_dir = _get_session_dir(session_id)
    path = session_dir / "metadata.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)


def load_session_metadata(session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load session metadata."""
    session_dir = _get_session_dir(session_id)
    path = session_dir / "metadata.json"
    if path.exists():
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return None


def list_cached_sessions() -> list:
    """List available cached sessions."""
    if not CACHE_DIR.exists():
        return []
    sessions = []
    for d in CACHE_DIR.iterdir():
        if d.is_dir():
            meta = load_session_metadata(d.name)
            sessions.append({
                "session_id": d.name,
                "metadata": meta,
                "files": [f.name for f in d.iterdir()],
            })
    return sessions


def data_hash(df: pd.DataFrame) -> str:
    """Compute a hash of a DataFrame for versioning."""
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df).values.tobytes())
    return h.hexdigest()[:12]


def generate_reproducibility_manifest(
    session_id: Optional[str] = None,
    random_seed: int = 42,
    data_df: Optional[pd.DataFrame] = None,
    model_configs: Optional[Dict] = None,
    preprocessing_config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Generate a reproducibility manifest with all software versions and configs."""
    import sys
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "random_seed": random_seed,
    }

    # Package versions
    packages = {}
    for pkg_name in ["streamlit", "torch", "sklearn", "numpy", "pandas", "scipy",
                     "plotly", "shap", "optuna", "statsmodels"]:
        try:
            mod = __import__(pkg_name)
            packages[pkg_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass
    # sklearn version
    try:
        import sklearn
        packages["scikit-learn"] = sklearn.__version__
    except ImportError:
        pass
    manifest["package_versions"] = packages

    if data_df is not None:
        manifest["data_hash"] = data_hash(data_df)
        manifest["data_shape"] = list(data_df.shape)

    if model_configs:
        manifest["model_configs"] = model_configs

    if preprocessing_config:
        manifest["preprocessing_config"] = preprocessing_config

    return manifest

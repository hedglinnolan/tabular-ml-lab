"""
Global random seed management.
Ensures reproducibility across numpy, sklearn, torch, pandas.
"""
import numpy as np
import random
try:                       # optional: only needed for the neural-network model
    import torch
except ImportError:        # pragma: no cover - exercised on lean installs
    torch = None
from typing import Optional


def set_global_seed(seed: int):
    """
    Set random seed for numpy, random, torch, sklearn.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Note: sklearn uses numpy's random state, so setting numpy seed covers it


def get_global_seed() -> int:
    """Get current global random seed from session state."""
    import streamlit as st
    return st.session_state.get('random_seed', 42)

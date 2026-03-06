"""
Table export utilities for Tabular ML Lab.
Provides native copy-to-clipboard for all tables with header support.
"""
import streamlit as st
import pandas as pd


def table(df: pd.DataFrame, use_container_width=True, hide_index=True, **kwargs):
    """
    Render a table with full copy support (including headers).
    
    Uses st.data_editor in disabled mode for superior copy/paste:
    - Select cells/columns → Cmd+C / Ctrl+C → paste in Excel with headers
    - Multi-column selection works properly
    - Headers are selectable and copy with data
    
    Args:
        df: DataFrame to display
        use_container_width: Expand to container width (default: True)
        hide_index: Hide the index column (default: True)
        **kwargs: Additional arguments passed to st.data_editor()
    
    Returns:
        The st.data_editor component
    """
    # Set defaults
    kwargs.setdefault('disabled', True)  # Read-only
    kwargs.setdefault('use_container_width', use_container_width)
    kwargs.setdefault('hide_index', hide_index)
    
    return st.data_editor(df, **kwargs)

"""
Table export utilities for Tabular ML Lab.
Provides easy copy-to-clipboard functionality for dataframes.
"""
import streamlit as st
import pandas as pd
from typing import Optional


def render_table_with_copy(df: pd.DataFrame, key: str, label: str = "Copy Table to Clipboard", **kwargs):
    """
    Render a dataframe with a copy-to-clipboard button that preserves headers.
    
    Args:
        df: DataFrame to display
        key: Unique key for the button
        label: Button label
        **kwargs: Additional arguments passed to st.dataframe()
    """
    # Display the table
    st.dataframe(df, **kwargs)
    
    # Convert to TSV (tab-separated) which Excel understands and preserves structure
    tsv_data = df.to_csv(sep='\t', index=False)
    
    # Add copy button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.download_button(
            label=f"📋 {label}",
            data=tsv_data,
            file_name=f"table_{key}.tsv",
            mime="text/tab-separated-values",
            key=f"copy_table_{key}",
            help="Download as TSV (opens in Excel with preserved formatting)",
            type="secondary"
        )

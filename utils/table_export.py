"""
Table export utilities for Tabular ML Lab.
Provides copy-to-clipboard for all tables with TSV download button.
"""
import streamlit as st
import pandas as pd


def table(df: pd.DataFrame, hide_index=True, **kwargs):
    """
    Render a table with downloadable export (TSV format with headers).
    
    Reality check: Streamlit doesn't provide Plotly-level native header copy.
    - st.dataframe: cells copyable, but headers aren't easily selectable
    - st.data_editor: still doesn't give full header selection when disabled
    
    Solution: Show table normally + provide TSV download button that users can import to Excel.
    
    Args:
        df: DataFrame to display
        use_container_width: Expand to container width (default: True)
        hide_index: Hide the index column (default: True)
        **kwargs: Additional arguments passed to st.dataframe()
    
    Returns:
        The st.dataframe component
    """
    # Map parameters for st.dataframe
    if use_container_width:
        kwargs['width'] = kwargs.pop('width', 'stretch')
    kwargs.pop('use_container_width', None)  # Remove if present
    
    # Handle hide_index
    if hide_index:
        kwargs['hide_index'] = True
    
    # Get key for download button (if provided)
    table_key = kwargs.pop('key', None)
    
    # Display the table
    result = st.dataframe(df, **kwargs)
    
    # Add download button for Excel export with headers
    if table_key:
        tsv_data = df.to_csv(sep='\t', index=False)
        st.download_button(
            label="📋 Download as TSV (opens in Excel)",
            data=tsv_data,
            file_name=f"{table_key}.tsv",
            mime="text/tab-separated-values",
            key=f"dl_{table_key}",
            help="Download table with headers (opens directly in Excel)",
            type="secondary",
            use_container_width=False
        )
    
    return result

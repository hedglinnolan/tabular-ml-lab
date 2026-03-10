"""
Session save/load functionality for Tabular ML Lab.
Allows users to download and resume their analysis workflow.
"""
import streamlit as st
import pickle
import io
from datetime import datetime
from typing import Dict, Any, Set
import sys


def _get_excluded_keys() -> Set[str]:
    """Get set of session state keys that should not be serialized."""
    return {
        # Streamlit internal keys
        '_uploaded_file_data',
        'FileUploader',
        'FormSubmitter',
        
        # Widget state (auto-managed by Streamlit)
        '_widget_manager',
        '_script_run_ctx',
        
        # Large non-serializable objects that may cause issues
        # Add more here if specific keys cause problems
    }


def _is_serializable(obj: Any) -> bool:
    """Test if an object can be pickled."""
    try:
        pickle.dumps(obj)
        return True
    except (TypeError, AttributeError, pickle.PicklingError):
        return False


def _collect_session_data() -> Dict[str, Any]:
    """Collect serializable session state data.
    
    Returns:
        Dictionary containing session data and metadata
    """
    exclude_keys = _get_excluded_keys()
    session_data = {}
    skipped_keys = []
    
    for key, value in st.session_state.items():
        # Skip internal keys, excluded keys
        if key.startswith('_') or key in exclude_keys:
            continue
        
        # Test if serializable
        if _is_serializable(value):
            session_data[key] = value
        else:
            skipped_keys.append(key)
    
    # Add metadata
    session_data['_metadata'] = {
        'saved_at': datetime.now().isoformat(),
        'version': '1.0',
        'workflow_step': st.session_state.get('current_page', 'Unknown'),
        'skipped_keys': skipped_keys,
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'session_keys_count': len(session_data),
    }
    
    return session_data


def _calculate_session_size(session_data: Dict[str, Any]) -> tuple[int, str]:
    """Calculate the size of serialized session data.
    
    Returns:
        Tuple of (bytes, human_readable_string)
    """
    serialized = pickle.dumps(session_data)
    size_bytes = len(serialized)
    
    # Human readable
    if size_bytes < 1024:
        size_str = f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        size_str = f"{size_bytes / 1024:.1f} KB"
    else:
        size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
    
    return size_bytes, size_str


def _restore_session_data(session_data: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    """Restore session data to st.session_state.
    
    Returns:
        Tuple of (restored_count, metadata)
    """
    restored_count = 0
    metadata = session_data.get('_metadata', {})
    
    for key, value in session_data.items():
        if key != '_metadata':
            st.session_state[key] = value
            restored_count += 1
    
    return restored_count, metadata


def render_session_controls():
    """Render save/load session controls in sidebar.
    
    This should be called in the sidebar of every page to make
    session management available throughout the workflow.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💾 Session Management")
    
    # ========================================================================
    # SAVE SESSION
    # ========================================================================
    if st.sidebar.button("📥 Save Progress", help="Download your current workflow state"):
        try:
            # Collect session state
            session_data = _collect_session_data()
            
            # Check if there's any meaningful data
            if len(session_data) <= 1:  # Only metadata
                st.sidebar.warning("⚠️ No session data to save yet. Start your analysis first!")
                return
            
            # Serialize
            session_bytes = pickle.dumps(session_data)
            size_bytes, size_str = _calculate_session_size(session_data)
            
            # Warn about large files
            if size_bytes > 50 * 1024 * 1024:  # > 50 MB
                st.sidebar.warning(f"""
                ⚠️ **Large Session File**
                
                Session size: {size_str}
                
                Large files may take time to download and upload.
                Consider completing your analysis before closing the browser.
                """)
            
            # Offer download
            st.sidebar.download_button(
                label=f"⬇️ Download Session ({size_str})",
                data=session_bytes,
                file_name=f"tabular_ml_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                mime="application/octet-stream",
                help="Save this file to resume your work later",
                key="download_session_button"
            )
            
            # Show success with details
            metadata = session_data['_metadata']
            skipped = metadata.get('skipped_keys', [])
            
            success_msg = f"""
            ✅ **Session Ready for Download!**
            
            - **Items saved:** {metadata['session_keys_count']}
            - **File size:** {size_str}
            - **Current step:** {metadata['workflow_step']}
            """
            
            if skipped:
                success_msg += f"\n- **Note:** {len(skipped)} non-serializable items skipped"
            
            st.sidebar.success(success_msg)
            
        except Exception as e:
            st.sidebar.error(f"❌ **Error saving session:**\n\n{str(e)}")
    
    # ========================================================================
    # LOAD SESSION
    # ========================================================================
    st.sidebar.markdown("**Or resume previous session:**")
    
    uploaded_session = st.sidebar.file_uploader(
        "📂 Upload Session File",
        type=['pkl'],
        help="Upload a previously saved session file",
        key="upload_session_file"
    )
    
    if uploaded_session is not None:
        try:
            # Load session data
            session_bytes = uploaded_session.read()
            session_data = pickle.loads(session_bytes)
            
            # Validate it's a session file
            if '_metadata' not in session_data:
                st.sidebar.error("❌ Invalid session file (missing metadata)")
                return
            
            # Check version compatibility
            metadata = session_data.get('_metadata', {})
            session_version = metadata.get('version', 'unknown')
            current_version = '1.0'  # Match what we save
            
            if session_version != current_version:
                st.sidebar.warning(f"""
                ⚠️ **Version Mismatch**
                
                Session was saved with version {session_version}, 
                but current version is {current_version}.
                
                Attempting to restore anyway, but some features may not work correctly.
                """)
            
            # Restore to session state
            restored_count, metadata = _restore_session_data(session_data)
            
            # Show success
            saved_at = metadata.get('saved_at', 'Unknown')
            saved_date = saved_at[:10] if saved_at != 'Unknown' else 'Unknown'
            workflow_step = metadata.get('workflow_step', 'Unknown')
            
            st.sidebar.success(f"""
            ✅ **Session Restored!**
            
            - **Saved:** {saved_date}
            - **Items restored:** {restored_count}
            - **Last step:** {workflow_step}
            
            Navigate to a page to continue your work.
            """)
            
        except pickle.UnpicklingError:
            st.sidebar.error("❌ **Invalid session file**\n\nFile is corrupted or not a valid session.")
        except Exception as e:
            st.sidebar.error(f"❌ **Error loading session:**\n\n{str(e)}")
    
    # ========================================================================
    # PRIVACY & SECURITY WARNINGS
    # ========================================================================
    st.sidebar.warning("""
    🔒 **Security Warning**
    
    Only load session files you created yourself.  
    **Never load files from untrusted sources.**
    """)
    
    st.sidebar.info("""
    ⚠️ **Privacy Note**
    
    Session files contain your data and analysis results.
    Store them securely and don't share if data is sensitive.
    """)


def get_session_summary() -> Dict[str, Any]:
    """Get summary of current session state for debugging.
    
    Returns:
        Dictionary with session statistics
    """
    exclude_keys = _get_excluded_keys()
    
    total_keys = len(st.session_state)
    serializable_keys = sum(
        1 for key in st.session_state
        if not key.startswith('_') 
        and key not in exclude_keys
        and _is_serializable(st.session_state[key])
    )
    
    return {
        'total_keys': total_keys,
        'serializable_keys': serializable_keys,
        'has_data': st.session_state.get('raw_data') is not None,
        'has_models': len(st.session_state.get('trained_models', {})) > 0,
    }

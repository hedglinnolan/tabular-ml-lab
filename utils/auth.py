"""
Reverse proxy authentication for enterprise deployment.

Expects authentication to be handled by upstream reverse proxy (nginx/Apache)
with Active Directory integration. The proxy passes authenticated username
via HTTP header (typically X-Remote-User).

This module verifies the header is present and extracts user information.
"""
import os
import streamlit as st
from typing import Optional


def is_auth_enabled() -> bool:
    """Check if authentication is enabled via environment variable."""
    return os.getenv("AUTH_ENABLED", "false").lower() in ("true", "1", "yes")


def get_auth_header_name() -> str:
    """Get the name of the authentication header from environment."""
    return os.getenv("AUTH_HEADER", "X-Remote-User")


def get_authenticated_user() -> Optional[str]:
    """
    Get authenticated username from reverse proxy header.
    
    Returns:
        Username string if authenticated, None if auth disabled or header missing
    """
    if not is_auth_enabled():
        return None
    
    # Streamlit doesn't expose request headers directly in community edition
    # We use a workaround via query params or session state
    # The reverse proxy should be configured to pass user as query param
    # OR the deployment should use Streamlit Enterprise with custom component
    
    # Method 1: Check session state (set by auth gate)
    if "authenticated_user" in st.session_state:
        return st.session_state["authenticated_user"]
    
    # Method 2: Check query params (fallback)
    # Proxy can be configured to redirect to /?user=<username>
    try:
        query_params = st.query_params
        if "user" in query_params:
            username = query_params["user"]
            st.session_state["authenticated_user"] = username
            return username
    except Exception:
        pass
    
    return None


def require_authentication():
    """
    Authentication gate - blocks access if auth enabled and no valid user.
    
    Call this at the top of app.py before rendering any content.
    """
    if not is_auth_enabled():
        # Auth disabled - allow access
        return
    
    user = get_authenticated_user()
    
    if not user:
        # No authenticated user - show error and stop
        st.error("🔒 Authentication Required")
        st.markdown("""
        **Access Denied**
        
        This application requires authentication through your institution's identity provider.
        
        If you reached this page directly:
        - You must access the application through the institutional portal
        - Your session may have expired - please log in again
        - Contact IT support if you continue to see this message
        """)
        st.stop()
    
    # User authenticated - store in session state for access by other pages
    st.session_state["authenticated_user"] = user


def get_current_user() -> str:
    """
    Get the current authenticated user's username.
    
    Returns:
        Username string, or "anonymous" if auth is disabled
    """
    if not is_auth_enabled():
        return "anonymous"
    
    user = get_authenticated_user()
    return user if user else "anonymous"


def show_user_info():
    """Display authenticated user info in sidebar (optional)."""
    if not is_auth_enabled():
        return
    
    user = get_current_user()
    if user != "anonymous":
        st.sidebar.markdown(f"👤 **Logged in as:** {user}")

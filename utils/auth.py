"""
KeyCloak OIDC authentication for West Point enterprise deployment.

Uses Streamlit's native st.login() with KeyCloak OIDC provider.
Requires Streamlit >= 1.42.0 and authlib >= 1.3.2.

KeyCloak endpoint: https://login.microsoftonline.com/99ff8811-3517-40a9-bf10-45ea0a321f0b/saml2
Configuration in .streamlit/secrets.toml
"""
import os
import streamlit as st
from typing import Optional, Dict


def is_auth_enabled() -> bool:
    """Check if authentication is enabled via environment variable."""
    return os.getenv("AUTH_ENABLED", "false").lower() in ("true", "1", "yes")


def get_authenticated_user() -> Optional[Dict[str, str]]:
    """
    Get authenticated user info from Streamlit OIDC session.
    
    Returns:
        Dict with user info (name, email, username) if authenticated, None otherwise
    """
    if not is_auth_enabled():
        return None
    
    user_info = getattr(st, "experimental_user", None) or getattr(st, "user", None)
    if user_info and getattr(user_info, "is_logged_in", False):
        email = getattr(user_info, "email", "") or ""
        name = getattr(user_info, "name", "") or email or "Unknown User"
        return {
            "username": email.split("@")[0] if email else "unknown",
            "email": email,
            "name": name,
        }
    
    return None


def require_authentication():
    """
    Authentication gate for the application.
    Shows login button if not authenticated, blocks access until user logs in.
    """
    if not is_auth_enabled():
        return  # Auth disabled, allow access
    
    user_info = getattr(st, "experimental_user", None) or getattr(st, "user", None)
    if not (user_info and getattr(user_info, "is_logged_in", False)):
        st.title("🔐 Tabular ML Lab - West Point")
        st.markdown("""
        ### Authentication Required
        
        This application requires West Point credentials via KeyCloak SSO.
        Click the button below to authenticate.
        """)
        
        # Streamlit's native OIDC login button
        if st.button("🔑 Login with West Point SSO", type="primary"):
            st.login("keycloak")
        
        st.stop()  # Block access until authenticated


def show_user_info():
    """Display authenticated user info in sidebar (optional)."""
    user = get_authenticated_user()
    if user:
        with st.sidebar:
            st.markdown("---")
            st.markdown(f"**Logged in as:** {user['name']}")
            if st.button("🚪 Logout"):
                st.logout()

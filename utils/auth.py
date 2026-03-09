"""
KeyCloak OIDC authentication for enterprise/university deployment.

Uses Streamlit's native st.login() (≥1.42.0) with KeyCloak as the OIDC provider.
Requires authlib>=1.3.2 and a .streamlit/secrets.toml with [auth.keycloak] config.

KeyCloak federates with the institution's identity provider (e.g. Microsoft Entra ID
via SAML2) and exposes an OIDC endpoint that Streamlit talks to directly — no reverse
proxy auth layer needed.

To disable authentication for local development, set AUTH_ENABLED=false.
"""
import os
import streamlit as st
from typing import Optional


def is_auth_enabled() -> bool:
    """Check if authentication is enabled via environment variable."""
    return os.getenv("AUTH_ENABLED", "false").lower() in ("true", "1", "yes")


def get_authenticated_user() -> Optional[str]:
    """
    Get authenticated username from Streamlit OIDC session.

    Returns:
        Username string if authenticated, None if auth disabled or not logged in.
    """
    if not is_auth_enabled():
        return None

    if not st.experimental_user.is_logged_in:
        return None

    # Prefer 'name', fall back to 'email', then 'sub' (OIDC subject claim)
    name = st.experimental_user.get("name")
    if name:
        return name
    email = st.experimental_user.get("email")
    if email:
        return email
    sub = st.experimental_user.get("sub")
    if sub:
        return sub
    return "authenticated_user"


def require_authentication():
    """
    Authentication gate — blocks access if auth is enabled and user hasn't logged in.

    Call this at the top of app.py before rendering any content.
    Uses Streamlit's native OIDC flow via st.login('keycloak').
    """
    if not is_auth_enabled():
        return

    if st.experimental_user.is_logged_in:
        # User is authenticated — proceed
        return

    # Not logged in — show login page and stop
    st.set_page_config(
        page_title="Tabular ML Lab — Sign In",
        page_icon="🔒",
        layout="centered",
    )

    st.markdown("""
    <div style="text-align: center; padding: 3rem 0;">
        <h1>🔬 Tabular ML Lab</h1>
        <p style="font-size: 1.2rem; color: #64748b;">
            Sign in with your institutional credentials to continue.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔐 Sign in with KeyCloak", use_container_width=True, type="primary"):
            st.login("keycloak")

    st.markdown("""
    <div style="text-align: center; padding-top: 2rem; color: #94a3b8; font-size: 0.85rem;">
        Authentication is provided by your institution's identity provider via KeyCloak.<br/>
        If you have trouble signing in, contact your IT department.
    </div>
    """, unsafe_allow_html=True)

    st.stop()


def get_current_user() -> str:
    """
    Get the current authenticated user's display name.

    Returns:
        Username string, or "anonymous" if auth is disabled.
    """
    if not is_auth_enabled():
        return "anonymous"

    user = get_authenticated_user()
    return user if user else "anonymous"


def show_user_info():
    """Display authenticated user info and logout button in sidebar."""
    if not is_auth_enabled():
        return

    if not st.experimental_user.is_logged_in:
        return

    name = st.experimental_user.get("name", "")
    email = st.experimental_user.get("email", "")

    display = name or email or "User"
    st.sidebar.markdown(f"👤 **{display}**")
    if email and name:
        st.sidebar.caption(email)

    if st.sidebar.button("Sign out", key="auth_logout"):
        st.logout()

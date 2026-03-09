"""
Flexible authentication for university deployment.

Supports multiple authentication methods via Streamlit's native auth (≥1.42.0):
- OIDC providers (KeyCloak, Azure AD, Google Workspace, Okta, etc.)
- SAML providers
- Reverse proxy authentication (fallback for older deployments)

Configure via .streamlit/secrets.toml or disable with AUTH_ENABLED=false.
"""
import os
import streamlit as st
from typing import Optional, Dict


def is_auth_enabled() -> bool:
    """Check if authentication is enabled via environment variable."""
    return os.getenv("AUTH_ENABLED", "false").lower() in ("true", "1", "yes")


def get_auth_provider() -> str:
    """
    Get configured authentication provider from environment or secrets.
    
    Returns:
        Provider name (e.g., 'keycloak', 'saml', 'reverseproxy', 'azure', 'google')
    """
    # Check environment variable first
    provider = os.getenv("AUTH_PROVIDER", "").lower()
    if provider:
        return provider
    
    # Check secrets.toml for configured providers
    try:
        if hasattr(st, "secrets") and "connections" in st.secrets:
            connections = st.secrets["connections"]
            if "keycloak" in connections:
                return "keycloak"
            elif "azure" in connections:
                return "azure"
            elif "google" in connections:
                return "google"
            elif "saml" in connections:
                return "saml"
    except Exception:
        pass
    
    # Default to reverse proxy (legacy)
    return "reverseproxy"


def get_authenticated_user() -> Optional[Dict[str, str]]:
    """
    Get authenticated user info from current auth provider.
    
    Returns:
        Dict with user info (name, email, username) if authenticated, None otherwise
    """
    if not is_auth_enabled():
        return None
    
    provider = get_auth_provider()
    
    # OIDC/SAML providers via Streamlit's native auth
    if provider in ("keycloak", "azure", "google", "saml", "okta", "oidc"):
        if st.experimental_user.is_logged_in:
            return {
                "username": st.experimental_user.email.split("@")[0] if st.experimental_user.email else "unknown",
                "email": st.experimental_user.email or "",
                "name": st.experimental_user.name or st.experimental_user.email or "Unknown User"
            }
        return None
    
    # Reverse proxy authentication (legacy)
    elif provider == "reverseproxy":
        return _get_user_from_reverse_proxy()
    
    return None


def _get_user_from_reverse_proxy() -> Optional[Dict[str, str]]:
    """
    Extract user info from reverse proxy headers.
    
    This is a legacy method for deployments where upstream nginx/Apache handles
    authentication and passes user info via HTTP header (typically X-Remote-User).
    """
    import streamlit as st
    
    # Streamlit doesn't expose request headers directly in community edition
    # Check session state (set by auth gate)
    if "authenticated_user" in st.session_state:
        username = st.session_state["authenticated_user"]
        return {
            "username": username,
            "email": f"{username}@university.edu",  # Placeholder
            "name": username
        }
    
    # Check query params (fallback for reverse proxy that redirects with ?user=<username>)
    try:
        query_params = st.query_params
        if "user" in query_params:
            username = query_params["user"]
            st.session_state["authenticated_user"] = username
            return {
                "username": username,
                "email": f"{username}@university.edu",
                "name": username
            }
    except Exception:
        pass
    
    return None


def require_authentication():
    """
    Authentication gate — blocks access if auth is enabled and user hasn't logged in.
    
    Call this at the top of app.py before rendering any content.
    Adapts to configured auth provider automatically.
    """
    if not is_auth_enabled():
        return  # Auth disabled
    
    provider = get_auth_provider()
    
    # OIDC/SAML providers via Streamlit's native auth
    if provider in ("keycloak", "azure", "google", "saml", "okta", "oidc"):
        if not st.experimental_user.is_logged_in:
            st.title("🔐 Tabular ML Lab")
            st.markdown(f"""
            ### Authentication Required
            
            This application requires institutional credentials.
            Click the button below to authenticate via {provider.title()}.
            """)
            
            # Streamlit's native OIDC/SAML login button
            if st.button(f"🔑 Login with {provider.title()}", type="primary"):
                st.login(provider)
            
            st.stop()  # Block access until authenticated
    
    # Reverse proxy authentication (legacy)
    elif provider == "reverseproxy":
        user = _get_user_from_reverse_proxy()
        if not user:
            st.error("""
            ⚠️ **Authentication Error**
            
            Expected authenticated user from reverse proxy, but no user info found.
            
            **For administrators:** Ensure your reverse proxy (nginx/Apache) is configured to:
            1. Handle authentication (LDAP/AD/SAML)
            2. Pass authenticated username via `X-Remote-User` header
            3. Redirect to this app with `?user=<username>` query parameter
            
            Or configure native OIDC/SAML in `.streamlit/secrets.toml` and set `AUTH_PROVIDER=keycloak|azure|google|saml`.
            """)
            st.stop()


def show_user_info():
    """Display authenticated user info in sidebar (optional)."""
    user = get_authenticated_user()
    if user:
        with st.sidebar:
            st.markdown("---")
            st.markdown(f"**Logged in as:** {user['name']}")
            
            provider = get_auth_provider()
            if provider in ("keycloak", "azure", "google", "saml", "okta", "oidc"):
                if st.button("🚪 Logout"):
                    st.logout()

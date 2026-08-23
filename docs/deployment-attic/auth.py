"""
User identity from reverse proxy headers.

Authentication must be enforced at the infrastructure layer — a reverse proxy
(Shibboleth, CAS, KeyCloak, Azure AD App Proxy, OAuth2 Proxy, nginx auth_request)
that gates ALL traffic before it reaches Streamlit.

This module only reads the authenticated user's identity from forwarded headers
for display purposes (sidebar badge). It never handles credentials or login flows.

Configuration via environment variables:
  AUTH_MODE          = disabled | proxy     (default: disabled)
  AUTH_HEADER        = X-Remote-User        (username header from proxy)
  AUTH_EMAIL_HEADER  = X-Remote-Email       (email header, optional)
  AUTH_NAME_HEADER   = X-Remote-Name        (display name header, optional)
  AUTH_LOGOUT_URL    = https://...          (redirect URL for logout, optional)
"""
import os
import logging
from dataclasses import dataclass
from typing import Optional

import streamlit as st

logger = logging.getLogger(__name__)


@dataclass
class UserInfo:
    """Authenticated user identity."""
    username: str
    email: Optional[str] = None
    display_name: Optional[str] = None

    @property
    def label(self) -> str:
        """Best available display string."""
        return self.display_name or self.email or self.username


def get_auth_mode() -> str:
    """Return configured auth mode from environment."""
    return os.environ.get("AUTH_MODE", "disabled").lower().strip()


def get_current_user() -> Optional[UserInfo]:
    """Return the current authenticated user, or None if auth is disabled.

    Reads identity from HTTP headers set by the reverse proxy.
    Uses st.context.headers (Streamlit 1.44+) for header access.
    """
    mode = get_auth_mode()

    if mode == "disabled":
        return None

    if mode == "proxy":
        return _get_user_from_proxy_headers()

    logger.warning(f"Unknown AUTH_MODE: {mode!r} — treating as disabled")
    return None


def _get_user_from_proxy_headers() -> Optional[UserInfo]:
    """Read user identity from reverse proxy headers.

    Works with any proxy that sets headers after authentication:
    Shibboleth, CAS, KeyCloak (as proxy), Azure AD App Proxy, nginx auth_request.
    """
    header_key = os.environ.get("AUTH_HEADER", "X-Remote-User")
    email_key = os.environ.get("AUTH_EMAIL_HEADER", "X-Remote-Email")
    name_key = os.environ.get("AUTH_NAME_HEADER", "X-Remote-Name")

    try:
        headers = st.context.headers
    except Exception:
        logger.debug("st.context.headers unavailable — auth check skipped")
        return None

    username = headers.get(header_key)
    if not username:
        return None

    return UserInfo(
        username=username.strip(),
        email=(headers.get(email_key) or "").strip() or None,
        display_name=(headers.get(name_key) or "").strip() or None,
    )


def render_user_badge():
    """Show the authenticated user's identity in the sidebar.

    Reads from reverse proxy headers. No-op when AUTH_MODE=disabled
    or when running without a proxy.
    """
    mode = get_auth_mode()
    if mode == "disabled":
        return

    user = get_current_user()
    if user is None:
        return

    logout_url = os.environ.get("AUTH_LOGOUT_URL", "").strip()

    with st.sidebar:
        st.markdown(f"**{user.label}**")
        if user.email:
            st.caption(user.email)
        if logout_url:
            st.markdown(f"[Logout]({logout_url})")

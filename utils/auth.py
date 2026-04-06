"""
Authentication stub for the main branch.

On main, auth is always disabled. The university-docker branch has the full
implementation with reverse proxy, OIDC, and compute profile support.

This stub exists so that any code that imports from utils.auth won't break.
"""
from typing import Optional
from dataclasses import dataclass


@dataclass
class UserInfo:
    username: str
    email: Optional[str] = None
    display_name: Optional[str] = None

    @property
    def label(self) -> str:
        return self.display_name or self.email or self.username


def get_auth_mode() -> str:
    return "disabled"


def get_current_user() -> Optional[UserInfo]:
    return None


def require_auth() -> Optional[UserInfo]:
    return None


def render_user_badge():
    pass

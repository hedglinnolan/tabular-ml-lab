"""Server mode's per-user workspaces (V2 definition of done §4).

Each account works in its own workspace, ``$TURBOTAB_HOME/users/<name>/`` (its own ``projects/``
and ``uploads/``), served by its own :class:`ProjectService`, so every route that names a project
looks it up in the signed-in user's workspace and nowhere else: another user's project id is
simply not there. The services share one pool of job workers, so a server's memory does not grow
with its number of accounts. The gate (``turbotab.server.auth``) also checks ownership before any
route that takes a project id runs (:func:`owns`), so a route added later cannot forget it.
"""
from __future__ import annotations

import dataclasses
import threading
from pathlib import Path

from turbotab.core.config import Settings
from turbotab.core.jobs import JobRunner
from turbotab.core.workspace import PROJECT_ID_RE
from turbotab.server.service import WORKER_PRELOAD, ProjectService
from turbotab.server.users import check_username


def user_home(home: Path, user: str) -> Path:
    """``<home>/users/<user>``; ``user`` is checked first, so it can never name another folder."""
    return Path(home) / "users" / check_username(user)


def owns(home: Path, user: str, pid: str) -> bool:
    """Whether ``pid`` is a project in ``user``'s workspace."""
    if not PROJECT_ID_RE.fullmatch(pid or ""):
        return False
    return (user_home(home, user) / "projects" / pid).is_dir()


class Tenants:
    """One :class:`ProjectService` per user, made on the user's first request."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.runner = JobRunner(settings.workers, preload=WORKER_PRELOAD)
        self._lock = threading.Lock()
        self._services: dict[str, ProjectService] = {}
        self._closed = False

    @property
    def workers(self) -> int:
        return self.runner.workers

    def for_user(self, user: str) -> ProjectService:
        home = user_home(self.settings.home, user)
        with self._lock:
            if self._closed:
                raise RuntimeError("the server is shutting down")
            service = self._services.get(user)
            if service is None:
                settings = dataclasses.replace(self.settings, home=home)
                service = self._services[user] = ProjectService(settings, runner=self.runner)
            return service

    def close(self) -> None:
        with self._lock:
            self._closed = True
            services = list(self._services.values())
            self._services.clear()
        try:
            for service in services:
                service.close()
        finally:
            self.runner.shutdown()

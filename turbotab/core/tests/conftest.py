from __future__ import annotations

import pytest

from turbotab.core.jobs import JobRunner


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: over 35 s on a developer machine; the nightly tier of CI (.github/workflows/v2.yml) "
        "runs it and the fast tier does not (-m 'not slow'). The server tests register it too.")


@pytest.fixture(scope="session")
def runner():
    """One small pool for the whole run: spawning workers costs about a second."""
    with JobRunner(workers=2) as pool:
        yield pool

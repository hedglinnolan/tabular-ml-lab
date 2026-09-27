from __future__ import annotations

import pytest

from turbotab.core.jobs import JobRunner


@pytest.fixture(scope="session")
def runner():
    """One small pool for the whole run: spawning workers costs about a second."""
    with JobRunner(workers=2) as pool:
        yield pool

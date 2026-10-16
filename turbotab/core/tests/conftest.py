from __future__ import annotations

import os

import pytest

from turbotab.core.jobs import JobRunner

# The claims registry's strict mode (CLAIMS_REGISTRY.md §2.2), on for every test: a claim rendered
# outside its regime raises instead of falling back. Set at import, so each xdist worker has it.
os.environ["TURBOTAB_CLAIMS_STRICT"] = "1"


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: over 35 s on a developer machine; the full tier of CI "
        "(.github/workflows/v2-full.yml) runs it and the fast tier does not (-m 'not slow'). "
        "The server tests register it too.")


@pytest.fixture(scope="session")
def runner():
    """One small pool for the whole run: spawning workers costs about a second."""
    with JobRunner(workers=2) as pool:
        yield pool

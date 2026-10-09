"""Markers for the tests under turbotab/ (the fast tier of CI runs ``-m "not slow"``)."""
from __future__ import annotations


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: takes over 35 s on a developer machine; the nightly tier of CI "
        "(.github/workflows/v2.yml) runs it, the fast tier does not (-m 'not slow')")

"""The swarm's tests: the driver, its options and the analyst's hook (DRIVER_PLAN D1, D2)."""
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: runs a real server through a whole run (over 35 s)")


@pytest.fixture(scope="session", autouse=True)
def no_pilot_launches(tmp_path_factory):
    """The pre-registration's fills read the graded runs from an empty folder of the tests' own,
    never the pilot's launches (``prereg_fill.graded_runs``); a test of that guard names its
    own folder."""
    from research.swarm import prereg_fill

    before = getattr(prereg_fill, "GRADED", None)
    prereg_fill.GRADED = tmp_path_factory.mktemp("launches")
    yield
    prereg_fill.GRADED = before

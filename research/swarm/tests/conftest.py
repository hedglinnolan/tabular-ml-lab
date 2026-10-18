"""The swarm's tests: the driver, its options and the analyst's hook (DRIVER_PLAN D1, D2)."""


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: runs a real server through a whole run (over 35 s)")

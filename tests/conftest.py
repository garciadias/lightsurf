import pytest

from tests.fixtures.data_fixtures import *  # noqa
from tests.fixtures.domain_fixtures import *  # noqa
from tests.fixtures.infrastructure_fixtures import *  # noqa


def pytest_addoption(parser):
    # Configuring pytest to run slow tests
    parser.addoption(
        "--slow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    run_slow = config.getoption("--slow")
    if not run_slow:
        for item in items:
            if "slow" in item.keywords:
                print(f"Skipping slow test: {item.nodeid}")
                item.add_marker(pytest.mark.skip(reason="need --slow option to run"))

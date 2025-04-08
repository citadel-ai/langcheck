import os

import pytest


@pytest.fixture(autouse=True)
def clean_environment():
    original_env = dict(os.environ)

    yield

    os.environ.clear()
    os.environ.update(original_env)

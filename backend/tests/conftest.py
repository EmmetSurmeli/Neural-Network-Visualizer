import pytest
from backend.runtime import limiter


@pytest.fixture(autouse=True)
def isolated_rate_windows():
    # Each test is a distinct usage scenario. Rate-limit behavior is tested explicitly.
    limiter.buckets.clear()
    yield
    limiter.buckets.clear()

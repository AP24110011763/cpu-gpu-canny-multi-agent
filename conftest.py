"""Makes the project root importable for pytest, and keeps the test suite fast."""
import pytest

import tools.canny_common as canny_common


@pytest.fixture(autouse=True)
def no_timed_warmup(monkeypatch):
    """Real runs warm up for at least 1 s (tools/canny_common.warm_up). Tests do not wait for that."""
    monkeypatch.setattr(canny_common, "MIN_WARMUP_SECONDS", 0.0)

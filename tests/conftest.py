"""Session-wide pytest fixtures."""

import pytest

from extract_fixtures import CaptureFixture
from fixture_utils import CAPTURE_NAMES, parse_capture


@pytest.fixture(scope="session")
def all_captures() -> dict[str, CaptureFixture]:
    """The parsed framing captures, keyed by name."""
    return {name: parse_capture(name) for name in CAPTURE_NAMES}

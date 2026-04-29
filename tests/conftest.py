"""
Pytest fixtures and parametrization for XL2HB framing tests.

Parses captures once per session; parametrizes block-level tests
with descriptive IDs like ``test_fullwidth_k_block0_K_L3495_x84``.
"""

import pytest

from extract_fixtures import PLANE_NAMES, CaptureFixture, parse_xl2hb_capture
from fixture_utils import read_fixture

# Captures to test (name -> .xl2hb filename stem)
CAPTURE_NAMES = [
    "a4_white",
    "test_fullwidth_k",
    "allblack_1000",
    "halfblack_1000",
    "a4_black",
    "test_fullwidth_y",
    "red_100",
    "test_fullwidth_c",
    "gray75_1000",
]


@pytest.fixture(scope="session")
def all_captures() -> dict[str, CaptureFixture]:
    """Parse all test captures once per session."""
    result = {}
    for name in CAPTURE_NAMES:
        data = read_fixture(f"{name}.xl2hb")
        if data is None:
            pytest.skip(f"Fixture {name}.xl2hb not found")
        result[name] = parse_xl2hb_capture(data)
    return result


def _build_block_params() -> list:
    """Build (capture_name, block_idx, test_id) triples for parametrize.

    Parses captures eagerly so pytest can collect IDs at import time.
    """
    params = []
    for name in CAPTURE_NAMES:
        data = read_fixture(f"{name}.xl2hb")
        if data is None:
            continue
        cap = parse_xl2hb_capture(data)
        for i, block in enumerate(cap.blocks):
            plane = PLANE_NAMES.get(block.plane_id, f"P{block.plane_id}")
            tid = f"{name}_block{i}_{plane}_L{block.start_line}_x{len(block.entries)}"
            params.append(pytest.param(name, i, id=tid))
    return params


BLOCK_PARAMS = _build_block_params()

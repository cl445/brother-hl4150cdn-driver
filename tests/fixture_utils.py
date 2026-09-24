"""Reading zstd-compressed test fixtures and comparing output against them.

Fixtures live in ``tests/fixtures/`` as ``.zst`` files and are committed
to the repository, so a missing fixture is an error, not a skip.
"""

import functools
from pathlib import Path

import pytest
import zstandard

from extract_fixtures import PLANE_NAMES, CaptureFixture, parse_xl2hb_capture

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

# Captures whose framing (PJL, blocks, plane buffers) the framing tests check.
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


def read_fixture(name: str) -> bytes:
    """Read a zstd-compressed fixture by its original filename.

    Args:
        name: Original filename (e.g. ``"a4_white.xl2hb"``).

    Returns:
        Decompressed bytes.
    """
    return zstandard.decompress((FIXTURES_DIR / f"{name}.zst").read_bytes())


def assert_bytes_equal(actual: bytes, expected: bytes, label: str) -> None:
    """Assert byte-for-byte equality, reporting the first difference on failure."""
    if actual == expected:
        return
    msg_parts = [f"{label}: length {len(actual)} vs {len(expected)}"]
    for i in range(min(len(actual), len(expected))):
        if actual[i] != expected[i]:
            start = max(0, i - 8)
            end_a = min(len(actual), i + 8)
            end_e = min(len(expected), i + 8)
            msg_parts.append(f"  First diff at byte {i}: got 0x{actual[i]:02x}, expected 0x{expected[i]:02x}")
            msg_parts.append(f"  Expected [{start}:{end_e}]: {expected[start:end_e].hex()}")
            msg_parts.append(f"  Actual   [{start}:{end_a}]: {actual[start:end_a].hex()}")
            break
    pytest.fail("\n".join(msg_parts))


def assert_matches_fixture(name: str, out: bytes) -> None:
    """Assert that `out` equals the capture ``<name>.xl2hb``."""
    assert_bytes_equal(out, read_fixture(f"{name}.xl2hb"), name)


@functools.cache
def parse_capture(name: str) -> CaptureFixture:
    """Parse the capture ``<name>.xl2hb`` once per session."""
    return parse_xl2hb_capture(read_fixture(f"{name}.xl2hb"), name)


def capture_block_params() -> list:
    """One pytest.param (capture_name, block_idx) per ReadImage block of every capture.

    IDs look like ``test_fullwidth_k_block0_K_L3495_x84``.
    """
    params = []
    for name in CAPTURE_NAMES:
        for i, block in enumerate(parse_capture(name).blocks):
            plane = PLANE_NAMES.get(block.plane_id, f"P{block.plane_id}")
            tid = f"{name}_block{i}_{plane}_L{block.start_line}_x{len(block.entries)}"
            params.append(pytest.param(name, i, id=tid))
    return params

"""The Rust kernels: byte-identity vs. the Cython kernels and the pure-Python reference."""

import numpy as np
import pytest

from brother_xl2hb import _native
from brother_xl2hb.color import dither
from brother_xl2hb.models import HL4150CDN

_build = _native.rust_build()
if isinstance(_build, str):
    pytest.skip(f"Rust build not in use ({_build}); see scripts/build_rust.sh", allow_module_level=True)
_rust_dither = _build._dither_fast

try:
    from brother_xl2hb._native import _dither_fast as _cython_dither
except ImportError:
    _cython_dither = None


@pytest.mark.parametrize("width", [0, 1, 7, 8, 9, 4761, 4768])
def test_dither_row_1bpp_matches_numpy_and_cython(width):
    rng = np.random.default_rng(width)
    thresholds = dither.load_dither_tables(cache=HL4150CDN.dither_cache)["C"].tiled_thresholds(width + 3)
    for y in (0, 17, 31):
        row = rng.integers(0, 256, 2 * width + 6, dtype=np.uint8)[::2]  # strided, as a KCMY column
        expected = bytes(np.packbits((255 - row[:width]) > thresholds[y % len(thresholds)][:width]))
        got = _rust_dither.dither_row_1bpp(row, thresholds[y % len(thresholds)], width)
        assert got == expected
        if _cython_dither is not None:
            assert got == _cython_dither.dither_row_1bpp(row, thresholds[y % len(thresholds)], width)


def test_dither_row_1bpp_reversed_row():
    row = np.arange(64, dtype=np.uint8) * 4
    thresholds = np.full(64, 100, np.uint8)
    expected = bytes(np.packbits((255 - row[::-1]) > thresholds))
    assert _rust_dither.dither_row_1bpp(row[::-1], thresholds, 64) == expected


def test_dither_row_1bpp_accepts_bytes():
    row, thresholds = bytes(range(15, 256, 16)), bytes(16)  # only the last pixel is white
    assert _rust_dither.dither_row_1bpp(row, thresholds, 16) == bytes([0xFF, 0xFE])


@pytest.mark.parametrize(
    ("row", "thresholds", "width"),
    [
        (np.zeros(4, np.uint8), np.zeros(8, np.uint8), 5),  # row too short
        (np.zeros(8, np.uint8), np.zeros(4, np.uint8), 5),  # thresholds too short
        (np.zeros(8, np.uint8), np.zeros(16, np.uint8)[::2], 5),  # thresholds not contiguous
        (np.zeros((2, 4), np.uint8), np.zeros(8, np.uint8), 5),  # row not one-dimensional
    ],
)
def test_dither_row_1bpp_rejects_bad_buffers(row, thresholds, width):
    with pytest.raises(ValueError):  # noqa: PT011 -- the messages differ from Cython's
        _rust_dither.dither_row_1bpp(row, thresholds, width)
    if _cython_dither is not None:
        with pytest.raises(ValueError):  # noqa: PT011
            _cython_dither.dither_row_1bpp(row, thresholds, width)

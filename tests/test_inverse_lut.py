"""Inverse-LUT cache: byte-identity vs. the per-pixel interpolation path."""

import numpy as np
import pytest

import color_lut
from color_lut import (
    _rgb_to_cmyk_interp_arr,
    inverse_lut,
    rgb_to_cmyk_lut,
    write_inverse_lut,
)


@pytest.fixture(autouse=True)
def _fresh_inverse_lut():
    """Tests point INVERSE_LUT_PATH elsewhere; never leak a cached LUT between tests."""
    inverse_lut.cache_clear()
    yield
    inverse_lut.cache_clear()


def _sample_rgb(seed: int, width: int) -> bytes:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (width, 3), dtype=np.uint8).tobytes()


def _interp(rgb: bytes, width: int) -> tuple[bytes, bytes, bytes, bytes]:
    """The per-pixel interpolation path, as bytes like rgb_to_cmyk_lut returns."""
    k, c, m, y = _rgb_to_cmyk_interp_arr(rgb, width)
    return k.tobytes(), c.tobytes(), m.tobytes(), y.tobytes()


def test_interp_returns_one_byte_per_pixel_per_channel() -> None:
    sample = _sample_rgb(seed=1, width=128)
    k, c, m, y = _interp(sample, 128)
    assert len(k) == len(c) == len(m) == len(y) == 128


def test_cached_lookup_matches_direct_interp(tmp_path, monkeypatch) -> None:
    target = tmp_path / "inverse_lut.npy"
    monkeypatch.setattr(color_lut, "INVERSE_LUT_PATH", target)

    written = write_inverse_lut(target)
    assert written.exists()
    assert written.stat().st_size > 60 * 1024 * 1024

    sample = _sample_rgb(seed=42, width=4096)
    k_fast, c_fast, m_fast, y_fast = rgb_to_cmyk_lut(sample, 4096)
    k_slow, c_slow, m_slow, y_slow = _interp(sample, 4096)
    assert k_fast == k_slow
    assert c_fast == c_slow
    assert m_fast == m_slow
    assert y_fast == y_slow


def test_falls_back_when_cache_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(color_lut, "INVERSE_LUT_PATH", tmp_path / "absent.npy")

    sample = _sample_rgb(seed=7, width=512)
    assert rgb_to_cmyk_lut(sample, 512) == _interp(sample, 512)


@pytest.mark.parametrize(
    "rgb",
    [(0, 0, 0), (255, 255, 255), (128, 128, 128), (255, 0, 0), (0, 255, 0), (0, 0, 255)],
)
def test_special_values_match(tmp_path, monkeypatch, rgb) -> None:
    target = tmp_path / "inverse_lut.npy"
    monkeypatch.setattr(color_lut, "INVERSE_LUT_PATH", target)
    write_inverse_lut(target)

    pixel = bytes(rgb)
    assert rgb_to_cmyk_lut(pixel, 1) == _interp(pixel, 1)


def test_native_gather_matches_numpy_path(monkeypatch) -> None:
    """The Cython gather and the numpy fancy-index path agree, incl. over-long rows."""
    if not color_lut.HAS_CYTHON_COLOR:
        pytest.skip("_color_fast extension not built")
    rng = np.random.default_rng(3)
    lut = rng.integers(0, 256, color_lut._INVERSE_LUT_SHAPE, dtype=np.uint8)
    monkeypatch.setattr(color_lut, "inverse_lut", lambda table=None: lut)

    sample = _sample_rgb(seed=11, width=5000)
    native = color_lut.rgb_to_cmyk_lut_arr(sample, 4768)
    monkeypatch.setattr(color_lut, "HAS_CYTHON_COLOR", False)
    reference = color_lut.rgb_to_cmyk_lut_arr(sample, 4768)
    for got, want in zip(native, reference, strict=True):
        np.testing.assert_array_equal(got, want)

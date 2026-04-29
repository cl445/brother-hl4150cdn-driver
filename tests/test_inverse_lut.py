"""Inverse-LUT cache: byte-identity vs. the per-pixel interpolation path."""

import numpy as np
import pytest

import color_lut
from color_lut import (
    _load_inverse_lut,
    _rgb_to_cmyk_interp,
    rgb_to_cmyk_lut,
    write_inverse_lut,
)


def _sample_rgb(seed: int, width: int) -> bytes:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (width, 3), dtype=np.uint8).tobytes()


def test_interp_returns_one_byte_per_pixel_per_channel() -> None:
    sample = _sample_rgb(seed=1, width=128)
    k, c, m, y = _rgb_to_cmyk_interp(sample, 128)
    assert len(k) == len(c) == len(m) == len(y) == 128


def test_cached_lookup_matches_direct_interp(tmp_path, monkeypatch) -> None:
    target = tmp_path / "inverse_lut.npy"
    monkeypatch.setattr(color_lut, "INVERSE_LUT_PATH", target)
    _load_inverse_lut.cache_clear()

    written = write_inverse_lut(target)
    assert written.exists()
    assert written.stat().st_size > 60 * 1024 * 1024

    sample = _sample_rgb(seed=42, width=4096)
    k_fast, c_fast, m_fast, y_fast = rgb_to_cmyk_lut(sample, 4096)
    k_slow, c_slow, m_slow, y_slow = _rgb_to_cmyk_interp(sample, 4096)
    assert k_fast == k_slow
    assert c_fast == c_slow
    assert m_fast == m_slow
    assert y_fast == y_slow

    _load_inverse_lut.cache_clear()


def test_falls_back_when_cache_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(color_lut, "INVERSE_LUT_PATH", tmp_path / "absent.npy")
    _load_inverse_lut.cache_clear()

    sample = _sample_rgb(seed=7, width=512)
    assert rgb_to_cmyk_lut(sample, 512) == _rgb_to_cmyk_interp(sample, 512)

    _load_inverse_lut.cache_clear()


@pytest.mark.parametrize(
    "rgb",
    [(0, 0, 0), (255, 255, 255), (128, 128, 128), (255, 0, 0), (0, 255, 0), (0, 0, 255)],
)
def test_special_values_match(tmp_path, monkeypatch, rgb) -> None:
    target = tmp_path / "inverse_lut.npy"
    monkeypatch.setattr(color_lut, "INVERSE_LUT_PATH", target)
    _load_inverse_lut.cache_clear()
    write_inverse_lut(target)

    pixel = bytes(rgb)
    assert rgb_to_cmyk_lut(pixel, 1) == _rgb_to_cmyk_interp(pixel, 1)

    _load_inverse_lut.cache_clear()

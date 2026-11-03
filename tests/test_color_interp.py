"""Colour interpolation: the native kernel against the numpy reference, and the compact tables it reads."""

import dataclasses

import numpy as np
import pytest

from brother_xl2hb.color import color_lut
from brother_xl2hb.color.color_lut import ColorTable, _rgb_to_cmyk_interp_arr
from brother_xl2hb.models import HL4150CDN


def _sample_rgb(seed: int, width: int) -> bytes:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (width, 3), dtype=np.uint8).tobytes()


def _interp(rgb: bytes, width: int, table: ColorTable | None = None) -> tuple[bytes, bytes, bytes, bytes]:
    """The per-pixel interpolation path, as bytes like rgb_to_cmyk_lut returns."""
    k, c, m, y = _rgb_to_cmyk_interp_arr(rgb, width, table or HL4150CDN.grid())
    return k.tobytes(), c.tobytes(), m.tobytes(), y.tobytes()


def test_interp_returns_one_byte_per_pixel_per_channel() -> None:
    sample = _sample_rgb(seed=1, width=128)
    k, c, m, y = _interp(sample, 128)
    assert len(k) == len(c) == len(m) == len(y) == 128


def test_full_cube_matches_numpy_interp() -> None:
    """All 16.7M colours of the PPD default grid (improve gray), natively and in numpy."""
    native = color_lut._color_fast
    if native is None:
        pytest.skip("_color_fast extension not built")
    table = HL4150CDN.grid(improve_gray=True)
    g, b = np.meshgrid(np.arange(256, dtype=np.uint8), np.arange(256, dtype=np.uint8), indexing="ij")
    rgb = np.empty((65536, 3), dtype=np.uint8)
    rgb[:, 1] = g.ravel()
    rgb[:, 2] = b.ravel()
    planes = np.empty((4, 65536), dtype=np.uint8)
    for r in range(256):
        rgb[:, 0] = r
        native.interp_kcmy(rgb.tobytes(), 65536, *color_lut.interp_arrays(table), *planes)
        np.testing.assert_array_equal(planes, np.array(_rgb_to_cmyk_interp_arr(rgb.tobytes(), 65536, table)))


def test_native_interp_matches_numpy_path(monkeypatch) -> None:
    """rgb_to_cmyk_lut_arr gives the same planes with and without the native module, incl. over-long rows."""
    if color_lut._color_fast is None:
        pytest.skip("_color_fast extension not built")
    sample = _sample_rgb(seed=11, width=5000)
    native = color_lut.rgb_to_cmyk_lut_arr(sample, 4768, HL4150CDN.grid())
    monkeypatch.setattr(color_lut, "_color_fast", None)
    reference = color_lut.rgb_to_cmyk_lut_arr(sample, 4768, HL4150CDN.grid())
    for got, want in zip(native, reference, strict=True):
        np.testing.assert_array_equal(got, want)


_ALL_GRIDS = [
    HL4150CDN.grid(profile, improve_gray=gray, variant=variant, rich_black=rich)
    for profile in ("rgb", "srgb", "cmyk")
    for gray in (False, True)
    for variant in ("default", "density2", "glossy")
    for rich in (False, True)
]


def _interp_sample() -> bytes:
    """Every grid point and its neighbours, every axis end, and random colours."""
    axis = np.unique(np.clip(np.arange(0, 257, 16)[:, None] + np.array([-1, 0, 1]), 0, 255))
    r, g, b = np.meshgrid(axis, axis, axis, indexing="ij")
    grid_points = np.stack([r.ravel(), g.ravel(), b.ravel()], axis=1).astype(np.uint8)
    rng = np.random.default_rng(4913)
    return np.concatenate([grid_points, rng.integers(0, 256, (200_000, 3), dtype=np.uint8)]).tobytes()


@pytest.mark.parametrize("table", _ALL_GRIDS, ids=lambda t: f"{t.name}{'_rich' if t.rich_black else ''}")
def test_native_interp_matches_numpy_for_every_grid(table) -> None:
    """The compact native interpolation equals the numpy reference on every colour grid."""
    native = color_lut._color_fast
    if native is None:
        pytest.skip("_color_fast extension not built")
    rgb = _interp_sample()
    n = len(rgb) // 3
    planes = np.empty((4, n), dtype=np.uint8)
    native.interp_kcmy(rgb, n, *color_lut.interp_arrays(table), planes[0], planes[1], planes[2], planes[3])
    np.testing.assert_array_equal(planes, np.array(color_lut._rgb_to_cmyk_interp_arr(rgb, n, table)))


def test_interp_arrays_compact_form() -> None:
    interp = color_lut.interp_arrays(HL4150CDN.grid())
    assert interp.grid.dtype == np.uint8
    assert interp.grid.size == 4913 * 4
    assert interp.weights.size == 4913 * 8
    for array in interp:
        assert not array.flags.writeable
    # Each entry keeps the non-zero weights of the 9-byte table, corners in weight order.
    _, raw = color_lut._load_data(HL4150CDN.grid())
    entries = raw.reshape(-1, 9)
    compact = interp.weights.reshape(-1, 8)
    for i in (0, 1, 1000, 4912):
        packed = int(compact[i, 5]) | int(compact[i, 6]) << 8
        corners = [(packed >> (3 * k)) & 7 for k in range(4)]
        kept = {c: int(w) for c, w in zip(corners, compact[i, :4], strict=True) if w}
        assert kept == {c: int(w) for c, w in enumerate(entries[i, 1:]) if w}
        assert compact[i, 4] == entries[i, 0]


def test_interp_arrays_rejects_grids_it_cannot_compact(monkeypatch) -> None:
    lut, interp = color_lut._load_data(HL4150CDN.grid())
    table = dataclasses.replace(HL4150CDN.grid(), variant="test")
    wide = lut.copy()
    wide[7, 1] = 256
    monkeypatch.setattr(color_lut, "_load_data", lambda t: (wide, interp))
    with pytest.raises(ValueError, match="outside 0-255"):
        color_lut.interp_arrays(table)
    negative = lut.copy()
    negative[7, 1] = -1
    monkeypatch.setattr(color_lut, "_load_data", lambda t: (negative, interp))
    with pytest.raises(ValueError, match="outside 0-255"):
        color_lut.interp_arrays(dataclasses.replace(table, variant="test3"))
    eight = interp.copy().reshape(-1, 9)
    eight[3, 1:] = 28
    monkeypatch.setattr(color_lut, "_load_data", lambda t: (lut, eight))
    with pytest.raises(ValueError, match="more than 4 corners"):
        color_lut.interp_arrays(dataclasses.replace(table, variant="test2"))

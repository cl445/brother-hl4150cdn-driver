"""Inverse-LUT cache: byte-identity vs. the per-pixel interpolation path, source-digest checks."""

import dataclasses
import logging
from pathlib import Path

import numpy as np
import pytest

from brother_xl2hb.color import color_lut
from brother_xl2hb.color.color_lut import (
    ColorTable,
    _rgb_to_cmyk_interp_arr,
    inverse_lut,
    inverse_lut_path,
    rgb_to_cmyk_lut,
    source_digest,
    write_inverse_lut,
    write_inverse_luts,
)
from brother_xl2hb.models import HL4150CDN

INVERSE_LUT_TABLES = HL4150CDN.inverse_lut_tables
_TABLE_IDS = [t.name for t in INVERSE_LUT_TABLES]


def _grid_dir(directory: Path) -> Path:
    """`directory` with symlinks to the real grids and interpolation tables, but no inverse LUTs."""
    directory.mkdir(parents=True, exist_ok=True)
    for blob in HL4150CDN.color_data_dir.glob("*.bin"):
        (directory / blob.name).symlink_to(blob)
    return directory


def _at(table: ColorTable, directory: Path) -> ColorTable:
    """`table` with its colour data (and so its inverse LUT) in `directory`."""
    return dataclasses.replace(table, data_dir=directory)


@pytest.fixture(autouse=True)
def _fresh_inverse_lut():
    """Tests write inverse LUTs to temporary directories; never leak a cached LUT or digest between tests."""
    inverse_lut.cache_clear()
    source_digest.cache_clear()
    yield
    inverse_lut.cache_clear()
    source_digest.cache_clear()


@pytest.fixture
def installed(installed_inverse_luts) -> Path:
    """A colour data directory with all inverse LUTs, as install.sh writes them."""
    return installed_inverse_luts / HL4150CDN.key


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


def test_improve_gray_tables_have_inverse_luts() -> None:
    """The PPD default BRGray=ON selects the _ig tables; they must not fall back to interpolation."""
    names = {t.name for t in INVERSE_LUT_TABLES}
    assert names == {"rgb_default", "rgb_ig_default", "srgb_default", "srgb_ig_default"}


def test_paths_are_distinct_and_carry_the_digest() -> None:
    paths = [inverse_lut_path(t) for t in INVERSE_LUT_TABLES]
    assert len(set(paths)) == len(paths)
    for table, path in zip(INVERSE_LUT_TABLES, paths, strict=True):
        assert path is not None
        assert path.name == f"inverse_lut_{table.name}-{source_digest(table)}.npy"
    assert inverse_lut_path(HL4150CDN.grid("cmyk")) is None
    assert inverse_lut_path(HL4150CDN.grid("rgb", rich_black=True)) is None


@pytest.mark.parametrize("table", INVERSE_LUT_TABLES, ids=_TABLE_IDS)
def test_cached_lookup_matches_direct_interp(installed, table) -> None:
    table = _at(table, installed)
    path = inverse_lut_path(table)
    assert path is not None
    assert path.parent == installed
    assert path.stat().st_size > 60 * 1024 * 1024
    assert inverse_lut(table) is not None

    sample = _sample_rgb(seed=42, width=4096)
    assert rgb_to_cmyk_lut(sample, 4096, table) == _interp(sample, 4096, table)


@pytest.mark.parametrize("table", INVERSE_LUT_TABLES, ids=_TABLE_IDS)
def test_full_cube_matches_native_interp(installed, table) -> None:
    """Every one of the 16.7M entries equals what the native kernel interpolates without the LUT."""
    if not color_lut.HAS_CYTHON_COLOR:
        pytest.skip("_color_fast extension not built")
    table = _at(table, installed)
    inv = inverse_lut(table)
    assert inv is not None
    r, g, b = np.meshgrid(*(np.arange(256, dtype=np.uint8),) * 3, indexing="ij")
    rgb = np.stack([r.ravel(), g.ravel(), b.ravel()], axis=1).tobytes()
    n = 256**3
    planes = np.empty((4, n), dtype=np.uint8)
    color_lut.interp_kcmy(rgb, n, *color_lut.interp_arrays(table), planes[0], planes[1], planes[2], planes[3])
    np.testing.assert_array_equal(inv.reshape(-1, 4), planes.T)


def test_falls_back_when_cache_missing(tmp_path) -> None:
    table = _at(HL4150CDN.grid(), _grid_dir(tmp_path))
    sample = _sample_rgb(seed=7, width=512)
    assert inverse_lut(table) is None
    assert rgb_to_cmyk_lut(sample, 512, table) == _interp(sample, 512, table)


@pytest.mark.parametrize(
    "rgb",
    [(0, 0, 0), (255, 255, 255), (128, 128, 128), (255, 0, 0), (0, 255, 0), (0, 0, 255)],
)
@pytest.mark.parametrize("table", INVERSE_LUT_TABLES, ids=_TABLE_IDS)
def test_special_values_match(installed, table, rgb) -> None:
    table = _at(table, installed)
    pixel = bytes(rgb)
    assert rgb_to_cmyk_lut(pixel, 1, table) == _interp(pixel, 1, table)


def test_lut_from_other_grid_is_ignored(tmp_path, monkeypatch, caplog) -> None:
    """A LUT computed from the generated fallback grid is not used once the real grid is present."""
    from brother_xl2hb.color.color_lut_gen import generate_rgb_default_lut

    table = _at(HL4150CDN.grid(), _grid_dir(tmp_path))
    real_digest = source_digest(table)
    real_load = color_lut._load_data
    generated = generate_rgb_default_lut()
    monkeypatch.setattr(color_lut, "_load_data", lambda t=table: (generated, real_load(t)[1]))
    source_digest.cache_clear()
    stale = write_inverse_lut(table)
    assert source_digest(table) != real_digest

    monkeypatch.setattr(color_lut, "_load_data", real_load)
    source_digest.cache_clear()
    with caplog.at_level(logging.WARNING, logger="brother_xl2hb.color.color_lut"):
        assert inverse_lut(table) is None
    assert stale.name in caplog.text
    sample = _sample_rgb(seed=5, width=256)
    assert rgb_to_cmyk_lut(sample, 256, table) == _interp(sample, 256, table)


def test_digest_less_legacy_lut_is_ignored(tmp_path, caplog) -> None:
    table = _at(HL4150CDN.grid(), _grid_dir(tmp_path))
    np.save(tmp_path / "inverse_lut.npy", np.zeros((1,), np.uint8))
    with caplog.at_level(logging.WARNING, logger="brother_xl2hb.color.color_lut"):
        assert inverse_lut(table) is None
    assert "inverse_lut.npy" in caplog.text


def test_write_inverse_luts_removes_stale_files(tmp_path) -> None:
    legacy = tmp_path / "inverse_lut.npy"
    old = tmp_path / "inverse_lut_rgb_default-0000000000000000.npy"
    other = tmp_path / "inverse_lut_srgb_default-0000000000000000.npy"  # not in the patched tables
    for p in (legacy, old, other):
        p.write_bytes(b"")

    (written,) = write_inverse_luts([HL4150CDN.grid()], tmp_path)

    assert written == inverse_lut_path(HL4150CDN.grid(), tmp_path)
    assert written.exists()
    assert not legacy.exists()
    assert not old.exists()
    assert other.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_native_gather_matches_numpy_path(monkeypatch) -> None:
    """The Cython gather and the numpy fancy-index path agree, incl. over-long rows."""
    if not color_lut.HAS_CYTHON_COLOR:
        pytest.skip("_color_fast extension not built")
    rng = np.random.default_rng(3)
    lut = rng.integers(0, 256, color_lut._INVERSE_LUT_SHAPE, dtype=np.uint8)
    monkeypatch.setattr(color_lut, "inverse_lut", lambda table=None: lut)

    sample = _sample_rgb(seed=11, width=5000)
    native = color_lut.rgb_to_cmyk_lut_arr(sample, 4768, HL4150CDN.grid())
    monkeypatch.setattr(color_lut, "HAS_CYTHON_COLOR", False)
    reference = color_lut.rgb_to_cmyk_lut_arr(sample, 4768, HL4150CDN.grid())
    for got, want in zip(native, reference, strict=True):
        np.testing.assert_array_equal(got, want)

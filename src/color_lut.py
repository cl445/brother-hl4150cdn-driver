"""3D colour LUT interpolation for the Brother HL-4150CDN.

Maps RGB input to CMYK ink values via a 17x17x17 grid plus tetrahedral
interpolation tables. Which of the 18 grids applies is decided like
`lookup_color_transform_table` in the original driver (see `ColorTable`).
"""

import functools
import logging
from collections.abc import Buffer
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

try:
    from _color_fast import gather_kcmy, interp_kcmy  # type: ignore[import-not-found]

    HAS_CYTHON_COLOR = True
except ImportError:
    HAS_CYTHON_COLOR = False

logger = logging.getLogger(__name__)

# Data directory containing extracted binary tables
_DATA_DIR = Path(__file__).resolve().parent / "color_data"

# LUT dimensions
_LUT_DIM = 17  # 17x17x17 grid
_LUT_ENTRIES = _LUT_DIM**3  # 4913
_LUT_BYTES = _LUT_ENTRIES * 8  # 39304 = 0x9988
_INTERP_TABLE_SIZE = _LUT_DIM * _LUT_DIM * 9  # 2601 bytes per table
_NUM_INTERP_TABLES = 17  # one per b_frac value (0..16)

# Pre-computed lookup tables for splitting 0-255 into grid index + fractional weight.
# Value 0-254: hi = v >> 4, frac = v & 0xF
# Value 255:   hi = 15,     frac = 16  (top of last grid cell)
_HI = np.array([v >> 4 for v in range(256)], dtype=np.int32)
_FRAC = np.array([16 if v == 255 else v & 0xF for v in range(256)], dtype=np.int32)

# 8 cube corner offsets in the flat LUT array.
# Maps to C int32 pointer offsets: 0, 0x22, 2, 0x24, 0x242, 0x264, 0x244, 0x266
# Converted to entry offsets: 0, 17, 1, 18, 289, 306, 290, 307
_CORNER_OFFSETS = np.array([0, 17, 1, 18, 289, 306, 290, 307], dtype=np.int32)


Profile = Literal["rgb", "srgb", "cmyk"]
Variant = Literal["default", "density2", "glossy"]


@dataclass(frozen=True, slots=True)
class ColorTable:
    """One of the colour grids `lookup_color_transform_table` chooses from.

    Attributes:
        profile: `rgb` for Normal colour matching, `srgb` for Vivid, `cmyk`
            for colour matching None.
        improve_gray: the ImpGray=ON grids (BRGray).
        variant: `density2` with toner save, `glossy` for glossy media.
        rich_black: pure black takes grid entry 0 (C, M, Y and K) instead of
            K only; the original does this with BREnhanceBlkPrt in the
            rgb/srgb profiles (`lut_selection` 0 in `load_color_profile`).
    """

    profile: Profile = "rgb"
    improve_gray: bool = False
    variant: Variant = "default"
    rich_black: bool = False

    @property
    def name(self) -> str:
        """Blob name stem, e.g. `rgb_ig_density2` for `rgb_ig_density2_lut.bin`."""
        return f"{self.profile}{'_ig' if self.improve_gray else ''}_{self.variant}"


DEFAULT_TABLE = ColorTable()

# Tables with a precomputed inverse LUT (install.sh writes both); every
# other table is interpolated per pixel.
INVERSE_LUT_TABLES: tuple[ColorTable, ...] = (ColorTable("rgb"), ColorTable("srgb"))


def _load_lut(table: ColorTable = DEFAULT_TABLE) -> npt.NDArray[np.int32]:
    """Load the 3D color grid of `table` as unpacked CMYK channels.

    The binary file contains packed int32 pairs (cm_packed, yk_packed).
    We unpack at load time into (4913, 4) int32 array [C, M, Y, K]
    to avoid packed arithmetic and int64 at runtime.

    Falls back to parametric generation if the binary file is missing: the
    generated rgb/srgb default grids stand in for every variant (with a
    warning, since the output then differs from the original's).

    Returns:
        Unpacked LUT of shape (_LUT_ENTRIES, 4), columns [C, M, Y, K].

    Raises:
        ValueError: If the binary LUT file has an unexpected size.
    """
    path = _DATA_DIR / f"{table.name}_lut.bin"
    if not path.exists():
        from color_lut_gen import generate_rgb_default_lut, generate_srgb_default_lut

        if table.name not in ("rgb_default", "srgb_default"):
            logger.warning("LUT binary %s missing (run scripts/extract_blobs.sh); using a generated grid", path.name)
        else:
            logger.info("LUT binary %s not found, generating from parametric model", path.name)
        return generate_srgb_default_lut() if table.profile == "srgb" else generate_rgb_default_lut()
    data = path.read_bytes()
    if len(data) != _LUT_BYTES:
        msg = f"LUT size {len(data)}, expected {_LUT_BYTES}"
        raise ValueError(msg)
    packed = np.frombuffer(data, dtype=np.int32).reshape(-1, 2)
    unpacked = np.empty((_LUT_ENTRIES, 4), dtype=np.int32)
    unpacked[:, 0] = packed[:, 0] & 0xFFFF  # cyan
    unpacked[:, 1] = packed[:, 0] >> 16  # magenta
    unpacked[:, 2] = packed[:, 1] & 0xFFFF  # yellow
    unpacked[:, 3] = packed[:, 1] >> 16  # black
    return unpacked


def _load_interp_tables() -> npt.NDArray[np.uint8]:
    """Load interpolation weight tables (17 tables, each 17*17*9 bytes).

    Falls back to analytical generation if the binary file is missing.

    Returns:
        Array of shape (_NUM_INTERP_TABLES, _LUT_DIM*_LUT_DIM, 9), uint8.

    Raises:
        ValueError: If the binary file has an unexpected size.
    """
    path = _DATA_DIR / "interp_tables.bin"
    if not path.exists():
        logger.info("Interpolation tables binary not found, generating analytically")
        from color_lut_gen import generate_interp_tables

        return generate_interp_tables()
    data = path.read_bytes()
    expected = _INTERP_TABLE_SIZE * _NUM_INTERP_TABLES
    if len(data) != expected:
        msg = f"Interp tables size {len(data)}, expected {expected}"
        raise ValueError(msg)
    return np.frombuffer(data, dtype=np.uint8).reshape(_NUM_INTERP_TABLES, _LUT_DIM * _LUT_DIM, 9)


# Ink for pure black (R=G=B=0) unless the table asks for rich black.
_K_PRESET = np.array([0, 0, 0, 255], dtype=np.int32)

# Precomputed full RGB→KCMY lookup table.
# Last axis order is K, C, M, Y in pixel-brightness convention
# (0 = full ink, 255 = no ink).
INVERSE_LUT_PATH = _DATA_DIR / "inverse_lut.npy"
_INVERSE_LUT_SHAPE = (256, 256, 256, 4)


@functools.lru_cache(maxsize=8)
def _load_data(table: ColorTable = DEFAULT_TABLE) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.uint8]]:
    """Load the grid of `table` and the interpolation tables (cached).

    Returns:
        Tuple of (lut, interp_tables); see `_load_lut` and `_load_interp_tables`.
    """
    return _load_lut(table), _load_interp_tables()


def black_ink(table: ColorTable, lut: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
    """C, M, Y, K ink for pure black: grid entry 0 for rich black, else K only.

    Returns:
        int32 array of 4 ink values.
    """
    return lut[0].copy() if table.rich_black else _K_PRESET.copy()


def interp_arrays(
    table: ColorTable = DEFAULT_TABLE,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.uint8], npt.NDArray[np.int32]]:
    """Flat, contiguous grid, interpolation weights and black ink for the native interpolation.

    Views on the cached `_load_data` arrays, so cheap to call per scanline.

    Returns:
        (grid of 4913*4 int32 as C, M, Y, K; 17*289*9 uint8 weights; 4 int32 black ink).
    """
    lut, interp = _load_data(table)
    return (
        np.ascontiguousarray(lut, dtype=np.int32).reshape(-1),
        np.ascontiguousarray(interp).reshape(-1),
        black_ink(table, lut),
    )


def inverse_lut_path(table: ColorTable = DEFAULT_TABLE) -> Path | None:
    """Where the inverse LUT of `table` is installed, next to `INVERSE_LUT_PATH`.

    Returns:
        `INVERSE_LUT_PATH` for the rgb default table, a sibling file for the
        srgb one, None for tables without a precomputed inverse LUT.
    """
    if table not in INVERSE_LUT_TABLES:
        return None
    if table.profile == "rgb":
        return INVERSE_LUT_PATH
    return INVERSE_LUT_PATH.with_name(f"inverse_lut_{table.profile}.npy")


@functools.lru_cache(maxsize=len(INVERSE_LUT_TABLES))
def inverse_lut(table: ColorTable = DEFAULT_TABLE) -> npt.NDArray[np.uint8] | None:
    """Memory-map the installed RGB→KCMY inverse LUT of `table`, or None if absent.

    Mapped read-only instead of loaded: each CUPS job is a fresh process,
    and reading all 64 MiB up front costs 1-3 s on a Pi. With the mapping
    only the pages for colours actually on the page are read, and the OS
    page cache shares them between jobs.

    Returns:
        Array of shape (256, 256, 256, 4) uint8, or None when the cache
        file is missing or has an unexpected shape.
    """
    path = inverse_lut_path(table)
    if path is None or not path.exists():
        return None
    try:
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
    except (ValueError, OSError) as exc:
        logger.warning("Failed to load inverse LUT %s: %s", path, exc)
        return None
    if arr.shape != _INVERSE_LUT_SHAPE or arr.dtype != np.uint8:
        logger.warning("Inverse LUT %s has unexpected shape %s/%s, ignoring", path, arr.shape, arr.dtype)
        return None
    # Plain ndarray view of the mapping: reshaping a np.memmap per scanline
    # runs memmap.__array_finalize__ each time (~50 ms per page on a Pi 3).
    return np.asarray(arr)


def precompute_inverse_lut(table: ColorTable = DEFAULT_TABLE) -> npt.NDArray[np.uint8]:
    """Evaluate the tetrahedral interpolation over all 16.7M RGB inputs.

    Iterates one R-slice at a time so the working set stays small enough
    for memory-constrained hosts.

    Returns:
        (256, 256, 256, 4) uint8 array; last axis is K, C, M, Y.
    """
    out = np.empty(_INVERSE_LUT_SHAPE, dtype=np.uint8)
    g_grid, b_grid = np.meshgrid(np.arange(256, dtype=np.uint8), np.arange(256, dtype=np.uint8), indexing="ij")
    gb_flat = np.empty((65536, 3), dtype=np.uint8)
    gb_flat[:, 1] = g_grid.ravel()
    gb_flat[:, 2] = b_grid.ravel()
    for r in range(256):
        gb_flat[:, 0] = r
        for i, plane in enumerate(_rgb_to_cmyk_interp_arr(gb_flat, 65536, table)):
            out[r, :, :, i] = plane.reshape(256, 256)
    return out


def write_inverse_lut(path: Path | None = None, table: ColorTable = DEFAULT_TABLE) -> Path:
    """Precompute the inverse LUT of `table` and save it to ``path`` (default `inverse_lut_path`).

    Returns:
        The path the array was written to.

    Raises:
        ValueError: If no path is given and `table` has no default location.
    """
    target = path or inverse_lut_path(table)
    if target is None:
        msg = f"no default inverse LUT path for {table.name}"
        raise ValueError(msg)
    target.parent.mkdir(parents=True, exist_ok=True)
    arr = precompute_inverse_lut(table)
    np.save(target, arr, allow_pickle=False)
    return target


_NDArrayU8 = npt.NDArray[np.uint8]


def rgb_to_cmyk_lut_arr(
    rgb_row: Buffer, width: int, table: ColorTable = DEFAULT_TABLE
) -> tuple[_NDArrayU8, _NDArrayU8, _NDArrayU8, _NDArrayU8]:
    """Like :func:`rgb_to_cmyk_lut` but returns ndarrays directly.

    Lets callers in the hot path avoid a bytes→ndarray roundtrip.

    Returns:
        (k, c, m, y) uint8 arrays of length `width`.
    """
    inv = inverse_lut(table)
    if inv is not None and HAS_CYTHON_COLOR:
        planes = np.empty((4, width), dtype=np.uint8)
        gather_kcmy(rgb_row, width, inv.reshape(-1), planes[0], planes[1], planes[2], planes[3])
        return planes[0], planes[1], planes[2], planes[3]
    if inv is not None:
        rgb = np.frombuffer(rgb_row, dtype=np.uint8, count=width * 3).reshape(width, 3)
        idx = (rgb[:, 0].astype(np.uint32) << 16) | (rgb[:, 1].astype(np.uint32) << 8) | rgb[:, 2].astype(np.uint32)
        kcmy = np.ascontiguousarray(inv.reshape(-1, 4)[idx])
        return kcmy[:, 0], kcmy[:, 1], kcmy[:, 2], kcmy[:, 3]
    if HAS_CYTHON_COLOR:
        planes = np.empty((4, width), dtype=np.uint8)
        interp_kcmy(rgb_row, width, *interp_arrays(table), planes[0], planes[1], planes[2], planes[3])
        return planes[0], planes[1], planes[2], planes[3]
    _warn_interp_fallback(table)
    return _rgb_to_cmyk_interp_arr(rgb_row, width, table)


@functools.cache
def _warn_interp_fallback(table: ColorTable) -> None:
    """Log once per process and table that the ~20x slower numpy interpolation is in use."""
    logger.warning(
        "No inverse LUT or native interpolation for colour table %s; using numpy per-pixel "
        "interpolation (much slower). Build the Cython modules or precompute the inverse LUT.",
        table.name,
    )


def rgb_to_cmyk_lut(
    rgb_row: Buffer, width: int, table: ColorTable = DEFAULT_TABLE
) -> tuple[bytes, bytes, bytes, bytes]:
    """Convert one RGB scanline to CMYK using the driver's 3D LUT.

    Uses the precomputed inverse LUT when present; otherwise interpolates
    per pixel, natively when the Cython module is built.

    Args:
        rgb_row: Raw RGB pixel data (width * 3 bytes).
        width: Number of pixels in the row.
        table: the colour grid to use (see `ColorTable`).

    Returns:
        (k_arr, c_arr, m_arr, y_arr) each of `width` bytes.
        Values use pixel-brightness convention (0=full ink, 255=no ink)
        matching dither_channel_1bpp input.
    """
    k, c, m, y = rgb_to_cmyk_lut_arr(rgb_row, width, table)
    return k.tobytes(), c.tobytes(), m.tobytes(), y.tobytes()


def _rgb_to_cmyk_interp_arr(
    rgb_row: Buffer, width: int, table: ColorTable = DEFAULT_TABLE
) -> tuple[_NDArrayU8, _NDArrayU8, _NDArrayU8, _NDArrayU8]:
    """Per-pixel tetrahedral interpolation through the 17x17x17 LUT grid.

    Returns:
        (k, c, m, y) in pixel-brightness convention (0=full ink, 255=no ink).
    """
    lut, interp = _load_data(table)

    rgb = np.frombuffer(rgb_row, dtype=np.uint8, count=width * 3).reshape(width, 3)

    r_hi = _HI[rgb[:, 0]]
    r_frac = _FRAC[rgb[:, 0]]
    g_hi = _HI[rgb[:, 1]]
    g_frac = _FRAC[rgb[:, 1]]
    b_hi = _HI[rgb[:, 2]]
    b_frac = _FRAC[rgb[:, 2]]

    # Look up interpolation weights: interp[b_frac][(g_frac * 17 + r_frac)] → 9 bytes
    entry_idx = g_frac * _LUT_DIM + r_frac
    weights = interp[b_frac, entry_idx]

    total = weights[:, 0].astype(np.int32)
    if np.any(total == 0):
        logger.error("Zero interpolation weight detected, LUT data may be corrupt")
        total = np.maximum(total, 1)
    w = weights[:, 1:9].astype(np.int32)

    # Cube base index + 8 corner gathering
    cube_base = r_hi * 289 + g_hi * 17 + b_hi
    corner_idx = cube_base[:, None] + _CORNER_OFFSETS
    np.clip(corner_idx, 0, _LUT_ENTRIES - 1, out=corner_idx)
    corners = lut[corner_idx]

    # Per channel, max accumulator value is 255*255*8 = 520200 — fits int32.
    accum = (w[:, :, None] * corners).sum(axis=1)

    rounding = (total >> 1)[:, None]
    cmyk = (accum + rounding) // total[:, None]

    is_black = (rgb[:, 0] == 0) & (rgb[:, 1] == 0) & (rgb[:, 2] == 0)
    is_white = (rgb[:, 0] == 255) & (rgb[:, 1] == 255) & (rgb[:, 2] == 255)
    if np.any(is_black):
        cmyk[is_black] = black_ink(table, lut)
    if np.any(is_white):
        cmyk[is_white] = 0

    # Ink amount (0=no ink, 255=full) → pixel-brightness (0=full ink, 255=no ink).
    result = np.ascontiguousarray((255 - cmyk).astype(np.uint8))

    return result[:, 3], result[:, 0], result[:, 1], result[:, 2]

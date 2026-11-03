"""3D colour LUT interpolation for the Brother HL-4150CDN.

Maps RGB input to CMYK ink values via a 17x17x17 grid plus tetrahedral
interpolation tables. Which of the 18 grids applies is decided like
`lookup_color_transform_table` in the original driver (see `ColorTable`).
"""

import functools
import logging
from collections.abc import Buffer
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
import numpy.typing as npt

from brother_xl2hb import _native

_color_fast = _native.load("_color_fast")

logger = logging.getLogger(__name__)

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


@dataclass(frozen=True, slots=True)
class ColorTable:
    """One of the colour grids `lookup_color_transform_table` chooses from.

    Built by `PrinterModel.grid`, which knows where the model's grids are.
    As the key of every cache here, `data_dir` keeps the grids of different
    models apart.

    Attributes:
        profile: `rgb` for Normal colour matching, `srgb` for Vivid, `cmyk`
            for colour matching None.
        improve_gray: the ImpGray=ON grids (BRGray).
        variant: the model's grid variant, e.g. `density2` for toner save on
            the HL-4150CDN; `default` otherwise.
        rich_black: pure black takes grid entry 0 (C, M, Y and K) instead of
            K only; the original does this with BREnhanceBlkPrt in the
            rgb/srgb profiles (`lut_selection` 0 in `load_color_profile`).
        data_dir: the model's colour data directory (grids and
            interpolation tables).
        generated_fallback: without the extracted grid, `color_lut_gen`'s
            parametric grid stands in (with a warning); otherwise loading fails.
    """

    profile: Profile = "rgb"
    improve_gray: bool = False
    variant: str = "default"
    rich_black: bool = False
    data_dir: Path = field(kw_only=True)
    generated_fallback: bool = field(default=False, kw_only=True)

    @property
    def name(self) -> str:
        """Blob name stem, e.g. `rgb_ig_density2` for `rgb_ig_density2_lut.bin`."""
        return f"{self.profile}{'_ig' if self.improve_gray else ''}_{self.variant}"


def _load_lut(table: ColorTable) -> npt.NDArray[np.int32]:
    """Load the 3D color grid of `table` as unpacked CMYK channels.

    The binary file contains packed int32 pairs (cm_packed, yk_packed).
    We unpack at load time into (4913, 4) int32 array [C, M, Y, K]
    to avoid packed arithmetic and int64 at runtime.

    If the binary file is missing and the table allows it
    (`generated_fallback`), the generated rgb/srgb default grids stand in
    for every variant (with a warning, since the output then differs from
    the original's).

    Returns:
        Unpacked LUT of shape (_LUT_ENTRIES, 4), columns [C, M, Y, K].

    Raises:
        FileNotFoundError: If the binary file is missing and the table has
            no generated fallback.
        ValueError: If the binary LUT file has an unexpected size.
    """
    path = table.data_dir / f"{table.name}_lut.bin"
    if not path.exists():
        if not table.generated_fallback:
            msg = f"colour grid {path} missing; run scripts/extract_blobs.sh {table.data_dir.name}"
            raise FileNotFoundError(msg)
        from brother_xl2hb.color.color_lut_gen import generate_rgb_default_lut, generate_srgb_default_lut

        generated = f"{'srgb' if table.profile == 'srgb' else 'rgb'}_default"
        logger.warning(
            "Colour grid %s missing in %s (run scripts/extract_blobs.sh %s); using a generated %s grid, "
            "output will differ from the original driver",
            path.name,
            table.data_dir,
            table.data_dir.name,
            generated,
        )
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


def _load_interp_tables(data_dir: Path) -> npt.NDArray[np.uint8]:
    """Load the interpolation weight tables from `data_dir` (17 tables, each 17*17*9 bytes).

    Falls back to analytical generation, which gives the same tables, if
    the binary file is missing.

    Returns:
        Array of shape (_NUM_INTERP_TABLES, _LUT_DIM*_LUT_DIM, 9), uint8.

    Raises:
        ValueError: If the binary file has an unexpected size.
    """
    path = data_dir / "interp_tables.bin"
    if not path.exists():
        # The generated tables equal Brother's byte for byte
        # (test_color_lut_gen.py), so the output does not change.
        logger.info("Interpolation tables %s not found, generating them", path)
        from brother_xl2hb.color.color_lut_gen import generate_interp_tables

        return generate_interp_tables()
    data = path.read_bytes()
    expected = _INTERP_TABLE_SIZE * _NUM_INTERP_TABLES
    if len(data) != expected:
        msg = f"Interp tables size {len(data)}, expected {expected}"
        raise ValueError(msg)
    return np.frombuffer(data, dtype=np.uint8).reshape(_NUM_INTERP_TABLES, _LUT_DIM * _LUT_DIM, 9)


# Ink for pure black (R=G=B=0) unless the table asks for rich black.
_K_PRESET = np.array([0, 0, 0, 255], dtype=np.int32)


@functools.lru_cache(maxsize=8)
def _load_data(table: ColorTable) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.uint8]]:
    """Load the grid of `table` and the interpolation tables (cached).

    Returns:
        Tuple of (lut, interp_tables); see `_load_lut` and `_load_interp_tables`.
    """
    return _load_lut(table), _load_interp_tables(table.data_dir)


def black_ink(table: ColorTable, lut: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
    """C, M, Y, K ink for pure black: grid entry 0 for rich black, else K only.

    Returns:
        int32 array of 4 ink values.
    """
    return lut[0].copy() if table.rich_black else _K_PRESET.copy()


class InterpArrays(NamedTuple):
    """A colour table in the compact form the native interpolation (`interp_pixel`) reads."""

    grid: npt.NDArray[np.uint8]
    """4913 * 4 bytes: C, M, Y, K ink of each grid entry."""
    weights: npt.NDArray[np.uint8]
    """4913 * 8 bytes: per weight entry the 4 non-zero corner weights, their
    total, and the 4 corner numbers (0-7) packed 3 bits each into bytes 5-6."""
    recip: npt.NDArray[np.uint64]
    """256 values ceil(2**32 / total), 2**32 for a total of 0 (divides by 1)."""
    black: npt.NDArray[np.int32]
    """C, M, Y, K ink for pure black."""


# Weight entries have at most this many non-zero corners (tetrahedral
# interpolation: one tetrahedron of the cube).
_INTERP_CORNERS = 4


@functools.lru_cache(maxsize=32)
def interp_arrays(table: ColorTable) -> InterpArrays:
    """The grid, interpolation weights and black ink of `table` for the native interpolation.

    Built once per table and shared (read-only). The compact weights keep
    only the non-zero corners, and the quotient by the total becomes a
    multiplication; both give the same values as `_rgb_to_cmyk_interp_arr`.

    Returns:
        The compact arrays.

    Raises:
        ValueError: If the grid has values outside 0-255 or a weight entry
            more than 4 non-zero corners; never for Brother's tables or the
            generated ones.
    """
    lut, interp = _load_data(table)
    if lut.min() < 0 or lut.max() > 255:
        msg = f"colour grid {table.name} has values outside 0-255"
        raise ValueError(msg)
    entries = interp.reshape(-1, 9)
    corner_weights = entries[:, 1:]
    nonzero = corner_weights != 0
    if int(nonzero.sum(axis=1).max()) > _INTERP_CORNERS:
        msg = f"interpolation weights of {table.name} have more than {_INTERP_CORNERS} corners"
        raise ValueError(msg)
    # Corner numbers of the non-zero weights first (stable, so in weight order), then zero weights.
    order = np.argsort(~nonzero, axis=1, kind="stable")[:, :_INTERP_CORNERS]
    weights = np.zeros((entries.shape[0], 8), dtype=np.uint8)
    weights[:, :_INTERP_CORNERS] = np.take_along_axis(corner_weights, order, axis=1)
    weights[:, 4] = entries[:, 0]
    packed = (order << (3 * np.arange(_INTERP_CORNERS))).sum(axis=1)
    weights[:, 5] = packed & 0xFF
    weights[:, 6] = packed >> 8

    divisors = np.maximum(np.arange(256, dtype=np.uint64), np.uint64(1))
    recip = ((np.uint64(1) << np.uint64(32)) + divisors - np.uint64(1)) // divisors
    _check_recip(recip, weights)

    grid = np.ascontiguousarray(lut, dtype=np.uint8).reshape(-1)
    black = black_ink(table, lut)
    for array in (grid, weights, recip, black):
        array.flags.writeable = False
    return InterpArrays(grid, weights.reshape(-1), recip, black)


def _check_recip(recip: npt.NDArray[np.uint64], weights: npt.NDArray[np.uint8]) -> None:
    """Check (n * recip[t]) >> 32 == n // t for every numerator the interpolation can reach.

    A numerator is the weighted sum of four grid values (<= 255 * the sum
    of the weights) plus half the total.

    Raises:
        ValueError: If a quotient would differ; ceil(2**32 / t) never does
            for these numerators.
    """
    sums = weights[:, :_INTERP_CORNERS].astype(np.int64).sum(axis=1)
    for total in np.unique(weights[:, 4]).tolist():
        divisor = max(total, 1)
        top = int(sums[weights[:, 4] == total].max()) * 255 + (total >> 1)
        n = np.arange(top + 1, dtype=np.uint64)
        if not np.array_equal((n * recip[total]) >> np.uint64(32), n // np.uint64(divisor)):
            msg = f"reciprocal of {total} is not exact up to {top}"
            raise ValueError(msg)


_NDArrayU8 = npt.NDArray[np.uint8]


def rgb_to_cmyk_lut_arr(
    rgb_row: Buffer, width: int, table: ColorTable
) -> tuple[_NDArrayU8, _NDArrayU8, _NDArrayU8, _NDArrayU8]:
    """Like :func:`rgb_to_cmyk_lut` but returns ndarrays directly.

    Lets callers in the hot path avoid a bytes→ndarray roundtrip.

    Returns:
        (k, c, m, y) uint8 arrays of length `width`.
    """
    if _color_fast is not None:
        planes = np.empty((4, width), dtype=np.uint8)
        _color_fast.interp_kcmy(rgb_row, width, *interp_arrays(table), planes[0], planes[1], planes[2], planes[3])
        return planes[0], planes[1], planes[2], planes[3]
    _warn_interp_fallback(table)
    return _rgb_to_cmyk_interp_arr(rgb_row, width, table)


@functools.cache
def _warn_interp_fallback(table: ColorTable) -> None:
    """Log once per process and table that the ~20x slower numpy interpolation is in use."""
    logger.warning(
        "No native interpolation for colour table %s; using numpy per-pixel "
        "interpolation (much slower). Build the Cython modules.",
        table.name,
    )


def rgb_to_cmyk_lut(rgb_row: Buffer, width: int, table: ColorTable) -> tuple[bytes, bytes, bytes, bytes]:
    """Convert one RGB scanline to CMYK using the driver's 3D LUT.

    Interpolates per pixel, natively when the Cython module is built.

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
    rgb_row: Buffer, width: int, table: ColorTable
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

    # Cube base index + 8 corner gathering. _HI tops out at 15, so the
    # largest index is 15 * 307 + 307 = 4912 = _LUT_ENTRIES - 1.
    cube_base = r_hi * 289 + g_hi * 17 + b_hi
    corner_idx = cube_base[:, None] + _CORNER_OFFSETS
    corners = lut[corner_idx]

    # Per channel the weights sum to the entry's total (at most 229), so the
    # accumulator stays below 229 * 255 = 58395 — fits int32.
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

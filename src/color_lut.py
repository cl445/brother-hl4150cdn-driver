"""3D colour LUT interpolation for the Brother HL-4150CDN.

Maps RGB input to CMYK ink values via a 17x17x17 grid plus tetrahedral
interpolation tables. The default profile (`rgb_default_lut.bin`) is
used for the "Normal" colour-matching mode; `srgb_default_lut.bin`
provides a higher-saturation alternative.
"""

import functools
import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt

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


def _load_lut() -> npt.NDArray[np.int32]:
    """Load the 3D color LUT as unpacked CMYK channels.

    The binary file contains packed int32 pairs (cm_packed, yk_packed).
    We unpack at load time into (4913, 4) int32 array [C, M, Y, K]
    to avoid packed arithmetic and int64 at runtime.

    Falls back to parametric generation if the binary file is missing.

    Returns:
        Unpacked LUT of shape (_LUT_ENTRIES, 4), columns [C, M, Y, K].

    Raises:
        ValueError: If the binary LUT file has an unexpected size.
    """
    path = _DATA_DIR / "rgb_default_lut.bin"
    if not path.exists():
        logger.info("LUT binary not found, generating from parametric model")
        from color_lut_gen import generate_rgb_default_lut

        return generate_rgb_default_lut()
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


# K-preset values for pure black (R=G=B=0)
# The driver has two modes controlled by lut_selection:
#   lut_selection=0: reads from LUT[0,0,0] → C=83, M=55, Y=65, K=255 (rich black)
#   lut_selection≠0: uses C=0, M=0, Y=0, K=255 (pure K black)
# Captures show the standard driver uses pure K black.
_K_PRESET = np.array([0, 0, 0, 255], dtype=np.int32)

# Precomputed full RGB→KCMY lookup table.
# Last axis order is K, C, M, Y in pixel-brightness convention
# (0 = full ink, 255 = no ink).
INVERSE_LUT_PATH = _DATA_DIR / "inverse_lut.npy"
_INVERSE_LUT_SHAPE = (256, 256, 256, 4)


@functools.lru_cache(maxsize=1)
def _load_data() -> tuple[npt.NDArray[np.int32], npt.NDArray[np.uint8]]:
    """Load LUT and interpolation tables (cached singleton).

    Returns:
        Tuple of (lut, interp_tables); see `_load_lut` and `_load_interp_tables`.
    """
    return _load_lut(), _load_interp_tables()


@functools.lru_cache(maxsize=1)
def _load_inverse_lut() -> npt.NDArray[np.uint8] | None:
    """Load the precomputed RGB→KCMY inverse LUT into RAM, or None if absent.

    Returns:
        Array of shape (256, 256, 256, 4) uint8, or None when the cache
        file is missing or has an unexpected shape.
    """
    path = INVERSE_LUT_PATH
    if not path.exists():
        return None
    try:
        arr = np.load(path, allow_pickle=False)
    except (ValueError, OSError) as exc:
        logger.warning("Failed to load inverse LUT %s: %s", path, exc)
        return None
    if arr.shape != _INVERSE_LUT_SHAPE or arr.dtype != np.uint8:
        logger.warning("Inverse LUT %s has unexpected shape %s/%s, ignoring", path, arr.shape, arr.dtype)
        return None
    return arr


def precompute_inverse_lut() -> npt.NDArray[np.uint8]:
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
        k_b, c_b, m_b, y_b = _rgb_to_cmyk_interp(gb_flat.tobytes(), 65536)
        out[r, :, :, 0] = np.frombuffer(k_b, dtype=np.uint8).reshape(256, 256)
        out[r, :, :, 1] = np.frombuffer(c_b, dtype=np.uint8).reshape(256, 256)
        out[r, :, :, 2] = np.frombuffer(m_b, dtype=np.uint8).reshape(256, 256)
        out[r, :, :, 3] = np.frombuffer(y_b, dtype=np.uint8).reshape(256, 256)
    return out


def write_inverse_lut(path: Path | None = None) -> Path:
    """Precompute the inverse LUT and save it to ``path`` (default INVERSE_LUT_PATH).

    Returns:
        The path the array was written to.
    """
    target = path or INVERSE_LUT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    arr = precompute_inverse_lut()
    np.save(target, arr, allow_pickle=False)
    return target


def rgb_to_cmyk_lut(rgb_row: bytes, width: int) -> tuple[bytes, bytes, bytes, bytes]:
    """Convert one RGB scanline to CMYK using the driver's 3D LUT.

    Uses the precomputed inverse LUT when present; otherwise falls back
    to per-pixel tetrahedral interpolation.

    Args:
        rgb_row: Raw RGB pixel data (width * 3 bytes).
        width: Number of pixels in the row.

    Returns:
        (k_arr, c_arr, m_arr, y_arr) each of `width` bytes.
        Values use pixel-brightness convention (0=full ink, 255=no ink)
        matching dither_channel_1bpp input.
    """
    inv = _load_inverse_lut()
    if inv is not None:
        rgb = np.frombuffer(rgb_row, dtype=np.uint8, count=width * 3).reshape(width, 3)
        idx = (
            (rgb[:, 0].astype(np.uint32) << 16)
            | (rgb[:, 1].astype(np.uint32) << 8)
            | rgb[:, 2].astype(np.uint32)
        )
        kcmy = inv.reshape(-1, 4)[idx]
        return (
            kcmy[:, 0].tobytes(),
            kcmy[:, 1].tobytes(),
            kcmy[:, 2].tobytes(),
            kcmy[:, 3].tobytes(),
        )
    return _rgb_to_cmyk_interp(rgb_row, width)


def _rgb_to_cmyk_interp(rgb_row: bytes, width: int) -> tuple[bytes, bytes, bytes, bytes]:
    """Per-pixel tetrahedral interpolation through the 17×17×17 LUT grid."""
    lut, interp = _load_data()

    rgb = np.frombuffer(rgb_row, dtype=np.uint8, count=width * 3).reshape(width, 3)

    # Split into grid index (hi) and fractional weight (frac) via pre-computed tables.
    # Replaces 6 np.where calls with 6 table lookups (fancy indexing).
    r_hi = _HI[rgb[:, 0]]
    r_frac = _FRAC[rgb[:, 0]]
    g_hi = _HI[rgb[:, 1]]
    g_frac = _FRAC[rgb[:, 1]]
    b_hi = _HI[rgb[:, 2]]
    b_frac = _FRAC[rgb[:, 2]]

    # Look up interpolation weights: interp[b_frac][(g_frac * 17 + r_frac)] → 9 bytes
    entry_idx = g_frac * _LUT_DIM + r_frac  # (width,)
    weights = interp[b_frac, entry_idx]  # (width, 9)

    total = weights[:, 0].astype(np.int32)  # (width,)
    if np.any(total == 0):
        logger.error("Zero interpolation weight detected, LUT data may be corrupt")
        total = np.maximum(total, 1)
    w = weights[:, 1:9].astype(np.int32)  # (width, 8)

    # Cube base index + 8 corner gathering
    cube_base = r_hi * 289 + g_hi * 17 + b_hi  # (width,)
    corner_idx = cube_base[:, None] + _CORNER_OFFSETS  # (width, 8)
    np.clip(corner_idx, 0, _LUT_ENTRIES - 1, out=corner_idx)
    corners = lut[corner_idx]  # (width, 8, 4) — unpacked [C, M, Y, K]

    # Weighted accumulation in int32 (max per channel: 255*255*8 = 520200, fits int32).
    # w (width, 8, 1) * corners (width, 8, 4) → sum axis 1 → (width, 4)
    accum = (w[:, :, None] * corners).sum(axis=1)  # (width, 4)

    # Normalize with rounding bias
    rounding = (total >> 1)[:, None]  # (width, 1)
    cmyk = (accum + rounding) // total[:, None]  # (width, 4)

    # Handle special cases via boolean indexing (replaces 8 np.where calls)
    is_black = (rgb[:, 0] == 0) & (rgb[:, 1] == 0) & (rgb[:, 2] == 0)
    is_white = (rgb[:, 0] == 255) & (rgb[:, 1] == 255) & (rgb[:, 2] == 255)
    if np.any(is_black):
        cmyk[is_black] = _K_PRESET
    if np.any(is_white):
        cmyk[is_white] = 0

    # Convert ink amount (0=no ink, 255=full) → pixel-brightness (0=full ink, 255=no ink)
    result = (255 - cmyk).astype(np.uint8)  # (width, 4)

    return result[:, 3].tobytes(), result[:, 0].tobytes(), result[:, 1].tobytes(), result[:, 2].tobytes()

"""3D colour LUT interpolation for the Brother HL-4150CDN.

Maps RGB input to CMYK ink values via a 17x17x17 grid plus tetrahedral
interpolation tables. Which of the 18 grids applies is decided like
`lookup_color_transform_table` in the original driver (see `ColorTable`).
"""

import functools
import hashlib
import logging
import time
from collections.abc import Buffer, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

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

    Built by `PrinterModel.grid`, which knows where the model's grids are
    and which of them have a precomputed inverse LUT. As the key of every
    cache here, `data_dir` keeps the grids of different models apart.

    Attributes:
        profile: `rgb` for Normal colour matching, `srgb` for Vivid, `cmyk`
            for colour matching None.
        improve_gray: the ImpGray=ON grids (BRGray).
        variant: the model's grid variant, e.g. `density2` for toner save on
            the HL-4150CDN; `default` otherwise.
        rich_black: pure black takes grid entry 0 (C, M, Y and K) instead of
            K only; the original does this with BREnhanceBlkPrt in the
            rgb/srgb profiles (`lut_selection` 0 in `load_color_profile`).
        data_dir: the model's colour data directory (grids, interpolation
            tables, inverse LUTs).
        precomputed: an inverse LUT is installed for this table (see
            `inverse_lut_path`); every other table is interpolated per pixel.
        generated_fallback: without the extracted grid, `color_lut_gen`'s
            parametric grid stands in (with a warning); otherwise loading fails.
    """

    profile: Profile = "rgb"
    improve_gray: bool = False
    variant: str = "default"
    rich_black: bool = False
    data_dir: Path = field(kw_only=True)
    precomputed: bool = field(default=False, kw_only=True)
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

# Precomputed full RGB→KCMY lookup tables, one file per table in its
# data_dir (see `inverse_lut_path`). Last axis order is K, C, M, Y in
# pixel-brightness convention (0 = full ink, 255 = no ink).
_INVERSE_LUT_SHAPE = (256, 256, 256, 4)
# Part of the source digest: bump when the interpolation itself changes, so
# inverse LUTs computed by the old code are no longer used.
_INVERSE_LUT_FORMAT = 1


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


def interp_arrays(
    table: ColorTable,
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


@functools.lru_cache(maxsize=32)
def source_digest(table: ColorTable) -> str:
    """Hash of everything the inverse LUT of `table` is computed from.

    Covers the grid and the interpolation weights as loaded (the extracted
    blobs, or the generated fallbacks when they are missing), the black ink
    and `_INVERSE_LUT_FORMAT`. An inverse LUT computed before
    `scripts/extract_blobs.sh` thus no longer matches once the real grids
    are in place. Costs one read of the 39 KB grid, once per process.

    Returns:
        The first 16 hex digits of the SHA-256.
    """
    lut, interp = _load_data(table)
    digest = hashlib.sha256(f"inverse-lut-v{_INVERSE_LUT_FORMAT}:{table.name}\n".encode())
    digest.update(np.ascontiguousarray(lut, dtype="<i4").tobytes())
    digest.update(np.ascontiguousarray(interp, dtype=np.uint8).tobytes())
    digest.update(black_ink(table, lut).astype("<i4").tobytes())
    return digest.hexdigest()[:16]


def inverse_lut_path(table: ColorTable, directory: Path | None = None) -> Path | None:
    """Where the inverse LUT of `table` is installed: ``inverse_lut_<name>-<digest>.npy``.

    The name carries `source_digest`, so a file computed from other grid
    data is simply not found.

    Args:
        table: the colour table.
        directory: where to look; the table's `data_dir` by default.

    Returns:
        The path, or None for tables without a precomputed inverse LUT.
    """
    if not table.precomputed:
        return None
    return _digest_path(table, table.data_dir if directory is None else directory)


def _digest_path(table: ColorTable, directory: Path) -> Path:
    """``<directory>/inverse_lut_<name>-<source digest>.npy``.

    Returns:
        The path; see `inverse_lut_path`.
    """
    return directory / f"inverse_lut_{table.name}-{source_digest(table)}.npy"


def _stale_inverse_luts(table: ColorTable, directory: Path) -> list[Path]:
    """Inverse LUT files for `table` in `directory` whose name has another source digest.

    Includes the files without a digest from before it was recorded
    (``inverse_lut.npy``, ``inverse_lut_srgb.npy``).

    Returns:
        Sorted paths.
    """
    current = _digest_path(table, directory)
    stale = [p for p in directory.glob(f"inverse_lut_{table.name}-*.npy") if p != current]
    if table.variant == "default" and not table.improve_gray:
        legacy = directory / ("inverse_lut.npy" if table.profile == "rgb" else f"inverse_lut_{table.profile}.npy")
        if legacy.exists():
            stale.append(legacy)
    return sorted(stale)


@functools.lru_cache(maxsize=32)
def inverse_lut(table: ColorTable) -> npt.NDArray[np.uint8] | None:
    """Memory-map the installed RGB→KCMY inverse LUT of `table`, or None if absent.

    Mapped read-only instead of loaded: each CUPS job is a fresh process,
    and reading all 64 MiB up front costs 1-3 s on a Pi. With the mapping
    only the pages for colours actually on the page are read, and the OS
    page cache shares them between jobs.

    Only the file for the current `source_digest` is used. One computed
    from other grid data (e.g. from the generated fallback grid, before
    `scripts/extract_blobs.sh`) is ignored with a warning.

    Returns:
        Array of shape (256, 256, 256, 4) uint8, or None when the cache
        file is missing, stale or has an unexpected shape.
    """
    path = inverse_lut_path(table)
    if path is None:
        return None
    if not path.exists():
        if stale := _stale_inverse_luts(table, path.parent):
            logger.warning(
                "Inverse LUT %s was computed from other colour grid data, ignoring it; "
                "re-run cups/install.sh (or write_inverse_luts) to recompute",
                ", ".join(p.name for p in stale),
            )
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


def precompute_inverse_lut(table: ColorTable) -> npt.NDArray[np.uint8]:
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


def write_inverse_lut(table: ColorTable, path: Path | None = None) -> Path:
    """Precompute the inverse LUT of `table` and save it to ``path`` (default `inverse_lut_path`).

    Written to a temporary file and renamed, so a job running meanwhile
    never maps a half-written LUT.

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
    tmp = target.with_name(f".{target.name}.tmp")
    try:
        with tmp.open("wb") as f:
            np.save(f, arr, allow_pickle=False)
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)
    return target


def write_inverse_luts(tables: Iterable[ColorTable], directory: Path | None = None) -> list[Path]:
    """Write the inverse LUTs of `tables` and remove stale ones.

    What `cups/install.sh` runs, with the model's `inverse_lut_tables`.
    Stale files are those of these tables with another source digest, plus
    the digest-less ones from older versions.

    Args:
        tables: the tables to precompute.
        directory: target directory; each table's `data_dir` by default.

    Returns:
        The written paths.
    """
    tables = tuple(tables)
    written = []
    for table in tables:
        target = _digest_path(table, table.data_dir if directory is None else directory)
        logger.info("Precomputing %s inverse LUT to %s ...", table.name, target)
        t0 = time.perf_counter()
        write_inverse_lut(table, target)
        logger.info("Wrote %.1f MiB in %.1fs", target.stat().st_size / 1024 / 1024, time.perf_counter() - t0)
        written.append(target)
    for table in tables:
        for stale in _stale_inverse_luts(table, table.data_dir if directory is None else directory):
            logger.info("Removing stale inverse LUT %s", stale)
            stale.unlink(missing_ok=True)
    return written


_NDArrayU8 = npt.NDArray[np.uint8]


def rgb_to_cmyk_lut_arr(
    rgb_row: Buffer, width: int, table: ColorTable
) -> tuple[_NDArrayU8, _NDArrayU8, _NDArrayU8, _NDArrayU8]:
    """Like :func:`rgb_to_cmyk_lut` but returns ndarrays directly.

    Lets callers in the hot path avoid a bytes→ndarray roundtrip.

    Returns:
        (k, c, m, y) uint8 arrays of length `width`.
    """
    inv = inverse_lut(table)
    if inv is not None and _color_fast is not None:
        planes = np.empty((4, width), dtype=np.uint8)
        _color_fast.gather_kcmy(rgb_row, width, inv.reshape(-1), planes[0], planes[1], planes[2], planes[3])
        return planes[0], planes[1], planes[2], planes[3]
    if inv is not None:
        rgb = np.frombuffer(rgb_row, dtype=np.uint8, count=width * 3).reshape(width, 3)
        idx = (rgb[:, 0].astype(np.uint32) << 16) | (rgb[:, 1].astype(np.uint32) << 8) | rgb[:, 2].astype(np.uint32)
        kcmy = np.ascontiguousarray(inv.reshape(-1, 4)[idx])
        return kcmy[:, 0], kcmy[:, 1], kcmy[:, 2], kcmy[:, 3]
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
        "No inverse LUT or native interpolation for colour table %s; using numpy per-pixel "
        "interpolation (much slower). Build the Cython modules or precompute the inverse LUT.",
        table.name,
    )


def rgb_to_cmyk_lut(rgb_row: Buffer, width: int, table: ColorTable) -> tuple[bytes, bytes, bytes, bytes]:
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

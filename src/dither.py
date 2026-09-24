"""Ordered dithering for the Brother HL-4150CDN.

Converts continuous-tone intensity values to 1bpp or 4bpp planes.
Loads BRCD cache files (per-channel threshold tables) when available
and falls back to a standard Bayer matrix otherwise.

1bpp mode: binary on/off dots, used in Normal mode.
4bpp mode: 16 intensity levels (0-15) per pixel, nibble-packed output,
used in Fine mode.

Both compare the ink amount against a threshold matrix tiled to the
page width, vectorised with numpy (or `_dither_fast` for 1bpp).
"""

import functools
import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

try:
    from _dither_fast import dither_row_1bpp  # type: ignore[import-not-found]

    HAS_CYTHON_DITHER = True
except ImportError:
    HAS_CYTHON_DITHER = False

logger = logging.getLogger(__name__)


@dataclass
class DitherChannel:
    """Ordered-dither threshold matrix for one color channel."""

    width: int  # matrix width in pixels (e.g. 32)
    height: int  # matrix height in pixels (e.g. 32)
    # shape (height, width), dtype uint8 — threshold[y][x]: dot if ink > threshold
    threshold_matrix: npt.NDArray[np.uint8] = field(repr=False)

    # Threshold matrix tiled to the target page width, keyed by width.
    _tiled_cache: dict[int, npt.NDArray[np.uint8]] = field(default_factory=dict, init=False, repr=False)

    def tiled_thresholds(self, width: int) -> npt.NDArray[np.uint8]:
        """Return the threshold matrix tiled to `width`, cached per width.

        Returns:
            Array of shape (height, width) with the repeated threshold matrix.
        """
        cached = self._tiled_cache.get(width)
        if cached is not None:
            return cached
        repeats = (width + self.width - 1) // self.width
        tiled = np.tile(self.threshold_matrix, (1, repeats))[:, :width].copy()  # (height, width)
        self._tiled_cache[width] = tiled
        return tiled


def _bayer_matrix(n: int) -> list[list[int]]:
    """Generate a 2^n x 2^n Bayer ordered dither threshold matrix.

    Returns:
        2D list with values in range [0, 2^(2n) - 1].
    """
    if n == 1:
        return [[0, 2], [3, 1]]

    prev = _bayer_matrix(n - 1)
    size = len(prev)
    result = [[0] * (2 * size) for _ in range(2 * size)]
    for y in range(size):
        for x in range(size):
            val = prev[y][x]
            result[y][x] = 4 * val
            result[y][x + size] = 4 * val + 2
            result[y + size][x] = 4 * val + 3
            result[y + size][x + size] = 4 * val + 1
    return result


def _normalize_matrix(matrix: list[list[int]], width: int, height: int) -> list[list[int]]:
    """Normalize Bayer matrix so each row independently covers [0, 254].

    The raw Bayer matrix has per-row threshold values clustered in bands,
    which prevents fine-grained coverage control within a single scanline.
    Per-row rank normalization maps each row's w values to evenly spaced
    thresholds in [0, 254], preserving the spatial ordering within each row
    while ensuring that coverage increases monotonically with ink level.

    Returns:
        New 2D list of the same shape with rank-normalized thresholds.
    """
    normalized = [[0] * width for _ in range(height)]
    for y in range(height):
        row_vals = [(matrix[y][x], x) for x in range(width)]
        row_vals.sort()
        for rank, (_, x) in enumerate(row_vals):
            normalized[y][x] = rank * 254 // (width - 1)
    return normalized


def _extract_threshold_matrix(patterns: list[bytes], width: int, height: int) -> npt.NDArray[np.uint8]:
    """Extract the threshold matrix from BRCD per-ink-level dot patterns.

    For each position (y, x) in the dither matrix, finds the largest ink
    value where the bit is NOT set. Since patterns are monotonic (once a bit
    appears at ink level k, it stays for all higher levels), we find the
    first ink where each bit turns on.

    Returns:
        Threshold matrix of shape (height, width), uint8.
    """
    total = height * width

    # Stack all 256 patterns and unpack to bits: (256, total)
    pat_arr = np.frombuffer(b"".join(patterns), dtype=np.uint8).reshape(256, -1)
    bits = np.unpackbits(pat_arr, axis=1)[:, :total]  # (256, total)

    # For each position, find first ink level where bit is set
    # argmax on bool array returns index of first True
    ever_set = np.any(bits, axis=0)  # (total,)
    first_set = np.argmax(bits, axis=0)  # (total,)

    # threshold = first_set - 1 where bit is ever set, else 255
    thresholds = np.where(ever_set, first_set - 1, 255).astype(np.uint8)
    return thresholds.reshape(height, width)


def dither_load_brcd(path: str) -> DitherChannel:
    """Load dither table from a BRCD cache file.

    File format: 4-byte magic "BRCD", 1-byte version ('0'),
    1-byte reserved, 2-byte LE width, 2-byte LE height,
    then 256 x row_bytes of pattern data.

    Returns:
        DitherChannel with the threshold matrix derived from the patterns.

    Raises:
        ValueError: If the magic bytes are wrong or pattern data is truncated.
    """
    with Path(path).open("rb") as f:
        magic = f.read(4)
        if magic != b"BRCD":
            msg = f"Not a BRCD file: {magic!r}"
            raise ValueError(msg)
        f.read(2)  # version + reserved
        width, height = struct.unpack("<HH", f.read(4))
        row_bytes = (width * height + 7) // 8
        patterns = []
        for i in range(256):
            pat = f.read(row_bytes)
            if len(pat) != row_bytes:
                msg = f"Truncated BRCD at pattern {i}"
                raise ValueError(msg)
            patterns.append(pat)
    return DitherChannel(
        width=width, height=height, threshold_matrix=_extract_threshold_matrix(patterns, width, height)
    )


# BRCD filename templates: f"{prefix}-{ch}{suffix}_cache09.bin"
_BRCD_PREFIX = {False: "0600", True: "capt"}  # keyed by `fine`
_BRCD_SUFFIX = {False: "", True: "-TS"}  # keyed by `toner_save`


def load_brcd_tables(
    lut_dir: str,
    *,
    fine: bool = False,
    toner_save: bool = False,
) -> dict[str, DitherChannel] | None:
    """Try to load all 4 BRCD channel files from `lut_dir`.

    Returns:
        Channel dict keyed by 'K', 'C', 'M', 'Y', or None if any
        expected file is missing.
    """
    prefix = _BRCD_PREFIX[fine]
    suffix = _BRCD_SUFFIX[toner_save]
    lut_path = Path(lut_dir)
    channels: dict[str, DitherChannel] = {}
    for ch in "KCMY":
        fpath = lut_path / f"{prefix}-{ch.lower()}{suffix}_cache09.bin"
        if not fpath.exists():
            logger.info("BRCD file not found: %s, falling back to Bayer", fpath)
            return None
        channels[ch] = dither_load_brcd(str(fpath))
    return channels


def _build_bayer_tables(*, toner_save: bool, width: int = 32, height: int = 32) -> dict[str, DitherChannel]:
    """Generate a square Bayer matrix and wrap it as four shared channels.

    With `toner_save=True`, the threshold values are shifted toward 254 by
    ~40 % so a higher ink level is needed before a dot is placed.

    Returns:
        Channel dict keyed by 'K', 'C', 'M', 'Y' (all share one matrix).

    Raises:
        ValueError: If `width` is not a power of 2 or `width != height`.
    """
    n = width.bit_length() - 1
    if (1 << n) != width or width != height:
        msg = f"Bayer matrix requires square power-of-2 dimensions, got {width}x{height}"
        raise ValueError(msg)

    matrix = _bayer_matrix(n)
    normalized = _normalize_matrix(matrix, width, height)
    if toner_save:
        normalized = [[min(254, v + (254 - v) * 2 // 5) for v in row] for row in normalized]

    channel = DitherChannel(width=width, height=height, threshold_matrix=np.array(normalized, dtype=np.uint8))
    return {"K": channel, "C": channel, "M": channel, "Y": channel}


def load_dither_tables(
    lut_dir: str | None = None,
    *,
    fine: bool = False,
    toner_save: bool = False,
) -> dict[str, DitherChannel]:
    """Return the dither channels for the requested mode.

    If `lut_dir` is given, BRCD files matching `(fine, toner_save)` are
    tried first. Fine-mode lookup falls through to Normal-mode BRCD if
    the Fine tables are absent. When no BRCD set is found a Bayer matrix
    is generated as fallback.

    Returns:
        Channel dict keyed by 'K', 'C', 'M', 'Y'.
    """
    if lut_dir is not None:
        if fine:
            tables = load_brcd_tables(lut_dir, fine=True, toner_save=toner_save)
            if tables is not None:
                return tables
            logger.info("Fine BRCD tables not available, falling through to Normal")
        tables = load_brcd_tables(lut_dir, fine=False, toner_save=toner_save)
        if tables is not None:
            return tables
        logger.info("BRCD tables not available, using Bayer fallback")
    return _build_bayer_tables(toner_save=toner_save)


@functools.cache
def _ensure_defaults() -> dict[str, DitherChannel]:
    """Return the default (Bayer) dither channels, built once.

    Returns:
        Channel dict keyed by 'K', 'C', 'M', 'Y'.
    """
    return _build_bayer_tables(toner_save=False)


def dither_channel_1bpp_arr(
    row_arr: npt.NDArray[np.uint8], y: int, width: int, channel: DitherChannel | None = None
) -> bytes:
    """Dither a single-channel pixel row to a 1bpp packed bitmap.

    Args:
        row_arr: at least `width` pixel values (0=black/full ink, 255=white/no ink);
            extra pixels are ignored
        y: scanline index (selects the threshold row)
        width: number of pixels
        channel: DitherChannel (uses the default K channel if None)

    Returns:
        ceil(width/8) bytes of packed 1bpp bitmap (MSB-first).
    """
    if channel is None:
        channel = _ensure_defaults()["K"]

    thresholds = channel.tiled_thresholds(width)[y % channel.height]
    if HAS_CYTHON_DITHER:
        return dither_row_1bpp(row_arr, thresholds, width)
    dots = (255 - row_arr[:width]) > thresholds
    return bytes(np.packbits(dots)[: (width + 7) // 8])


def dither_channel_1bpp(row: bytes, y: int, width: int, channel: DitherChannel | None = None) -> bytes:
    """Bytes form of :func:`dither_channel_1bpp_arr`.

    Returns:
        ceil(width/8) bytes of packed 1bpp bitmap (MSB-first).
    """
    return dither_channel_1bpp_arr(np.frombuffer(row, dtype=np.uint8, count=width), y, width, channel)


def _nibble_pack(levels: npt.NDArray[np.uint8], width: int) -> bytes:
    """Pack array of 4-bit level values into nibble-packed bytes.

    High nibble = first pixel, low nibble = second pixel.
    Pads with zero nibble if width is odd.

    Returns:
        ceil(width/2) bytes of packed nibbles.
    """
    if width % 2 == 1:
        levels = np.append(levels, np.uint8(0))
    return bytes((levels[0::2].astype(np.uint8) << 4) | levels[1::2].astype(np.uint8))


def dither_channel_4bpp_arr(
    row_arr: npt.NDArray[np.uint8], y: int, width: int, channel: DitherChannel | None = None
) -> bytes:
    """Dither a single-channel pixel row to 4bpp nibble-packed output.

    Args:
        row_arr: at least `width` pixel values (0=black/full ink, 255=white/no ink);
            extra pixels are ignored
        y: scanline index (selects the threshold row)
        width: number of pixels
        channel: DitherChannel (uses the default K channel if None)

    Returns:
        (width + 1) // 2 bytes of nibble-packed output (high nibble = first pixel).
        Each nibble is 0-15 (0=no ink, 15=full ink).
    """
    if channel is None:
        channel = _ensure_defaults()["K"]

    ink = (255 - row_arr[:width]).astype(np.int32)
    thresholds = channel.tiled_thresholds(width)[y % channel.height].astype(np.int32)

    base = (ink * 15) // 255
    frac = (ink * 15) % 255
    levels = np.minimum(15, base + (frac > thresholds).astype(np.int32))

    return _nibble_pack(levels.astype(np.uint8), width)


def dither_channel_4bpp(row: bytes, y: int, width: int, channel: DitherChannel | None = None) -> bytes:
    """Bytes form of :func:`dither_channel_4bpp_arr`.

    Returns:
        (width + 1) // 2 bytes of nibble-packed output.
    """
    return dither_channel_4bpp_arr(np.frombuffer(row, dtype=np.uint8, count=width), y, width, channel)

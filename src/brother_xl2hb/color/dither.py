"""Ordered dithering for the Brother HL-4150CDN.

Converts continuous-tone intensity values to 1bpp or 4bpp planes.
Loads BRCD cache files (per-channel threshold tables) when available
and falls back to a standard Bayer matrix otherwise.

1bpp mode: binary on/off dots, used in Normal mode. The ink amount is
compared against a threshold matrix tiled to the page width, vectorised
with numpy (or `_dither_fast`).

4bpp mode: 16 intensity levels (0-15) per pixel, nibble-packed output,
used in Fine mode. The original (`dither_ordered_4ch_4bpp` @ 08061404)
reads the same BRCD data as one byte per pixel: level `ink` is a
matrix of ready-made nibbles, indexed by page row and column. Without
BRCD tables, a Bayer threshold matrix interpolates between levels.
"""

import functools
import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from brother_xl2hb import _native

_dither_fast = _native.load("_dither_fast")
_fine_fast = _native.load("_fine_fast")

logger = logging.getLogger(__name__)


@dataclass
class DitherChannel:
    """Ordered-dither data for one color channel: a threshold matrix or per-level nibbles."""

    width: int  # matrix width in pixels (e.g. 32)
    height: int  # matrix height in pixels (e.g. 32)
    # shape (height, width), dtype uint8 — threshold[y][x]: dot if ink > threshold.
    # None for Fine BRCD channels, which have `nibble_levels` instead.
    threshold_matrix: npt.NDArray[np.uint8] | None = field(repr=False)
    # shape (256, height, width), dtype uint8 — nibble_levels[ink][y][x] is the 4bpp
    # output for that ink amount (Fine BRCD tables only).
    nibble_levels: npt.NDArray[np.uint8] | None = field(default=None, repr=False)

    # Threshold matrix tiled to the target page width, keyed by width.
    _tiled_cache: dict[int, npt.NDArray[np.uint8]] = field(default_factory=dict, init=False, repr=False)
    # Column of the nibble matrix for every pixel of a row, keyed by row width.
    _column_cache: dict[int, npt.NDArray[np.intp]] = field(default_factory=dict, init=False, repr=False)

    def matrix_columns(self, width: int) -> npt.NDArray[np.intp]:
        """Return `x % self.width` for x in range(width), cached per width.

        Returns:
            Array of `width` column indices into the matrix.
        """
        cached = self._column_cache.get(width)
        if cached is None:
            cached = np.arange(width) % self.width
            self._column_cache[width] = cached
        return cached

    def tiled_thresholds(self, width: int) -> npt.NDArray[np.uint8]:
        """Return the threshold matrix tiled to `width`, cached per width.

        Returns:
            Array of shape (height, width) with the repeated threshold matrix.

        Raises:
            ValueError: For a channel without a threshold matrix.
        """
        cached = self._tiled_cache.get(width)
        if cached is not None:
            return cached
        if self.threshold_matrix is None:
            msg = "channel has no threshold matrix (Fine BRCD tables are per-level nibbles)"
            raise ValueError(msg)
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


def dither_load_brcd(path: str, *, nibbles: bool = False) -> DitherChannel:
    """Load dither table from a BRCD cache file.

    File format: 4-byte magic "BRCD", 1-byte version ('0' for the Normal
    tables, '1' for Fine), 1-byte reserved, 2-byte LE width in bits,
    2-byte LE height, then 256 x row_bytes of pattern data, one block per
    ink level.

    With `nibbles`, each block is read as the 4bpp dither does: a
    (height, width / 8) matrix of nibble values, one byte per pixel.
    Otherwise it is a 1bpp dot pattern, reduced to a threshold matrix.

    Returns:
        DitherChannel with the threshold matrix or, with `nibbles`, the per-level nibbles.

    Raises:
        ValueError: If the magic bytes are wrong, pattern data is truncated, or a
            Fine table's width is not a whole number of bytes.
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
    if nibbles:
        if width % 8:
            msg = f"Fine BRCD width must be a multiple of 8 bits, got {width} in {path}"
            raise ValueError(msg)
        cols = width >> 3
        levels = np.frombuffer(b"".join(patterns), dtype=np.uint8).reshape(256, row_bytes)
        levels = levels[:, : height * cols].reshape(256, height, cols).copy()
        return DitherChannel(width=cols, height=height, threshold_matrix=None, nibble_levels=levels)
    return DitherChannel(
        width=width, height=height, threshold_matrix=_extract_threshold_matrix(patterns, width, height)
    )


# BRCD filename templates: f"{prefix}-{ch}{suffix}_{cache}.bin", `cache` per model (e.g. cache09)
_BRCD_PREFIX = {False: "0600", True: "capt"}  # keyed by `fine`
_BRCD_SUFFIX = {False: "", True: "-TS"}  # keyed by `toner_save`


def load_brcd_tables(
    lut_dir: str,
    *,
    cache: str,
    fine: bool = False,
    toner_save: bool = False,
) -> dict[str, DitherChannel] | None:
    """Try to load all 4 BRCD channel files of the `cache` set from `lut_dir`.

    Returns:
        Channel dict keyed by 'K', 'C', 'M', 'Y', or None if any
        expected file is missing.
    """
    prefix = _BRCD_PREFIX[fine]
    suffix = _BRCD_SUFFIX[toner_save]
    lut_path = Path(lut_dir)
    channels: dict[str, DitherChannel] = {}
    for ch in "KCMY":
        fpath = lut_path / f"{prefix}-{ch.lower()}{suffix}_{cache}.bin"
        if not fpath.exists():
            logger.debug("BRCD file not found: %s", fpath)
            return None
        channels[ch] = dither_load_brcd(str(fpath), nibbles=fine)
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


_fallback_warned: set[tuple[str, str, bool, bool]] = set()


def _brcd_pattern(cache: str, *, fine: bool, toner_save: bool) -> str:
    """File name pattern of a BRCD set, e.g. `capt-*-TS_cache09.bin`.

    Returns:
        The pattern, `*` standing for the channel letter.
    """
    return f"{_BRCD_PREFIX[fine]}-*{_BRCD_SUFFIX[toner_save]}_{cache}.bin"


def _warn_fallback(lut_dir: str, *, cache: str, fine: bool, toner_save: bool, stand_in: str) -> None:
    """Warn, once per process and table set, that the BRCD set is missing and `stand_in` replaces it.

    Every stand-in changes the output against the original driver's.
    """
    key = (lut_dir, cache, fine, toner_save)
    if key in _fallback_warned:
        return
    _fallback_warned.add(key)
    logger.warning(
        "Dither tables %s missing in %s (run scripts/extract_blobs.sh %s); "
        "using %s, output will differ from the original driver",
        _brcd_pattern(cache, fine=fine, toner_save=toner_save),
        lut_dir,
        Path(lut_dir).name,
        stand_in,
    )


def load_dither_tables(
    lut_dir: str | None = None,
    *,
    cache: str,
    fine: bool = False,
    toner_save: bool = False,
) -> dict[str, DitherChannel]:
    """Return the dither channels for the requested mode.

    If `lut_dir` is given, the `cache` set's BRCD files matching `(fine, toner_save)` are
    tried first. Fine-mode lookup falls through to Normal-mode BRCD if
    the Fine tables are absent. When no BRCD set is found a Bayer matrix
    is generated as fallback. Either fallback changes the output and
    warns once per table set.

    Returns:
        Channel dict keyed by 'K', 'C', 'M', 'Y'.
    """
    if lut_dir is not None:
        if fine:
            tables = load_brcd_tables(lut_dir, cache=cache, fine=True, toner_save=toner_save)
            if tables is not None:
                return tables
        tables = load_brcd_tables(lut_dir, cache=cache, fine=False, toner_save=toner_save)
        if tables is not None:
            if fine:
                normal = _brcd_pattern(cache, fine=False, toner_save=toner_save)
                _warn_fallback(
                    lut_dir, cache=cache, fine=True, toner_save=toner_save, stand_in=f"the Normal-mode tables {normal}"
                )
            return tables
        _warn_fallback(lut_dir, cache=cache, fine=fine, toner_save=toner_save, stand_in="a generated Bayer matrix")
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
    if _dither_fast is not None:
        return _dither_fast.dither_row_1bpp(row_arr, thresholds, width)
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

    if channel.nibble_levels is not None:
        # Plain table lookup, as dither_ordered_4ch_4bpp: levels[ink][y % h][x % w].
        if _fine_fast is not None:
            return _fine_fast.dither_row_4bpp(row_arr, channel.nibble_levels, y, width)
        levels = channel.nibble_levels[255 - row_arr[:width], y % channel.height, channel.matrix_columns(width)]
        return _nibble_pack(levels, width)

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

"""Ordered dithering for the Brother HL-4150CDN.

Converts continuous-tone intensity values to 1bpp or 4bpp planes.
Loads BRCD cache files (per-channel threshold tables) when available
and falls back to a standard Bayer matrix otherwise.

1bpp mode: binary on/off dots — used in Normal mode.
4bpp mode: 16 intensity levels (0-15) per pixel, nibble-packed output.

The numpy fast-path vectorizes the threshold comparison and uses
np.packbits() / nibble packing instead of per-pixel pattern lookups.
"""

import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


@dataclass
class DitherChannel:
    """Pre-computed dither patterns for one color channel."""

    width: int  # matrix width in pixels (e.g. 32)
    height: int  # matrix height in pixels (e.g. 32)
    row_bytes: int  # (width * height + 7) // 8
    patterns: list[bytes] = field(repr=False)  # 256 entries indexed by ink amount
    # patterns[0] = no ink = no dots
    # patterns[255] = full ink = all dots
    threshold_matrix: npt.NDArray[np.uint8] | None = field(default=None, repr=False)
    # shape (height, width), dtype uint8 — threshold[y][x]: dot if ink > threshold

    # Threshold matrix tiled to the target page width, keyed by width.
    _tiled_cache: dict[int, npt.NDArray[np.uint8]] = field(default_factory=dict, init=False, repr=False)


# Module-level cache for default channels (lazily initialized)
_cache: dict[str, dict[str, DitherChannel]] = {}


def _get_tiled_thresholds(channel: DitherChannel, width: int) -> npt.NDArray[np.uint8]:
    """Return the channel's threshold matrix tiled to `width`, cached on the channel."""
    cached = channel._tiled_cache.get(width)
    if cached is not None:
        return cached
    tm = channel.threshold_matrix
    assert tm is not None
    repeats = (width + channel.width - 1) // channel.width
    tiled = np.tile(tm, (1, repeats))[:, :width].copy()  # (height, width)
    channel._tiled_cache[width] = tiled
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


def _build_patterns(normalized: list[list[int]], width: int, height: int) -> list[bytes]:
    """Build 256-entry pattern LUT from normalized threshold matrix.

    Indexed by ink amount: patterns[0] = no dots, patterns[255] = all dots.
    Comparison: dot if ink > threshold (both in [0, 255] / [0, 254]).

    Returns:
        256-element list of packed-bit pattern bytes (one per ink level).
    """
    total = width * height
    row_bytes = (total + 7) // 8

    # Vectorized: compare all 256 ink levels against all thresholds at once
    norm_flat = np.array(normalized, dtype=np.uint16).ravel()  # (total,)
    ink_levels = np.arange(256, dtype=np.uint16).reshape(256, 1)
    # dots[ink, pos] = True if ink > threshold
    dots = ink_levels > norm_flat.reshape(1, -1)  # (256, total)
    # Pack bits (MSB first, matching 0x80 >> (pos & 7) convention)
    packed = np.packbits(dots, axis=1)  # (256, ceil(total/8))
    return [bytes(packed[ink, :row_bytes]) for ink in range(256)]


def _extract_threshold_matrix(channel: DitherChannel) -> npt.NDArray[np.uint8]:
    """Extract threshold matrix from pre-computed patterns via vectorized search.

    For each position (y, x) in the dither matrix, finds the largest ink
    value where the bit is NOT set. Since patterns are monotonic (once a bit
    appears at ink level k, it stays for all higher levels), we find the
    first ink where each bit turns on.

    Returns:
        Threshold matrix of shape (height, width), uint8.
    """
    total = channel.height * channel.width

    # Stack all 256 patterns and unpack to bits: (256, total)
    pat_arr = np.frombuffer(b"".join(channel.patterns), dtype=np.uint8).reshape(256, -1)
    bits = np.unpackbits(pat_arr, axis=1)[:, :total]  # (256, total)

    # For each position, find first ink level where bit is set
    # argmax on bool array returns index of first True
    ever_set = np.any(bits, axis=0)  # (total,)
    first_set = np.argmax(bits, axis=0)  # (total,)

    # threshold = first_set - 1 where bit is ever set, else 255
    thresholds = np.where(ever_set, first_set - 1, 255).astype(np.uint8)
    return thresholds.reshape(channel.height, channel.width)


def dither_load_brcd(path: str) -> DitherChannel:
    """Load dither table from a BRCD cache file.

    File format: 4-byte magic "BRCD", 1-byte version ('0'),
    1-byte reserved, 2-byte LE width, 2-byte LE height,
    then 256 x row_bytes of pattern data.

    Returns:
        DitherChannel populated from the file plus its derived threshold matrix.

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
    channel = DitherChannel(width=width, height=height, row_bytes=row_bytes, patterns=patterns)
    channel.threshold_matrix = _extract_threshold_matrix(channel)
    return channel


# BRCD filename templates: f"{prefix}-{ch}{suffix}_cache09.bin"
_BRCD_PREFIX = {False: "0600", True: "capt"}  # keyed by `fine`
_BRCD_SUFFIX = {False: "", True: "-TS"}  # keyed by `toner_save`


def _try_load_brcd(
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

    patterns = _build_patterns(normalized, width, height)
    row_bytes = (width * height + 7) // 8
    threshold_matrix = np.array(normalized, dtype=np.uint8)

    channel = DitherChannel(
        width=width,
        height=height,
        row_bytes=row_bytes,
        patterns=patterns,
        threshold_matrix=threshold_matrix,
    )
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
            tables = _try_load_brcd(lut_dir, fine=True, toner_save=toner_save)
            if tables is not None:
                return tables
            logger.info("Fine BRCD tables not available, falling through to Normal")
        tables = _try_load_brcd(lut_dir, fine=False, toner_save=toner_save)
        if tables is not None:
            return tables
        logger.info("BRCD tables not available, using Bayer fallback")
    return _build_bayer_tables(toner_save=toner_save)


def _ensure_defaults() -> dict[str, DitherChannel]:
    """Lazily initialize and return default (Bayer) dither channels.

    Returns:
        Cached channel dict keyed by 'K', 'C', 'M', 'Y'.
    """
    if "default" not in _cache:
        _cache["default"] = _build_bayer_tables(toner_save=False)
    return _cache["default"]


def dither_channel_1bpp(row: bytes, y: int, width: int, channel: DitherChannel | None = None) -> bytes:
    """Dither single-channel pixel row to 1bpp packed bitmap.

    Args:
        row: width bytes of pixel values (0=black/full ink, 255=white/no ink)
        y: scanline index (for pattern row selection)
        width: number of pixels
        channel: DitherChannel with LUT (uses default K if None)

    Returns:
        ceil(width/8) bytes of packed 1bpp bitmap (MSB-first)
    """
    if channel is None:
        channel = _ensure_defaults()["K"]

    bpl = (width + 7) // 8

    if channel.threshold_matrix is not None:
        # Numpy fast-path: vectorized threshold comparison
        ink = 255 - np.frombuffer(row, dtype=np.uint8, count=width)
        thresholds = _get_tiled_thresholds(channel, width)[y % channel.height]
        dots = ink > thresholds
        packed = np.packbits(dots)
        return bytes(packed[:bpl])

    # Pure-Python fallback
    out = bytearray(bpl)

    mat_w = channel.width
    mat_h = channel.height
    row_in_pattern = y % mat_h

    for x in range(width):
        pixel = row[x]
        if pixel == 255:
            continue  # white -> no dot

        # Convert pixel brightness to ink amount
        ink = 255 - pixel
        pattern = channel.patterns[ink]

        # Look up bit in pattern
        bit_offset = row_in_pattern * mat_w + (x % mat_w)
        if pattern[bit_offset >> 3] & (0x80 >> (bit_offset & 7)):
            out[x >> 3] |= 0x80 >> (x & 7)

    return bytes(out)


def dither_cmyk_1bpp(
    cmyk_row: bytes,
    y: int,
    width: int,
    channels: dict[str, DitherChannel] | None = None,
) -> tuple[bytes, bytes, bytes, bytes]:
    """Dither 4-channel CMYK intensity row to four 1bpp planes.

    Args:
        cmyk_row: width*4 bytes of interleaved C,M,Y,K ink values
                  (0=no ink, 255=full ink)
        y: scanline index
        width: number of pixels
        channels: dict of DitherChannels (uses defaults if None)

    Returns:
        (k_data, c_data, m_data, y_data) each ceil(width/8) bytes
    """
    if channels is None:
        channels = _ensure_defaults()

    bpl = (width + 7) // 8

    # Check if all channels have threshold matrices for numpy fast-path
    k_ch = channels["K"]
    c_ch = channels["C"]
    m_ch = channels["M"]
    y_ch = channels["Y"]

    # Pull threshold matrices out as locals so the chained `is not None` narrows each one;
    # the loop body then sees `tm` as a non-Optional NDArray with no further checks needed.
    k_tm = k_ch.threshold_matrix
    c_tm = c_ch.threshold_matrix
    m_tm = m_ch.threshold_matrix
    y_tm = y_ch.threshold_matrix
    if k_tm is not None and c_tm is not None and m_tm is not None and y_tm is not None:
        # Numpy fast-path
        cmyk = np.frombuffer(cmyk_row, dtype=np.uint8, count=width * 4).reshape(width, 4)
        c_ink = cmyk[:, 0]
        m_ink = cmyk[:, 1]
        y_ink = cmyk[:, 2]
        k_ink = cmyk[:, 3]

        results = []
        for ink_arr, ch in ((k_ink, k_ch), (c_ink, c_ch), (m_ink, m_ch), (y_ink, y_ch)):
            thresholds = _get_tiled_thresholds(ch, width)[y % ch.height]
            dots = ink_arr > thresholds
            packed = np.packbits(dots)
            results.append(bytes(packed[:bpl]))

        return results[0], results[1], results[2], results[3]

    # Pure-Python fallback
    k_out = bytearray(bpl)
    c_out = bytearray(bpl)
    m_out = bytearray(bpl)
    y_out = bytearray(bpl)

    for x in range(width):
        c_ink_val = cmyk_row[x * 4]
        m_ink_val = cmyk_row[x * 4 + 1]
        y_ink_val = cmyk_row[x * 4 + 2]
        k_ink_val = cmyk_row[x * 4 + 3]

        out_byte = x >> 3
        out_mask = 0x80 >> (x & 7)

        if k_ink_val > 0:
            pat = k_ch.patterns[k_ink_val]
            off = (y % k_ch.height) * k_ch.width + (x % k_ch.width)
            if pat[off >> 3] & (0x80 >> (off & 7)):
                k_out[out_byte] |= out_mask

        if c_ink_val > 0:
            pat = c_ch.patterns[c_ink_val]
            off = (y % c_ch.height) * c_ch.width + (x % c_ch.width)
            if pat[off >> 3] & (0x80 >> (off & 7)):
                c_out[out_byte] |= out_mask

        if m_ink_val > 0:
            pat = m_ch.patterns[m_ink_val]
            off = (y % m_ch.height) * m_ch.width + (x % m_ch.width)
            if pat[off >> 3] & (0x80 >> (off & 7)):
                m_out[out_byte] |= out_mask

        if y_ink_val > 0:
            pat = y_ch.patterns[y_ink_val]
            off = (y % y_ch.height) * y_ch.width + (x % y_ch.width)
            if pat[off >> 3] & (0x80 >> (off & 7)):
                y_out[out_byte] |= out_mask

    return bytes(k_out), bytes(c_out), bytes(m_out), bytes(y_out)


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


def dither_channel_4bpp(row: bytes, y: int, width: int, channel: DitherChannel | None = None) -> bytes:
    """Dither single-channel pixel row to 4bpp nibble-packed output.

    Args:
        row: width bytes of pixel values (0=black/full ink, 255=white/no ink)
        y: scanline index (for pattern row selection)
        width: number of pixels
        channel: DitherChannel with LUT (uses default K if None)

    Returns:
        (width + 1) // 2 bytes of nibble-packed output (high nibble = first pixel).
        Each nibble is 0-15 (0=no ink, 15=full ink).
    """
    if channel is None:
        channel = _ensure_defaults()["K"]

    out_len = (width + 1) // 2

    if channel.threshold_matrix is not None:
        # Numpy fast-path: multi-level ordered dithering
        pixels = np.frombuffer(row, dtype=np.uint8, count=width)
        ink = (255 - pixels).astype(np.int32)

        thresholds = _get_tiled_thresholds(channel, width)[y % channel.height].astype(np.int32)

        base = (ink * 15) // 255
        frac = (ink * 15) % 255
        levels = np.minimum(15, base + (frac > thresholds).astype(np.int32))

        return _nibble_pack(levels.astype(np.uint8), width)

    # Pure-Python fallback: pattern-based (matching driver byte reinterpretation)
    out = bytearray(out_len)
    stride = channel.width // 8
    mat_h = channel.height

    for x in range(width):
        pixel = row[x]
        if pixel == 255:
            continue  # white -> no ink, nibble stays 0

        ink = 255 - pixel
        pat = channel.patterns[ink]
        value = pat[(y % mat_h) * stride + (x % stride)] & 0x0F

        byte_idx = x // 2
        if x % 2 == 0:
            out[byte_idx] = value << 4
        else:
            out[byte_idx] |= value

    return bytes(out)


def dither_cmyk_4bpp(
    cmyk_row: bytes,
    y: int,
    width: int,
    channels: dict[str, DitherChannel] | None = None,
) -> tuple[bytes, bytes, bytes, bytes]:
    """Dither 4-channel CMYK intensity row to four 4bpp nibble-packed planes.

    Args:
        cmyk_row: width*4 bytes of interleaved C,M,Y,K ink values
                  (0=no ink, 255=full ink)
        y: scanline index
        width: number of pixels
        channels: dict of DitherChannels (uses defaults if None)

    Returns:
        (k_data, c_data, m_data, y_data) each (width + 1) // 2 bytes
        of nibble-packed output (high nibble = first pixel).
    """
    if channels is None:
        channels = _ensure_defaults()

    out_len = (width + 1) // 2

    k_ch = channels["K"]
    c_ch = channels["C"]
    m_ch = channels["M"]
    y_ch = channels["Y"]

    # Pull threshold matrices out as locals so the chained `is not None` narrows each one;
    # the loop body then sees `tm` as a non-Optional NDArray with no further checks needed.
    k_tm = k_ch.threshold_matrix
    c_tm = c_ch.threshold_matrix
    m_tm = m_ch.threshold_matrix
    y_tm = y_ch.threshold_matrix
    if k_tm is not None and c_tm is not None and m_tm is not None and y_tm is not None:
        # Numpy fast-path
        cmyk = np.frombuffer(cmyk_row, dtype=np.uint8, count=width * 4).reshape(width, 4)
        c_ink = cmyk[:, 0].astype(np.int32)
        m_ink = cmyk[:, 1].astype(np.int32)
        y_ink = cmyk[:, 2].astype(np.int32)
        k_ink = cmyk[:, 3].astype(np.int32)

        results = []
        for ink_arr, ch in ((k_ink, k_ch), (c_ink, c_ch), (m_ink, m_ch), (y_ink, y_ch)):
            thresholds = _get_tiled_thresholds(ch, width)[y % ch.height].astype(np.int32)

            base = (ink_arr * 15) // 255
            frac = (ink_arr * 15) % 255
            levels = np.minimum(15, base + (frac > thresholds).astype(np.int32))
            results.append(_nibble_pack(levels.astype(np.uint8), width))

        return results[0], results[1], results[2], results[3]

    # Pure-Python fallback
    k_out = bytearray(out_len)
    c_out = bytearray(out_len)
    m_out = bytearray(out_len)
    y_out = bytearray(out_len)

    for x in range(width):
        c_ink_val = cmyk_row[x * 4]
        m_ink_val = cmyk_row[x * 4 + 1]
        y_ink_val = cmyk_row[x * 4 + 2]
        k_ink_val = cmyk_row[x * 4 + 3]

        byte_idx = x // 2
        is_high = x % 2 == 0

        for ink_val, ch, out in (
            (k_ink_val, k_ch, k_out),
            (c_ink_val, c_ch, c_out),
            (m_ink_val, m_ch, m_out),
            (y_ink_val, y_ch, y_out),
        ):
            if ink_val == 0:
                continue
            stride = ch.width // 8
            value = ch.patterns[ink_val][(y % ch.height) * stride + (x % stride)] & 0x0F
            if is_high:
                out[byte_idx] = value << 4
            else:
                out[byte_idx] |= value

    return bytes(k_out), bytes(c_out), bytes(m_out), bytes(y_out)

"""Saturation adjustment for RGB pixel data.

Supports both positive saturation (boost) and negative (desaturation),
with min/mid/max channel sorting, hue-preserving mid-channel
interpolation, and gray/black/white pixel skipping.
"""

import functools
import logging
from collections.abc import Buffer

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# Divisor that scales the [-20, +20] saturation knob into per-channel deltas.
_SCALE_FACTOR = 50.0


def adjust_saturation(rgb_row: Buffer, width: int, sat_mode: int) -> bytes:
    """Adjust saturation of an RGB pixel row.

    Args:
        rgb_row: Raw RGB bytes (width * 3 bytes).
        width: Number of pixels.
        sat_mode: Saturation adjustment (-20..+20).
                  Positive = boost saturation, negative = desaturate.

    Returns:
        Adjusted RGB bytes (same length as input).

    Raises:
        ValueError: If `rgb_row` is shorter than `width * 3` bytes.
    """
    size = memoryview(rgb_row).nbytes
    if size < width * 3:
        msg = f"rgb_row too short: {size} < {width * 3}"
        raise ValueError(msg)
    if sat_mode == 0:
        return bytes(rgb_row)
    rgb = np.frombuffer(rgb_row, dtype=np.uint8, count=width * 3).reshape(width, 3).copy()
    _adjust_saturation_inplace(rgb, sat_mode)
    return rgb.tobytes()


def _adjust_saturation_inplace(rgb: npt.NDArray[np.uint8], sat_mode: int) -> None:
    """Adjust saturation in-place on a (N, 3) uint8 array."""
    r = rgb[:, 0].astype(np.int32)
    g = rgb[:, 1].astype(np.int32)
    b = rgb[:, 2].astype(np.int32)

    # Base condition: pixel must not be gray (at least one channel differs from green)
    not_gray = (r != g) | (b != g)

    if sat_mode > 0:
        # Positive saturation: also skip pixels with ANY channel at 0 or 255
        all_nonzero = (r != 0) & (g != 0) & (b != 0)
        all_non255 = (r != 255) & (g != 255) & (b != 255)
        mask = not_gray & all_nonzero & all_non255
    else:
        mask = not_gray

    if not np.any(mask):
        return

    pixels = rgb[mask].astype(np.int32)  # (M, 3)

    min_vals = pixels.min(axis=1)
    max_vals = pixels.max(axis=1)

    if sat_mode > 0:
        _boost_saturation(pixels, min_vals, max_vals, sat_mode)
    else:
        _reduce_saturation(pixels, min_vals, max_vals, sat_mode)

    rgb[mask] = np.clip(pixels, 0, 255).astype(np.uint8)


def _round_extended(num: int, den: int) -> tuple[int, int]:
    """Round num/den > 0 to nearest-even with a 64-bit significand (x87 extended precision).

    Returns:
        (m, s) with the rounded value m / 2**s and 2**63 <= m < 2**64.
    """
    s = 64 - (num.bit_length() - den.bit_length())
    while (num << s) // den >= 1 << 64:
        s -= 1
    while (num << s) // den < 1 << 63:
        s += 1
    m, rem = divmod(num << s, den)
    if 2 * rem > den or (2 * rem == den and m & 1):
        m += 1
        if m == 1 << 64:
            m >>= 1
            s -= 1
    return m, s


@functools.cache
def _x87_quotient_product(a: int, b: int, c: int) -> int:
    """trunc((a / b) * c) as the binary computes it on the x87 FPU.

    Returns:
        The truncated result.
    """
    q, s = _round_extended(a, b)
    p, t = _round_extended(q * c, 1 << s)
    return p >> t if t >= 0 else p << -t


def _x87_trunc_div_mul(
    a: npt.NDArray[np.int32], b: npt.NDArray[np.int32], c: npt.NDArray[np.int32]
) -> npt.NDArray[np.int32]:
    """Vectorised trunc((a / b) * c) for a, c >= 0 and b > 0, rounded like the binary.

    `compress_adjust_saturation` divides and multiplies in 80-bit extended
    precision, then truncates (FPU control word 0xC before fistp at
    0x0804f852). That equals floor(a * c / b) except when a * c / b is an
    integer and the rounded quotient lands just below it; only those
    entries go through the exact emulation.

    Returns:
        int32 array of the truncated results.
    """
    num = a * c
    result = num // b
    exact = (num % b == 0) & (num != 0)
    if np.any(exact):
        idx = np.flatnonzero(exact)
        result[idx] = [
            _x87_quotient_product(int(x), int(y), int(z)) for x, y, z in zip(a[idx], b[idx], c[idx], strict=True)
        ]
    return result


def _boost_saturation(
    pixels: npt.NDArray[np.int32],
    min_vals: npt.NDArray[np.int32],
    max_vals: npt.NDArray[np.int32],
    sat_mode: int,
) -> None:
    """Positive saturation: expand channel range, preserve hue."""
    n = pixels.shape[0]
    arange = np.arange(n)

    min_idx = pixels.argmin(axis=1)
    max_idx = pixels.argmax(axis=1)
    mid_idx = 3 - min_idx - max_idx

    mid_vals = pixels[arange, mid_idx]

    half_range = (max_vals - min_vals) // 2
    # boost = (half_range * sat_mode) / 50, integer division (truncate toward zero)
    boost = (half_range * sat_mode) // 50

    # Clamp boost so max + boost <= 255 and min - boost >= 0
    boost = np.minimum(boost, 255 - max_vals)
    boost = np.minimum(boost, min_vals)

    new_max = max_vals + boost
    new_min = min_vals - boost

    # Mid channel: proportional interpolation preserving hue ratio
    old_range = max_vals - min_vals  # always > 0 (non-gray pixels)
    new_range = new_max - new_min

    safe_range = np.maximum(old_range, 1)
    mid_offset = _x87_trunc_div_mul(new_range, safe_range, mid_vals - min_vals)
    new_mid = new_min + mid_offset

    pixels[arange, max_idx] = new_max
    pixels[arange, min_idx] = new_min
    pixels[arange, mid_idx] = new_mid


def _reduce_saturation(
    pixels: npt.NDArray[np.int32],
    min_vals: npt.NDArray[np.int32],
    max_vals: npt.NDArray[np.int32],
    sat_mode: int,
) -> None:
    """Negative saturation: move all channels toward center."""
    half_range = (max_vals - min_vals) // 2
    center = (min_vals + half_range).astype(np.float64)  # (M,)

    for ch in range(3):
        ch_vals = pixels[:, ch].astype(np.float64)

        # Truncate toward zero (matches the printer's expected rounding mode).
        adjusted = np.trunc(ch_vals + sat_mode * (ch_vals - center) / _SCALE_FACTOR)

        # Clamp: channel must not cross center
        above = ch_vals > center
        below = ch_vals < center
        adjusted = np.where(above, np.maximum(adjusted, center), adjusted)
        adjusted = np.where(below, np.minimum(adjusted, center), adjusted)

        pixels[:, ch] = adjusted.astype(np.int32)

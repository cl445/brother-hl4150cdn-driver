"""Per-pixel and per-row colour transforms applied before the 3D-LUT lookup.

* :func:`apply_vivid` boosts saturation by spreading each channel away from
  the per-pixel grey axis.
* :func:`build_input_remap_lut` / :func:`apply_input_remap_rgb` apply
  brightness, contrast, and per-channel RGB-key shifts as a single 256-entry
  per-channel input remap.
* :func:`rgb_line_to_cmyk_intensities` performs the RGB→CMYK separation
  (3D-LUT lookup, or simple GCR when colour-matching is disabled).
"""

import numpy as np
import numpy.typing as npt

from color_lut import rgb_to_cmyk_lut, rgb_to_cmyk_lut_arr
from settings import ColorMatching

_NDArrayU8 = npt.NDArray[np.uint8]


def apply_vivid(rgb_row: bytes, width: int) -> bytes:
    """Boost saturation by expanding distance from per-pixel gray axis.

    Returns:
        RGB bytes (same length as input) with boosted saturation.
    """
    rgb = np.frombuffer(rgb_row, dtype=np.uint8, count=width * 3).reshape(width, 3).astype(np.int16)
    avg = rgb.sum(axis=1, keepdims=True) // 3
    # Boost factor 1.4 -- shift each channel away from gray
    boosted = avg + (rgb - avg) * 14 // 10
    return np.clip(boosted, 0, 255).astype(np.uint8).tobytes()


def build_input_remap_lut(brightness: int, contrast: int, channel: int) -> npt.NDArray[np.uint8]:
    """Build a 256-entry RGB-input remap LUT for one channel.

    Per source value ``v``, applied left-to-right with clamp to [0,255]
    after each step:

        val = v + trunc(brightness * 255 / 128)
        val = val + trunc((val - 128) * contrast / 100)
        val = val + trunc(channel * 255 / 128)

    Truncation is toward zero.

    Returns:
        256-entry uint8 LUT mapping source value to remapped value.
    """
    bright_shift = int(brightness * 255 / 128)
    chan_shift = int(channel * 255 / 128)
    v = np.arange(256, dtype=np.int32)
    val = np.clip(v + bright_shift, 0, 255)
    contrast_delta = (val - 128) * contrast
    sign = np.sign(contrast_delta)
    contrast_delta = sign * (np.abs(contrast_delta) // 100)
    val = np.clip(val + contrast_delta, 0, 255)
    val = np.clip(val + chan_shift, 0, 255)
    return val.astype(np.uint8)


def apply_input_remap_rgb(
    rgb_row: bytes,
    width: int,
    lut_r: npt.NDArray[np.uint8],
    lut_g: npt.NDArray[np.uint8],
    lut_b: npt.NDArray[np.uint8],
) -> bytes:
    """Apply per-channel input-remap LUT to an RGB scanline.

    Pure-white pixels (255,255,255) are passed through untouched so negative
    brightness/contrast does not deposit ink on the page background.

    Returns:
        Remapped RGB bytes (same length as input).
    """
    rgb = np.frombuffer(rgb_row, dtype=np.uint8, count=width * 3).reshape(width, 3)
    out = np.empty_like(rgb)
    out[:, 0] = lut_r[rgb[:, 0]]
    out[:, 1] = lut_g[rgb[:, 1]]
    out[:, 2] = lut_b[rgb[:, 2]]
    is_white = (rgb[:, 0] == 255) & (rgb[:, 1] == 255) & (rgb[:, 2] == 255)
    out[is_white] = rgb[is_white]
    return out.tobytes()


def rgb_line_to_cmyk_intensities_arr(
    rgb_row: bytes,
    width: int,
    color_matching: ColorMatching = ColorMatching.NORMAL,
) -> tuple[_NDArrayU8, _NDArrayU8, _NDArrayU8, _NDArrayU8]:
    """Like :func:`rgb_line_to_cmyk_intensities` but returns ndarrays directly."""
    if color_matching == ColorMatching.NONE:
        rgb = np.frombuffer(rgb_row, dtype=np.uint8, count=width * 3).reshape(width, 3)
        k = np.full(width, 255, dtype=np.uint8)
        return k, rgb[:, 0].copy(), rgb[:, 1].copy(), rgb[:, 2].copy()
    return rgb_to_cmyk_lut_arr(rgb_row, width)


def rgb_line_to_cmyk_intensities(
    rgb_row: bytes,
    width: int,
    color_matching: ColorMatching = ColorMatching.NORMAL,
) -> tuple[bytes, bytes, bytes, bytes]:
    """Convert one RGB scanline to per-channel CMYK intensity arrays.

    Uses the 3D LUT for colour separation when ``color_matching`` is
    ``NORMAL`` or ``VIVID``, and a simple GCR pass when it is ``NONE``.

    Returns:
        Tuple ``(k_arr, c_arr, m_arr, y_arr)`` each of ``width`` bytes in
        pixel-brightness convention (0 = full ink, 255 = no ink), ready
        for :func:`dither.dither_channel_1bpp`.
    """
    if color_matching == ColorMatching.NONE:
        k, c, m, y = rgb_line_to_cmyk_intensities_arr(rgb_row, width, color_matching)
        return k.tobytes(), c.tobytes(), m.tobytes(), y.tobytes()
    return rgb_to_cmyk_lut(rgb_row, width)

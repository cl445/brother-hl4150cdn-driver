"""Per-pixel and per-row colour transforms applied before the 3D-LUT lookup.

* :func:`build_input_remap_lut` / :func:`apply_input_remap_rgb` apply
  brightness, contrast, and per-channel RGB-key shifts as a single 256-entry
  per-channel input remap.
* :func:`color_table` picks the colour grid the original driver would use
  for the settings; :func:`rgb_line_to_cmyk_intensities` separates RGB into
  CMYK through it. Colour matching, BRGray, toner save, glossy media and
  BREnhanceBlkPrt only choose the grid; they do not touch the pixels.
"""

from collections.abc import Buffer

import numpy as np
import numpy.typing as npt

from color_lut import DEFAULT_TABLE, ColorTable, Profile, Variant, rgb_to_cmyk_lut, rgb_to_cmyk_lut_arr
from settings import ColorMatching, MediaType, PrintSettings

_NDArrayU8 = npt.NDArray[np.uint8]

_PROFILES: dict[ColorMatching, Profile] = {
    ColorMatching.NORMAL: "rgb",
    ColorMatching.VIVID: "srgb",
    ColorMatching.NONE: "cmyk",
}


def color_table(settings: PrintSettings) -> ColorTable:
    """The colour grid `lookup_color_transform_table` selects for `settings`.

    Toner save wins over glossy media, as in the original. Rich black
    (BREnhanceBlkPrt) only exists for the rgb and srgb profiles.

    Returns:
        The table for the colour separation.
    """
    profile = _PROFILES[settings.color_matching]
    variant: Variant = "default"
    if settings.toner_save:
        variant = "density2"
    elif settings.media_type == MediaType.GLOSSY:
        variant = "glossy"
    return ColorTable(
        profile=profile,
        improve_gray=settings.improve_gray,
        variant=variant,
        rich_black=settings.enhance_black and profile != "cmyk",
    )


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
    rgb_row: Buffer,
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
    rgb_row: Buffer,
    width: int,
    table: ColorTable = DEFAULT_TABLE,
) -> tuple[_NDArrayU8, _NDArrayU8, _NDArrayU8, _NDArrayU8]:
    """Like :func:`rgb_line_to_cmyk_intensities` but returns ndarrays directly.

    Returns:
        (k, c, m, y) uint8 intensity arrays of length `width`.
    """
    return rgb_to_cmyk_lut_arr(rgb_row, width, table)


def rgb_line_to_cmyk_intensities(
    rgb_row: Buffer,
    width: int,
    table: ColorTable = DEFAULT_TABLE,
) -> tuple[bytes, bytes, bytes, bytes]:
    """Convert one RGB scanline to per-channel CMYK intensity arrays through `table`.

    Returns:
        Tuple ``(k_arr, c_arr, m_arr, y_arr)`` each of ``width`` bytes in
        pixel-brightness convention (0 = full ink, 255 = no ink), ready
        for :func:`dither.dither_channel_1bpp`.
    """
    return rgb_to_cmyk_lut(rgb_row, width, table)

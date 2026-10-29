"""Per-pixel and per-row colour transforms applied before the 3D-LUT lookup.

* :func:`build_input_remap_lut` / :func:`apply_input_remap_rgb` apply
  brightness, contrast, and per-channel RGB-key shifts as a single 256-entry
  per-channel input remap.
* :func:`color_profile` maps colour matching to the grid profile; the
  model's `color_table` picks the grid within it, and
  :func:`color_lut.rgb_to_cmyk_lut_arr` separates RGB into CMYK through
  it. Colour matching, BRGray, toner save, glossy media and
  BREnhanceBlkPrt only choose the grid; they do not touch the pixels.
"""

from collections.abc import Buffer

import numpy as np
import numpy.typing as npt

from brother_xl2hb.color.color_lut import Profile
from brother_xl2hb.settings import ColorMatching, PrintSettings

_NDArrayU8 = npt.NDArray[np.uint8]

_PROFILES: dict[ColorMatching, Profile] = {
    ColorMatching.NORMAL: "rgb",
    ColorMatching.VIVID: "srgb",
    ColorMatching.NONE: "cmyk",
}


def color_profile(settings: PrintSettings) -> Profile:
    """The colour grid profile for the colour matching setting: rgb, srgb or cmyk.

    Shared by Brother's filters; which variant of the profile's grid applies
    is the model's choice (`PrinterModel.color_table`).

    Returns:
        The profile name.
    """
    return _PROFILES[settings.color_matching]


def build_input_remap_lut(brightness: int, contrast: int, channel: int) -> npt.NDArray[np.uint8]:
    """Build a 256-entry RGB-input remap LUT for one channel.

    Per source value ``v``, applied left-to-right with clamp to [0,255]
    after each step:

        val = v + floor(brightness * 255 / 128)
        val = val + trunc((val - 128) * contrast / 100)
        val = val + trunc(channel * 255 / 128)

    The brightness shift rounds down (the original computes it as
    ``brightness * 255 >> 7``, printer_setup.c:1868), so -20 shifts by -40;
    the other two truncate toward zero.

    Returns:
        256-entry uint8 LUT mapping source value to remapped value.
    """
    bright_shift = brightness * 255 >> 7
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

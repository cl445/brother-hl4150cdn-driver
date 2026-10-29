"""The per-scanline path: colour, dither and plane encoding one line at a time.

Reference for the native band kernels (`bands`), which must match it byte for byte.
"""

from collections.abc import Buffer, Callable, Iterator
from functools import partial
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from brother_xl2hb.codec.fine_encoder import encode_fine_plane
from brother_xl2hb.codec.plane_encoders import encode_c_plane, encode_m_plane_10, encode_plane
from brother_xl2hb.color import color_lut
from brother_xl2hb.color.dither import (
    DitherChannel,
    dither_channel_1bpp_arr,
    dither_channel_4bpp_arr,
)
from brother_xl2hb.color.saturation import adjust_saturation
from brother_xl2hb.color.transforms import apply_input_remap_rgb
from brother_xl2hb.render.rows import RowBlocks, iter_rows, page_rows, white_rows
from brother_xl2hb.settings import PrintSettings

LineCodes = tuple[bytes, bytes, bytes, bytes]
"""Encoded K, C, M, Y data of one scanline; b"" for a plane without ink."""


BLANK_LINE: LineCodes = (b"", b"", b"", b"")


# Per-mode dither dispatcher (intensity ndarray, line_idx, sw, channel) -> packed bytes.
DITHER_FNS = {False: dither_channel_1bpp_arr, True: dither_channel_4bpp_arr}


# Per-mode plane encoders, keyed by plane id; partial() pins the K/Y label.
_PlaneEncoder = Callable[[bytes], bytes]


PLANE_ENCODERS: dict[bool, dict[int, _PlaneEncoder]] = {
    False: {  # Normal mode: 12-bit K/Y RLE, 20-bit C, 10-bit M
        0: partial(encode_plane, plane="K"),
        1: encode_c_plane,
        2: encode_m_plane_10,
        3: partial(encode_plane, plane="Y"),
    },
    True: dict.fromkeys(range(4), encode_fine_plane),  # Fine mode: same encoder for all planes
}


class ColourSetup(NamedTuple):
    """Per-page colour settings shared by the scanline encoders."""

    settings: PrintSettings
    table: color_lut.ColorTable
    input_remap: tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.uint8]] | None

    @property
    def saturation(self) -> int:
        """Saturation to apply; the cmyk profile skips it (`cmyk_basic` in the original)."""
        return 0 if self.table.profile == "cmyk" else self.settings.saturation

    @property
    def adjusts_rgb(self) -> bool:
        """Whether `adjust_rgb` changes pixels before the colour lookup."""
        return self.saturation != 0 or self.input_remap is not None


def adjust_rgb(rgb: Buffer, pixels: int, colour: ColourSetup) -> Buffer:
    """Apply saturation, then the input remap, to `pixels` RGB pixels.

    Every step is per pixel, so a whole band can go through in one call.

    Returns:
        The adjusted pixels, or `rgb` itself when nothing applies.
    """
    # Saturation is per-pixel; brightness/contrast/RGB-keys go through the
    # pre-LUT input remap.
    if colour.saturation != 0:
        rgb = adjust_saturation(rgb, pixels, colour.saturation)
    if colour.input_remap is not None:
        rgb = apply_input_remap_rgb(rgb, pixels, *colour.input_remap)
    return rgb


def fine_overrun_planes(row: npt.NDArray[np.uint8], width: int, inks: tuple[npt.NDArray[np.uint8], ...]) -> bool:
    """Whether the original's Fine dither clears byte 0 of the M, Y and K rows.

    `dither_ordered_4ch_4bpp` @ 08061404 fills a run without ink (all four
    channels zero, at least 2 identical RGB pixels) with memset and then
    writes one more zero byte after it. When that run ends the line, the
    byte lies just past the row buffer; the buffers are contiguous in the
    order C, M, Y, K, ceil(width / 8) * 4 bytes apart, so it is byte 0 of
    the next plane. `inks` are the K, C, M, Y rows (255 = no ink).

    Returns:
        True if byte 0 (pixels 0 and 1) of M, Y and K must be zeroed.
    """
    if any(arr[width - 1] != 255 for arr in inks):
        return False
    return bool(fine_overrun_rows(row[None, :], width)[0])


def _fine_overrun_possible(width: int) -> bool:
    """Whether a row of `width` pixels ends exactly at the end of its plane buffer (see `fine_overrun_planes`).

    Returns:
        True if the overrun byte can land in the next plane.
    """
    return width >= 2 and width // 2 == (width + 7) // 8 * 4


def fine_overrun_rows(band: npt.NDArray[np.uint8], width: int) -> npt.NDArray[np.uint8]:
    """The RGB part of `fine_overrun_planes` for every row of `band`.

    `fine_overrun_planes` and `render_band_fine` add the no-ink test of
    the last pixel. The test looks at the input pixels, before saturation
    and the input remap.

    Returns:
        One uint8 flag per row: the geometry allows the overrun and the last
        two input pixels are equal.
    """
    if not _fine_overrun_possible(width):
        return np.zeros(band.shape[0], dtype=np.uint8)
    tail = band[:, 3 * width - 6 : 3 * width]
    return (tail[:, :3] == tail[:, 3:]).all(axis=1).view(np.uint8)


def _clear_first_byte(data: bytes) -> bytes:
    return b"\x00" + data[1:]


def encode_lines(
    blocks: RowBlocks,
    width: int,
    sw: int,
    colour: ColourSetup,
    channels: dict[str, DitherChannel],
    *,
    is_fine: bool,
) -> Iterator[LineCodes]:
    """Colour-convert, dither and encode the page one scanline at a time.

    Reference path for the native band kernels; used on installs without
    them and for dither channels they do not cover (Normal channels without
    a threshold matrix, Fine channels without nibble tables).

    Yields:
        Encoded planes of each input row, top to bottom.
    """
    dither_fn = DITHER_FNS[is_fine]
    encoders = PLANE_ENCODERS[is_fine]
    pad_arr = np.full(sw - width, 255, dtype=np.uint8) if sw > width else None

    for line_idx, (row, is_white) in enumerate(iter_rows(blocks)):
        if is_white:
            yield BLANK_LINE
            continue
        rgb_row = adjust_rgb(row, width, colour)
        k_arr, c_arr, m_arr, y_arr = color_lut.rgb_to_cmyk_lut_arr(rgb_row, width, colour.table)
        if pad_arr is not None:
            k_arr = np.concatenate((k_arr, pad_arr))
            c_arr = np.concatenate((c_arr, pad_arr))
            m_arr = np.concatenate((m_arr, pad_arr))
            y_arr = np.concatenate((y_arr, pad_arr))
        k_dots = dither_fn(k_arr, line_idx, sw, channels["K"])
        c_dots = dither_fn(c_arr, line_idx, sw, channels["C"])
        m_dots = dither_fn(m_arr, line_idx, sw, channels["M"])
        y_dots = dither_fn(y_arr, line_idx, sw, channels["Y"])
        if is_fine and fine_overrun_planes(row, width, (k_arr, c_arr, m_arr, y_arr)):
            k_dots, m_dots, y_dots = map(_clear_first_byte, (k_dots, m_dots, y_dots))
        yield encoders[0](k_dots), encoders[1](c_dots), encoders[2](m_dots), encoders[3](y_dots)


# Rows converted to luma at a time. A flipped page arrives as one whole-page
# block, and `_luma` widens to uint32 with temporaries, ~9 bytes per input
# byte: unsplit, an A4 page peaks near 1 GB.
_MONO_CHUNK_ROWS = 128


def _split_rows(blocks: RowBlocks, rows: int) -> Iterator[npt.NDArray[np.uint8]]:
    """Cut row blocks into views of at most `rows` rows.

    Yields:
        (rows, width * 3) uint8 views, top to bottom.
    """
    for block in blocks:
        for lo in range(0, block.shape[0], rows):
            yield block[lo : lo + rows]


def _luma(rgb: npt.NDArray[np.uint8], width: int) -> npt.NDArray[np.uint8]:
    """Rec. 601 luma of (rows, width * 3) RGB rows, rounded as the original driver does.

    Returns:
        (rows, width) uint8 array.
    """
    px = rgb.reshape(-1, width, 3).astype(np.uint32)
    return ((px[..., 0] * 299 + px[..., 1] * 587 + px[..., 2] * 114 + 499) // 1000).astype(np.uint8)


def encode_lines_mono(
    blocks: RowBlocks,
    width: int,
    sw: int,
    colour: ColourSetup,
    channels: dict[str, DitherChannel],
    *,
    is_fine: bool,
) -> Iterator[LineCodes]:
    """Grayscale counterpart of `encode_lines`: luma into the K plane only.

    Mirrors `compress_separate_mono` in the original driver: K ink is
    255 - luma through the mono profile, which is the identity table, and
    C/M/Y stay empty. Saturation and vivid do not apply in this mode; the
    brightness/contrast input remap does.

    Yields:
        Encoded planes of each input row, top to bottom.
    """
    dither_fn = DITHER_FNS[is_fine]
    encode_k = PLANE_ENCODERS[is_fine][0]
    pad_arr = np.full(sw - width, 255, dtype=np.uint8) if sw > width else None
    line_idx = 0

    for block in _split_rows(blocks, _MONO_CHUNK_ROWS):
        n = block.shape[0]
        white = white_rows(block).tolist()
        rgb = block
        if colour.input_remap is not None:
            rgb = page_rows(
                apply_input_remap_rgb(np.ascontiguousarray(block), n * width, *colour.input_remap), width, n
            )
        luma = _luma(rgb, width)
        if pad_arr is not None:
            luma = np.concatenate((luma, np.broadcast_to(pad_arr, (n, pad_arr.size))), axis=1)
        for row, is_white in zip(luma, white, strict=True):
            if is_white:
                yield BLANK_LINE
            else:
                yield (encode_k(dither_fn(row, line_idx, sw, channels["K"])), b"", b"", b"")
            line_idx += 1

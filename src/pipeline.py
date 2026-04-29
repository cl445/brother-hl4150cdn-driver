"""End-to-end PPM → XL2HB pipeline.

`filter_page` is the single-page entry point. `filter_duplex_pages`
shares one XL2HB session across multiple pages so duplex jobs come out
as a single print job.
"""

import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import BinaryIO

import numpy as np

from brother_encode import encode_c_plane, encode_fine_plane, encode_m_plane_10, encode_plane
from dither import DitherChannel, dither_channel_1bpp_arr, dither_channel_4bpp_arr, load_dither_tables
from saturation import adjust_saturation
from settings import (
    DUPLEX_MAP,
    ColorMatching,
    ImproveOutput,
    MonoColor,
    PageSize,
    PrintSettings,
    Resolution,
    input_slot_to_tray,
)
from tone_curve import apply_tone_curve_arr, build_tone_curve
from transforms import (
    apply_input_remap_rgb,
    apply_vivid,
    build_input_remap_lut,
    rgb_line_to_cmyk_intensities_arr,
)
from xl2hb import (
    FLUSH_ORDER,
    PAPER_SIZES,
    PlaneBuffer,
    XL2HBWriter,
    generate_pjl_footer,
    generate_pjl_header,
    get_image_dimensions,
    get_image_dimensions_fine,
)

logger = logging.getLogger(__name__)

# Per-mode dither dispatcher (intensity ndarray, line_idx, sw, channel) -> packed bytes.
_DITHER_FNS = {False: dither_channel_1bpp_arr, True: dither_channel_4bpp_arr}

# Per-mode plane encoders, keyed by plane id; partial() pins the K/Y label.
_PlaneEncoder = Callable[[bytes], bytes]
_PLANE_ENCODERS: dict[bool, dict[int, _PlaneEncoder]] = {
    False: {  # Normal mode: 12-bit K/Y RLE, 20-bit C, 10-bit M
        0: partial(encode_plane, plane="K"),
        1: encode_c_plane,
        2: encode_m_plane_10,
        3: partial(encode_plane, plane="Y"),
    },
    True: dict.fromkeys(range(4), encode_fine_plane),  # Fine mode: same encoder for all planes
}


def _render_page(
    w: XL2HBWriter,
    width: int,
    height: int,
    pixel_data: bytes,
    settings: PrintSettings,
    channels: dict[str, DitherChannel],
    page_size: PageSize,
) -> None:
    """Render one page within an already-open session.

    Handles: BeginPage -> scanline loop -> flush -> EndPage.
    """
    is_fine = settings.resolution == Resolution.FINE

    if is_fine:
        sw, sh = get_image_dimensions_fine(page_size)
        _, paper_h = PAPER_SIZES.get(page_size, PAPER_SIZES["A4"])
        bpl = (sw + 1) // 2  # 4bpp: 2 pixels per byte
    else:
        _, paper_h = PAPER_SIZES.get(page_size, PAPER_SIZES["A4"])
        sw, sh = get_image_dimensions(page_size)
        bpl = (sw + 7) // 8  # 1bpp: 8 pixels per byte

    w.write_begin_page(
        media_size=page_size,
        media_source=1,
        media_type=settings.media_type,
        duplex_mode=DUPLEX_MAP.get(settings.duplex, 0),
    )
    w.write_set_page_origin()
    w.write_begin_image(sw, sh, copies=settings.copies, fine=is_fine)

    plane_bufs = {i: PlaneBuffer(plane_id=i, bpl=bpl, fine=is_fine) for i in range(4)}

    def flush_plane(pid: int, next_line: int) -> None:
        pb = plane_bufs[pid]
        result = pb.flush()
        if result:
            start, count, blob = result
            w.write_read_image(start, count, pid, blob)
        pb.reset(next_line)

    pad_arr = np.full(sw - width, 255, dtype=np.uint8) if sw > width else None

    # Pre-build LUTs (constant per page).
    tone_lut = None
    input_remap = None
    if settings.gamma_select is not None:
        tone_lut = build_tone_curve(settings.brightness, settings.contrast, settings.gamma_select)
    elif (
        settings.brightness != 0
        or settings.contrast != 0
        or settings.red != 0
        or settings.green != 0
        or settings.blue != 0
    ):
        input_remap = (
            build_input_remap_lut(settings.brightness, settings.contrast, settings.red),
            build_input_remap_lut(settings.brightness, settings.contrast, settings.green),
            build_input_remap_lut(settings.brightness, settings.contrast, settings.blue),
        )

    dither_fn = _DITHER_FNS[is_fine]
    encoders = _PLANE_ENCODERS[is_fine]

    row_bytes = width * 3
    blank_plane = bytes(bpl)
    blank_planes = dict.fromkeys(range(4), blank_plane)
    blank_plane_comp = dict.fromkeys(range(4), b"")
    # apply_input_remap_rgb explicitly preserves (255,255,255); saturation
    # and vivid leave the gray axis untouched; the LUT clamps white→0 ink.
    # Only tone_curve can deposit ink on white, so skip the short-circuit
    # when gamma_select is active.
    white_row = b"\xff\xff\xff" * width if tone_lut is None else None

    for line_idx in range(paper_h):
        if line_idx < height:
            row_start = line_idx * row_bytes
            rgb_row = pixel_data[row_start : row_start + row_bytes]
            if rgb_row == white_row:
                plane_data = blank_planes
            else:
                # Saturation and vivid are per-pixel; brightness/contrast/RGB-keys
                # go through the pre-LUT input remap.
                if settings.saturation != 0:
                    rgb_row = adjust_saturation(rgb_row, width, settings.saturation)
                elif settings.color_matching == ColorMatching.VIVID:
                    rgb_row = apply_vivid(rgb_row, width)
                if input_remap is not None:
                    rgb_row = apply_input_remap_rgb(rgb_row, width, *input_remap)
                k_arr, c_arr, m_arr, y_arr = rgb_line_to_cmyk_intensities_arr(
                    rgb_row, width, color_matching=settings.color_matching
                )
                if tone_lut is not None:
                    k_arr, c_arr, m_arr, y_arr = apply_tone_curve_arr(k_arr, c_arr, m_arr, y_arr, tone_lut)
                if pad_arr is not None:
                    k_arr = np.concatenate((k_arr, pad_arr))
                    c_arr = np.concatenate((c_arr, pad_arr))
                    m_arr = np.concatenate((m_arr, pad_arr))
                    y_arr = np.concatenate((y_arr, pad_arr))

                plane_data = {
                    0: dither_fn(k_arr, line_idx, sw, channels["K"]),
                    1: dither_fn(c_arr, line_idx, sw, channels["C"]),
                    2: dither_fn(m_arr, line_idx, sw, channels["M"]),
                    3: dither_fn(y_arr, line_idx, sw, channels["Y"]),
                }
        else:
            plane_data = blank_planes

        if plane_data is blank_planes:
            plane_comp = blank_plane_comp
        else:
            plane_comp = {pid: encoders[pid](data) for pid, data in plane_data.items()}

        # Per-plane independent flush. Process planes in order C, M, Y, K.
        # Empty line + accumulated data → flush that plane. Non-empty line →
        # append; if buffer nearly full → flush that plane.
        for pid in FLUSH_ORDER:
            comp = plane_comp[pid]
            pb = plane_bufs[pid]
            if not comp:
                if pb.line_count > 0:
                    flush_plane(pid, line_idx)
            else:
                pb.append_scanline(comp, line_idx)
                if pb.is_nearly_full():
                    flush_plane(pid, line_idx + 1)

    for pid in FLUSH_ORDER:
        pb = plane_bufs[pid]
        result = pb.flush()
        if result:
            start, count, blob = result
            w.write_read_image(start, count, pid, blob)

    w.write_end_image()
    w.write_end_page(copies=settings.copies)


def _init_channels(settings: PrintSettings, lut_dir: str | None = None) -> dict[str, DitherChannel]:
    """Initialize dither channels for the current print settings.

    `lut_dir` defaults to the installed `src/lut/` directory next to this
    module. The factory inside `load_dither_tables` handles the fine→normal
    fallback and the Bayer fallback when no BRCD set matches.

    Returns:
        Channel dict keyed by 'K', 'C', 'M', 'Y'.
    """
    if lut_dir is None:
        lut_dir = str(Path(__file__).resolve().parent / "lut")

    return load_dither_tables(
        lut_dir,
        fine=settings.resolution == Resolution.FINE,
        toner_save=settings.toner_save,
    )


def filter_page(
    width: int,
    height: int,
    pixel_data: bytes,
    settings: PrintSettings,
    output: BinaryIO,
    lut_dir: str | None = None,
) -> None:
    """Convert PPM pixel data to XL2HB and write to output."""
    if settings.skip_blank and pixel_data == b"\xff" * len(pixel_data):
        return

    filter_duplex_pages([(width, height, pixel_data)], settings, output, lut_dir=lut_dir)

    _, paper_h = PAPER_SIZES.get(settings.page_size, PAPER_SIZES["A4"])
    logger.debug("Processed %d lines, %dx%d input", paper_h, width, height)


def filter_duplex_pages(
    pages: list[tuple[int, int, bytes]],
    settings: PrintSettings,
    output: BinaryIO,
    lut_dir: str | None = None,
) -> None:
    """Render multiple pages inside a single XL2HB session (for duplex).

    Args:
        pages: list of (width, height, pixel_data) tuples
        settings: PrintSettings (should have duplex != "None" for actual duplex)
        output: writable binary stream
        lut_dir: optional path to BRCD LUT directory
    """
    page_size = settings.page_size

    # PJL header always reports 600 dpi; Fine mode differs only in dithering.
    color = settings.mono_color != MonoColor.MONO
    pjl = generate_pjl_header(
        resolution=600,
        color=color,
        # ECONOMODE is always OFF; toner-save is implemented through the
        # -TS_cache09.bin dither tables instead.
        economode=False,
        less_paper_curl=settings.improve_output == ImproveOutput.LESS_PAPER_CURL,
        fix_intensity=settings.improve_output == ImproveOutput.FIX_INTENSITY,
        apt_mode=(settings.resolution == Resolution.FINE),
        improve_gray=settings.improve_gray,
        ucrgcr=settings.improve_gray,
        source_tray=input_slot_to_tray(settings.input_slot),
    )
    output.write(pjl)

    # XL2HB stream — one session wrapping all pages.
    w = XL2HBWriter(output)
    w.write_stream_header()
    w.write_begin_session()
    w.write_open_data_source()

    channels = _init_channels(settings, lut_dir=lut_dir)

    for width, height, pixel_data in pages:
        _render_page(w, width, height, pixel_data, settings, channels, page_size)

    w.write_close_data_source()
    w.write_end_session()

    output.write(generate_pjl_footer())

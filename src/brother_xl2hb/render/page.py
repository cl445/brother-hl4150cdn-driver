"""One page of the XL2HB stream, from BeginPage to EndPage."""

import functools
import itertools
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt

from brother_xl2hb.color.dither import (
    DitherChannel,
    load_dither_tables,
)
from brother_xl2hb.color.transforms import build_input_remap_lut
from brother_xl2hb.models.base import PageSend, PrinterModel
from brother_xl2hb.protocol.xl2hb import FLUSH_ORDER, PlaneBuffer, XL2HBWriter
from brother_xl2hb.render.bands import (
    band_kernel_colour,
    encode_lines_banded,
    encode_lines_mono_banded,
    mono_band_kernel_ready,
)
from brother_xl2hb.render.lines import BLANK_LINE, ColourSetup, LineCodes, encode_lines, encode_lines_mono
from brother_xl2hb.render.rows import BUFFER_TYPES, LineBlocks, Page, RowBlocks, numbered_blocks, page_rows
from brother_xl2hb.settings import MonoColor, PrintSettings, Resolution


@dataclass(frozen=True, slots=True)
class JobContext:
    """What stays the same for every page of a job.

    Attributes:
        settings: the job's settings, restricted to what `model` supports.
        model: the printer.
        channels: the dither channels for `settings` (see `init_channels`).
    """

    settings: PrintSettings
    model: PrinterModel
    channels: dict[str, DitherChannel]

    @classmethod
    def create(cls, settings: PrintSettings, model: PrinterModel) -> Self:
        """The context for `settings` on `model`, with its dither channels loaded.

        Returns:
            The job context.
        """
        return cls(settings, model, init_channels(settings, model))


def render_page(w: XL2HBWriter, page: Page, job: JobContext, send: PageSend) -> None:
    """Render one page within an already-open session.

    Handles: BeginPage -> scanline loop -> flush -> EndPage, as `send`
    says (bottom-up with the back-side marker, PageCopies; see
    `PrinterModel.page_sends`). Scanlines are read as views into the
    page's pixel data, which may also be a stream of row blocks, and are
    read once, top to bottom, flipped pages included.

    Raises:
        ValueError: If the page has fewer than its `height` rows.
    """
    width, height, pixel_data = page
    settings, model, channels = job.settings, job.model, job.channels
    is_fine = settings.resolution == Resolution.FINE
    mono = settings.mono_color == MonoColor.MONO
    page_size = settings.page_size
    _, paper_h = model.paper_sizes[page_size]
    sw, sh = model.image_dimensions(page_size, fine=is_fine)

    w.write_begin_page(
        media_size=model.media_size_code(page_size),
        media_source=model.media_source[settings.input_slot],
        media_type=model.media_type_code(settings.media_type),
        duplex_mode=model.duplex_page_mode.get(settings.duplex),
        back_side_marker=send.flip,
    )
    w.write_set_page_origin(*model.page_origin)
    w.write_begin_image(sw, sh, copies=send.copies, fine=is_fine, color=not mono)

    encode = _line_encoder(settings, model, channels, width, sw)
    blocks: RowBlocks = (page_rows(pixel_data, width, height),) if isinstance(pixel_data, BUFFER_TYPES) else pixel_data
    if send.flip:
        lines = _flipped_lines(encode, blocks, height, paper_h)
        height = paper_h
    else:
        lines = encode(numbered_blocks(blocks))
    bpl = (sw + 1) // 2 if is_fine else (sw + 7) // 8  # 4bpp: 2 pixels per byte, 1bpp: 8
    _write_planes(w, lines, height, paper_h, bpl=bpl, is_fine=is_fine)

    w.write_end_image()
    w.write_end_page(copies=send.copies)


LineEncoder = Callable[[LineBlocks], Iterator[LineCodes]]
"""Encodes row blocks (with their first page lines) to scanlines, in input order."""


def _line_encoder(
    settings: PrintSettings, model: PrinterModel, channels: dict[str, DitherChannel], width: int, sw: int
) -> LineEncoder:
    """The scanline encoder for the page: a native band kernel where one applies, else per line.

    Returns:
        The encoder.
    """
    is_fine = settings.resolution == Resolution.FINE
    colour = _colour_setup(settings, model)
    if settings.mono_color == MonoColor.MONO:
        if mono_band_kernel_ready(channels, is_fine=is_fine):
            return functools.partial(encode_lines_mono_banded, width=width, sw=sw, colour=colour, channels=channels)
        return functools.partial(
            encode_lines_mono, width=width, sw=sw, colour=colour, channels=channels, is_fine=is_fine
        )
    kernel_colour = band_kernel_colour(colour.table, channels, is_fine=is_fine)
    if kernel_colour is not None:
        return functools.partial(
            encode_lines_banded,
            width=width,
            sw=sw,
            colour=colour,
            channels=channels,
            kernel_colour=kernel_colour,
            is_fine=is_fine,
        )
    return functools.partial(encode_lines, width=width, sw=sw, colour=colour, channels=channels, is_fine=is_fine)


def _flipped_lines(encode: LineEncoder, blocks: RowBlocks, height: int, paper_h: int) -> Iterator[LineCodes]:
    """Encode the page mirrored top to bottom over the full paper height (columns stay).

    Brother's filter sends long-edge duplex back pages and BRReverse pages
    this way (verified byte-for-byte against captures). Each output line
    depends only on its source row and its line number, so the blocks are
    read top to bottom as they come and encoded mirrored, source row r as
    line paper_h - 1 - r; the encoded lines, not the raster, are kept until
    the page is complete. Rows past `paper_h` are dropped; rows missing
    below a short page become blank lines at the top.

    Yields:
        The page's paper_h encoded lines, top to bottom as they are sent.

    Raises:
        ValueError: If `blocks` do not add up to `height` rows.
    """
    sizes: list[int] = []

    def mirrored() -> Iterator[tuple[npt.NDArray[np.uint8], int]]:
        filled = 0
        for block in blocks:
            rows = min(block.shape[0], height - filled)
            keep = max(0, min(rows, paper_h - filled))
            if keep:
                sizes.append(keep)
                yield block[:keep][::-1], paper_h - filled - keep
            filled += rows
        if filled != height:
            msg = f"page has {filled} rows, expected {height}"
            raise ValueError(msg)

    codes = list(encode(mirrored()))
    yield from itertools.repeat(BLANK_LINE, paper_h - len(codes))
    end = len(codes)
    for size in reversed(sizes):
        yield from codes[end - size : end]
        end -= size


def _colour_setup(settings: PrintSettings, model: PrinterModel) -> ColourSetup:
    """The page's colour grid and brightness/contrast/RGB-key input remap.

    The cmyk profile (colour matching None) takes no remap: the original
    sends it through cmyk_basic, which skips compress_color_manage.
    Grayscale keeps brightness and contrast whatever the colour matching,
    but the original zeroes the RGB keys there (printer_setup.c, RLE mode).

    Returns:
        The colour setup for the scanline encoders.
    """
    table = model.color_table(settings)
    mono = settings.mono_color == MonoColor.MONO
    keys = (0, 0, 0) if mono else (settings.red, settings.green, settings.blue)
    input_remap = None
    if (mono or table.profile != "cmyk") and (settings.brightness != 0 or settings.contrast != 0 or any(keys)):
        r, g, b = (build_input_remap_lut(settings.brightness, settings.contrast, key) for key in keys)
        input_remap = (r, g, b)
    return ColourSetup(settings, table, input_remap)


def _write_planes(
    w: XL2HBWriter, lines: Iterator[LineCodes], height: int, paper_h: int, *, bpl: int, is_fine: bool
) -> None:
    """Buffer the encoded scanlines per plane and write them as ReadImage blocks.

    Lines past `height` down to `paper_h` are blank. Each plane flushes on
    its own, in the order C, M, Y, K: an empty line flushes what the plane
    has gathered, a non-empty one is appended and flushes a nearly full
    buffer.

    Raises:
        ValueError: If `lines` ends before `height` lines.
    """
    plane_bufs = [PlaneBuffer(plane_id=i, bpl=bpl, fine=is_fine) for i in range(4)]

    def write_plane(pid: int) -> None:
        result = plane_bufs[pid].flush()
        if result:
            start, count, blob = result
            w.write_read_image(start, count, pid, blob)

    def flush_plane(pid: int, next_line: int) -> None:
        write_plane(pid)
        plane_bufs[pid].reset(next_line)

    for line_idx in range(paper_h):
        if line_idx < height:
            line = next(lines, None)
            if line is None:
                msg = f"page ended after {line_idx} rows, expected {height}"
                raise ValueError(msg)
            plane_comp = line
        else:
            plane_comp = BLANK_LINE

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
        write_plane(pid)


def init_channels(settings: PrintSettings, model: PrinterModel) -> dict[str, DitherChannel]:
    """Initialize dither channels for the current print settings.

    The BRCD tables are `model.dither_cache`'s set in `model.lut_dir`
    (`lut/<key>/` inside the package, in the repository and when installed).
    `load_dither_tables` handles the fine→normal fallback and the Bayer
    fallback when no BRCD set matches.

    Returns:
        Channel dict keyed by 'K', 'C', 'M', 'Y'.
    """
    return load_dither_tables(
        str(model.lut_dir),
        cache=model.dither_cache,
        fine=settings.resolution == Resolution.FINE,
        toner_save=settings.toner_save,
    )

"""One page of the XL2HB stream, from BeginPage to EndPage."""

from dataclasses import dataclass
from typing import Self

from brother_xl2hb.color.dither import (
    DitherChannel,
    load_dither_tables,
)
from brother_xl2hb.color.transforms import build_input_remap_lut
from brother_xl2hb.models.base import PageSend, PrinterModel
from brother_xl2hb.protocol.xl2hb import FLUSH_ORDER, PlaneBuffer, XL2HBWriter
from brother_xl2hb.render.bands import band_kernel_colour, encode_lines_banded
from brother_xl2hb.render.lines import BLANK_LINE, ColourSetup, encode_lines, encode_lines_mono
from brother_xl2hb.render.rows import BUFFER_TYPES, Page, RowBlocks, collect_rows, flip_vertical, page_rows
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
    page's pixel data, which may also be a stream of row blocks; only
    flipped pages are collected into one buffer to be mirrored.

    Raises:
        ValueError: If the page has fewer than its `height` rows.
    """
    width, height, pixel_data = page
    settings, model, channels = job.settings, job.model, job.channels
    is_fine = settings.resolution == Resolution.FINE
    page_size = settings.page_size
    _, paper_h = model.paper_sizes[page_size]

    sw, sh = model.image_dimensions(page_size, fine=is_fine)
    bpl = (sw + 1) // 2 if is_fine else (sw + 7) // 8  # 4bpp: 2 pixels per byte, 1bpp: 8

    # A flipped page is mirrored top to bottom, columns stay.
    if send.flip:
        blocks: RowBlocks = (flip_vertical(collect_rows(pixel_data, width, height), paper_h),)
        height = paper_h
    elif isinstance(pixel_data, BUFFER_TYPES):
        blocks = (page_rows(pixel_data, width, height),)
    else:
        blocks = pixel_data

    w.write_begin_page(
        media_size=model.media_size_code(page_size),
        media_source=model.media_source[settings.input_slot],
        media_type=model.media_type_code(settings.media_type),
        duplex_mode=model.duplex_page_mode.get(settings.duplex),
        back_side_marker=send.flip,
    )
    w.write_set_page_origin(*model.page_origin)
    mono = settings.mono_color == MonoColor.MONO
    w.write_begin_image(sw, sh, copies=send.copies, fine=is_fine, color=not mono)

    plane_bufs = {i: PlaneBuffer(plane_id=i, bpl=bpl, fine=is_fine) for i in range(4)}

    def write_plane(pid: int) -> None:
        result = plane_bufs[pid].flush()
        if result:
            start, count, blob = result
            w.write_read_image(start, count, pid, blob)

    def flush_plane(pid: int, next_line: int) -> None:
        write_plane(pid)
        plane_bufs[pid].reset(next_line)

    # Pre-build LUTs (constant per page). The cmyk profile (colour matching
    # None) takes no brightness/contrast/RGB-key remap: the original sends it
    # through cmyk_basic, which skips compress_color_manage.
    table = model.color_table(settings)
    input_remap = None
    if table.profile != "cmyk" and (
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

    colour = ColourSetup(settings, table, input_remap)
    kernel_colour = None if mono else band_kernel_colour(colour.table, channels, is_fine=is_fine)
    if mono:
        lines = encode_lines_mono(blocks, width, sw, colour, channels, is_fine=is_fine)
    elif kernel_colour is not None:
        lines = encode_lines_banded(blocks, width, sw, colour, channels, kernel_colour, is_fine=is_fine)
    else:
        lines = encode_lines(blocks, width, sw, colour, channels, is_fine=is_fine)

    for line_idx in range(paper_h):
        if line_idx < height:
            line = next(lines, None)
            if line is None:
                msg = f"page ended after {line_idx} rows, expected {height}"
                raise ValueError(msg)
            plane_comp = line
        else:
            plane_comp = BLANK_LINE

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
        write_plane(pid)

    w.write_end_image()
    w.write_end_page(copies=send.copies)


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

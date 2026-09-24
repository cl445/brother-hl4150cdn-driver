"""End-to-end PPM → XL2HB pipeline.

`filter_page` is the single-page entry point. `filter_pages` renders a
whole job (simplex or duplex) inside one XL2HB session, as the
manufacturer's filter does.
"""

import logging
import os
import tempfile
from collections import deque
from collections.abc import Buffer, Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from functools import cache, partial
from pathlib import Path
from typing import BinaryIO, NamedTuple

import numpy as np
import numpy.typing as npt

import color_lut
from brother_encode import encode_c_plane, encode_fine_plane, encode_m_plane_10, encode_plane
from dither import DitherChannel, dither_channel_1bpp_arr, dither_channel_4bpp_arr, load_dither_tables
from saturation import adjust_saturation
from settings import (
    DUPLEX_MAP,
    MEDIA_SOURCE,
    DuplexMode,
    ImproveOutput,
    MonoColor,
    PrintSettings,
    Resolution,
)
from transforms import (
    apply_input_remap_rgb,
    build_input_remap_lut,
    color_table,
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

try:
    from _band_fast import render_band  # type: ignore[import-not-found]

    HAS_BAND_KERNEL = True
except ImportError:
    HAS_BAND_KERNEL = False

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


PageBuffer = bytes | npt.NDArray[np.uint8]
"""A whole RGB page as packed bytes or as a (height, width, 3) uint8 array.

The array may be a strided view (e.g. the printable window of a larger
render) as long as each row is contiguous.
"""

RowBlocks = Iterable[npt.NDArray[np.uint8]]
"""A page streamed as (rows, width * 3) uint8 blocks with contiguous rows.

Consumed once, top to bottom (see `page_stream.iter_ppm_pages`).
"""

PageData = PageBuffer | RowBlocks


def _page_rows(pixel_data: Buffer, width: int, height: int) -> npt.NDArray[np.uint8]:
    """View an RGB page as (height, width * 3) rows without copying.

    Returns:
        uint8 array whose rows are contiguous RGB scanlines.
    """
    if isinstance(pixel_data, np.ndarray):
        return pixel_data.reshape(height, width * 3)
    return np.frombuffer(pixel_data, dtype=np.uint8, count=width * height * 3).reshape(height, width * 3)


_BUFFER_TYPES = (bytes, bytearray, memoryview, np.ndarray)


def collect_rows(pixel_data: PageData, width: int, height: int) -> npt.NDArray[np.uint8]:
    """Return the page as one (height, width * 3) array.

    A view for packed bytes and arrays; streamed row blocks are copied into
    a single page-sized buffer. Only needed where the whole page must be
    seen at once (long-edge back pages, skip-blank).

    Returns:
        uint8 array whose rows are contiguous RGB scanlines.

    Raises:
        ValueError: If streamed blocks do not add up to `height` rows.
    """
    if isinstance(pixel_data, _BUFFER_TYPES):
        return _page_rows(pixel_data, width, height)
    page = np.empty((height, width * 3), dtype=np.uint8)
    filled = 0
    for block in pixel_data:
        rows = min(block.shape[0], height - filled)
        page[filled : filled + rows] = block[:rows]
        filled += rows
    if filled != height:
        msg = f"page has {filled} rows, expected {height}"
        raise ValueError(msg)
    return page


def _white_rows(block: npt.NDArray[np.uint8]) -> npt.NDArray[np.bool_]:
    """Return which rows of `block` are pure white (all bytes 255).

    White rows carry no ink whatever the settings: apply_input_remap_rgb
    preserves (255,255,255), saturation and vivid leave the gray axis
    untouched, and every colour table maps white to zero ink.
    """
    if not block.shape[1]:
        return np.ones(block.shape[0], dtype=np.bool_)
    return block.min(axis=1) == 255


def _iter_rows(blocks: RowBlocks) -> Iterator[tuple[npt.NDArray[np.uint8], bool]]:
    """Yield (scanline, is_pure_white) for every row of every block.

    The white test runs vectorised per block.

    Yields:
        Contiguous RGB scanline view and whether all its bytes are 255.
    """
    for block in blocks:
        yield from zip(block, _white_rows(block).tolist(), strict=True)


def is_blank_page(pixel_data: PageBuffer) -> bool:
    """Return True if every byte of the page is 255 (pure white).

    Works on packed bytes and on strided array views without copying.
    """
    arr = pixel_data if isinstance(pixel_data, np.ndarray) else np.frombuffer(pixel_data, dtype=np.uint8)
    return arr.size == 0 or int(arr.min()) == 255


def _flip_vertical(rows: npt.NDArray[np.uint8], paper_h: int) -> npt.NDArray[np.uint8]:
    """Mirror page rows top-to-bottom over the full paper height.

    Brother's filter sends long-edge duplex back pages this way (verified
    byte-for-byte against captures of brhl4150cdnfilter). Rows missing below
    a short page become white rows at the top of the flipped page.

    Returns:
        (paper_h, row_bytes) array; a reversed view when the page covers the
        full paper height, otherwise a white-padded copy.
    """
    if rows.shape[0] >= paper_h:
        return rows[:paper_h][::-1]
    page = np.full((paper_h, rows.shape[1]), 255, dtype=np.uint8)
    page[: rows.shape[0]] = rows
    return page[::-1]


LineCodes = tuple[bytes, bytes, bytes, bytes]
"""Encoded K, C, M, Y data of one scanline; b"" for a plane without ink."""

_BLANK_LINE: LineCodes = (b"", b"", b"", b"")
_EMPTY_TABLE = np.empty(0, dtype=np.uint8)

# Scanlines per band handed to the native kernel; also the unit of work
# for the render threads.
_BAND_ROWS = 128


class _ColourSetup(NamedTuple):
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
        """Whether `_adjust_rgb` changes pixels before the colour lookup."""
        return self.saturation != 0 or self.input_remap is not None


def _adjust_rgb(rgb: Buffer, pixels: int, colour: _ColourSetup) -> Buffer:
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


def _encode_lines(
    blocks: RowBlocks,
    width: int,
    sw: int,
    colour: _ColourSetup,
    channels: dict[str, DitherChannel],
    *,
    is_fine: bool,
) -> Iterator[LineCodes]:
    """Colour-convert, dither and encode the page one scanline at a time.

    Reference path for Fine mode, for installs without the native band
    kernel, and for dither channels without a threshold matrix.

    Yields:
        Encoded planes of each input row, top to bottom.
    """
    dither_fn = _DITHER_FNS[is_fine]
    encoders = _PLANE_ENCODERS[is_fine]
    pad_arr = np.full(sw - width, 255, dtype=np.uint8) if sw > width else None

    for line_idx, (row, is_white) in enumerate(_iter_rows(blocks)):
        if is_white:
            yield _BLANK_LINE
            continue
        rgb_row = _adjust_rgb(row, width, colour)
        k_arr, c_arr, m_arr, y_arr = color_lut.rgb_to_cmyk_lut_arr(rgb_row, width, colour.table)
        if pad_arr is not None:
            k_arr = np.concatenate((k_arr, pad_arr))
            c_arr = np.concatenate((c_arr, pad_arr))
            m_arr = np.concatenate((m_arr, pad_arr))
            y_arr = np.concatenate((y_arr, pad_arr))
        yield (
            encoders[0](dither_fn(k_arr, line_idx, sw, channels["K"])),
            encoders[1](dither_fn(c_arr, line_idx, sw, channels["C"])),
            encoders[2](dither_fn(m_arr, line_idx, sw, channels["M"])),
            encoders[3](dither_fn(y_arr, line_idx, sw, channels["Y"])),
        )


def _luma(rgb: npt.NDArray[np.uint8], width: int) -> npt.NDArray[np.uint8]:
    """Rec. 601 luma of (rows, width * 3) RGB rows, rounded as the original driver does.

    Returns:
        (rows, width) uint8 array.
    """
    px = rgb.reshape(-1, width, 3).astype(np.uint32)
    return ((px[..., 0] * 299 + px[..., 1] * 587 + px[..., 2] * 114 + 499) // 1000).astype(np.uint8)


def _encode_lines_mono(
    blocks: RowBlocks,
    width: int,
    sw: int,
    colour: _ColourSetup,
    channels: dict[str, DitherChannel],
    *,
    is_fine: bool,
) -> Iterator[LineCodes]:
    """Grayscale counterpart of `_encode_lines`: luma into the K plane only.

    Mirrors `compress_separate_mono` in the original driver: K ink is
    255 - luma through the mono profile, which is the identity table, and
    C/M/Y stay empty. Saturation and vivid do not apply in this mode; the
    brightness/contrast input remap does.

    Yields:
        Encoded planes of each input row, top to bottom.
    """
    dither_fn = _DITHER_FNS[is_fine]
    encode_k = _PLANE_ENCODERS[is_fine][0]
    pad_arr = np.full(sw - width, 255, dtype=np.uint8) if sw > width else None
    line_idx = 0

    for block in blocks:
        n = block.shape[0]
        white = _white_rows(block).tolist()
        rgb = block
        if colour.input_remap is not None:
            rgb = _page_rows(
                apply_input_remap_rgb(np.ascontiguousarray(block), n * width, *colour.input_remap), width, n
            )
        luma = _luma(rgb, width)
        if pad_arr is not None:
            luma = np.concatenate((luma, np.broadcast_to(pad_arr, (n, pad_arr.size))), axis=1)
        for row, is_white in zip(luma, white, strict=True):
            if is_white:
                yield _BLANK_LINE
            else:
                yield (encode_k(dither_fn(row, line_idx, sw, channels["K"])), b"", b"", b"")
            line_idx += 1


class _KernelColour(NamedTuple):
    """Colour arguments of `render_band`: the inverse LUT, or the grid to interpolate."""

    inverse: npt.NDArray[np.uint8]
    grid: npt.NDArray[np.int32]
    weights: npt.NDArray[np.uint8]
    black: npt.NDArray[np.int32]


_EMPTY_INT = np.empty(0, dtype=np.int32)


def _band_kernel_colour(
    table: color_lut.ColorTable, channels: dict[str, DitherChannel], *, is_fine: bool
) -> _KernelColour | None:
    """Return the colour arguments for `render_band`, or None if it cannot render the page.

    The kernel covers Normal mode with threshold-matrix dither channels and
    needs the native module. It gathers through the installed inverse LUT of
    `table` when there is one and interpolates the grid otherwise.

    Returns:
        The kernel's colour arguments, or None for the per-line path.
    """
    if not HAS_BAND_KERNEL or is_fine:
        return None
    if any(channel.threshold_matrix is None for channel in channels.values()):
        return None
    inverse = color_lut.inverse_lut(table)
    if inverse is not None:
        return _KernelColour(inverse.reshape(-1), _EMPTY_INT, _EMPTY_TABLE, _EMPTY_INT)
    return _KernelColour(_EMPTY_TABLE, *color_lut.interp_arrays(table))


def _render_threads() -> int:
    """Number of band render threads.

    `BRHL4150CDN_RENDER_THREADS` overrides the default of one thread per
    spare core, at most three; 0 renders on the calling thread.

    Returns:
        Thread count, >= 0.
    """
    configured = os.environ.get("BRHL4150CDN_RENDER_THREADS")
    if configured is not None:
        return max(0, int(configured))
    return max(1, min(3, (os.cpu_count() or 1) - 1))


@cache
def _band_executor(workers: int) -> ThreadPoolExecutor:
    """Return the process-wide pool of `workers` render threads."""
    return ThreadPoolExecutor(max_workers=workers, thread_name_prefix="brhl4150cdn-band")


def _split_bands(blocks: RowBlocks) -> Iterator[tuple[npt.NDArray[np.uint8], int]]:
    """Cut row blocks into bands of at most `_BAND_ROWS` rows.

    Yields:
        (band, page line index of its first row).
    """
    first_line = 0
    for block in blocks:
        for lo in range(0, block.shape[0], _BAND_ROWS):
            band = block[lo : lo + _BAND_ROWS]
            yield band, first_line
            first_line += band.shape[0]


def _encode_lines_banded(
    blocks: RowBlocks,
    width: int,
    sw: int,
    colour: _ColourSetup,
    channels: dict[str, DitherChannel],
    kernel_colour: _KernelColour,
) -> Iterator[LineCodes]:
    """Like `_encode_lines`, but whole bands at a time in the native kernel.

    The kernel releases the GIL, so bands render in parallel on the
    render threads; results are yielded strictly in page order.
    `kernel_colour` comes from `_band_kernel_colour`.

    Yields:
        Encoded planes of each input row, top to bottom.
    """
    # Tile the thresholds up front; the render threads only read them.
    thresholds = tuple(channels[c].tiled_thresholds(sw) for c in "KCMY")

    def encode(band: npt.NDArray[np.uint8], first_line: int) -> list[LineCodes]:
        n = band.shape[0]
        skip = _white_rows(band).view(np.uint8)
        rgb = band
        if colour.adjusts_rgb:
            rgb = _page_rows(_adjust_rgb(np.ascontiguousarray(band), n * width, colour), width, n)
        lengths = np.empty((n, 4), dtype=np.int32)
        data = render_band(rgb, skip, width, first_line, sw, *kernel_colour, *thresholds, lengths)
        codes: list[LineCodes] = []
        pos = 0
        for k_len, c_len, m_len, y_len in lengths.tolist():
            c_pos = pos + k_len
            m_pos = c_pos + c_len
            y_pos = m_pos + m_len
            end = y_pos + y_len
            codes.append((data[pos:c_pos], data[c_pos:m_pos], data[m_pos:y_pos], data[y_pos:end]))
            pos = end
        return codes

    workers = _render_threads()
    if workers == 0:
        for band, first_line in _split_bands(blocks):
            yield from encode(band, first_line)
        return

    executor = _band_executor(workers)
    pending: deque[Future[list[LineCodes]]] = deque()
    for band, first_line in _split_bands(blocks):
        pending.append(executor.submit(encode, band, first_line))
        if len(pending) > 2 * workers:
            yield from pending.popleft().result()
    while pending:
        yield from pending.popleft().result()


def _render_page(
    w: XL2HBWriter,
    width: int,
    height: int,
    pixel_data: PageData,
    settings: PrintSettings,
    channels: dict[str, DitherChannel],
    *,
    back_side: bool = False,
) -> None:
    """Render one page within an already-open session.

    Handles: BeginPage -> scanline loop -> flush -> EndPage. `back_side`
    marks the second page of a duplex sheet. Scanlines are read as views
    into `pixel_data`, which may also be a stream of row blocks; only
    long-edge back pages are collected into one buffer to be mirrored.

    Raises:
        ValueError: If `pixel_data` has fewer than `height` rows.
    """
    is_fine = settings.resolution == Resolution.FINE
    page_size = settings.page_size
    _, paper_h = PAPER_SIZES[page_size]

    if is_fine:
        sw, sh = get_image_dimensions_fine(page_size)
        bpl = (sw + 1) // 2  # 4bpp: 2 pixels per byte
    else:
        sw, sh = get_image_dimensions(page_size)
        bpl = (sw + 7) // 8  # 1bpp: 8 pixels per byte

    # Long-edge back pages are marked and sent mirrored top-to-bottom.
    flip_back = back_side and settings.duplex == DuplexMode.NO_TUMBLE
    if flip_back:
        blocks: RowBlocks = (_flip_vertical(collect_rows(pixel_data, width, height), paper_h),)
        height = paper_h
    elif isinstance(pixel_data, _BUFFER_TYPES):
        blocks = (_page_rows(pixel_data, width, height),)
    else:
        blocks = pixel_data

    w.write_begin_page(
        media_size=page_size,
        media_source=MEDIA_SOURCE[settings.input_slot],
        media_type=settings.media_type,
        duplex_mode=DUPLEX_MAP.get(settings.duplex),
        back_side_marker=flip_back,
    )
    w.write_set_page_origin()
    mono = settings.mono_color == MonoColor.MONO
    w.write_begin_image(sw, sh, copies=settings.copies, fine=is_fine, color=not mono)

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
    table = color_table(settings)
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

    colour = _ColourSetup(settings, table, input_remap)
    kernel_colour = None if mono else _band_kernel_colour(colour.table, channels, is_fine=is_fine)
    if mono:
        lines = _encode_lines_mono(blocks, width, sw, colour, channels, is_fine=is_fine)
    elif kernel_colour is not None:
        lines = _encode_lines_banded(blocks, width, sw, colour, channels, kernel_colour)
    else:
        lines = _encode_lines(blocks, width, sw, colour, channels, is_fine=is_fine)

    for line_idx in range(paper_h):
        if line_idx < height:
            line = next(lines, None)
            if line is None:
                msg = f"page ended after {line_idx} rows, expected {height}"
                raise ValueError(msg)
            plane_comp = line
        else:
            plane_comp = _BLANK_LINE

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
    w.write_end_page(copies=settings.copies)


def _init_channels(settings: PrintSettings) -> dict[str, DitherChannel]:
    """Initialize dither channels for the current print settings.

    The BRCD tables are read from the `lut/` directory next to this module
    (`src/lut/` in the repository, the lib directory when installed).
    `load_dither_tables` handles the fine→normal fallback and the Bayer
    fallback when no BRCD set matches.

    Returns:
        Channel dict keyed by 'K', 'C', 'M', 'Y'.
    """
    return load_dither_tables(
        str(Path(__file__).resolve().parent / "lut"),
        fine=settings.resolution == Resolution.FINE,
        toner_save=settings.toner_save,
    )


def filter_page(
    width: int,
    height: int,
    pixel_data: PageBuffer,
    settings: PrintSettings,
    output: BinaryIO,
) -> None:
    """Convert PPM pixel data to XL2HB and write to output."""
    if settings.skip_blank and is_blank_page(pixel_data):
        return

    filter_pages([(width, height, pixel_data)], settings, output, page_count=1)

    _, paper_h = PAPER_SIZES[settings.page_size]
    logger.debug("Processed %d lines, %dx%d input", paper_h, width, height)


def _render_pages_reversed(
    pages: Iterable[tuple[int, int, PageData]],
    settings: PrintSettings,
    channels: dict[str, DitherChannel],
    output: BinaryIO,
    page_count: int | None,
) -> None:
    """Render pages in input order and write them to `output` last page first.

    Each rendered page (BeginPage..EndPage) goes to a temporary file, so
    only one raster page is in memory at a time. Page `i` of `n` ends up at
    position `n - 1 - i`, which decides its duplex side exactly as if the
    rasters had been reversed before rendering.
    """
    duplex = settings.duplex != DuplexMode.NONE
    spans: list[tuple[int, int]] = []
    with tempfile.TemporaryFile(prefix="brhl4150cdn-reverse-") as spool:
        for index, (width, height, pixel_data) in enumerate(pages):
            # Without page_count the side is unknown; that is only allowed
            # when it does not change the output (simplex, short-edge duplex).
            back_side = duplex and page_count is not None and (page_count - 1 - index) % 2 == 1
            start = spool.tell()
            _render_page(
                XL2HBWriter(spool),
                width,
                height,
                pixel_data,
                settings,
                channels,
                back_side=back_side,
            )
            spans.append((start, spool.tell() - start))

        if page_count is not None and len(spans) != page_count:
            logger.error(
                "Expected %d pages but rendered %d; duplex sides of the reversed job may be wrong",
                page_count,
                len(spans),
            )

        for start, length in reversed(spans):
            spool.seek(start)
            output.write(spool.read(length))


def filter_pages(
    pages: Iterable[tuple[int, int, PageData]],
    settings: PrintSettings,
    output: BinaryIO,
    page_count: int | None = None,
) -> None:
    """Render a job's pages inside a single XL2HB session.

    Pages are consumed lazily; with duplex, every second page is a back side.
    With `settings.reverse` the pages are still rendered one at a time in
    input order, spooled to a temporary file and emitted last page first,
    so memory use does not grow with the job.

    Args:
        pages: iterable of (width, height, pixel_data) tuples
        settings: PrintSettings of the job
        output: writable binary stream
        page_count: number of pages in `pages`. Required for reverse order
            with long-edge duplex, where a page's position in the reversed
            job decides whether it is a mirrored back side.

    Raises:
        ValueError: If reverse long-edge duplex is requested without `page_count`.
    """
    if settings.reverse and settings.duplex == DuplexMode.NO_TUMBLE and page_count is None:
        msg = "page_count is required for reverse order with long-edge duplex"
        raise ValueError(msg)
    # PJL header always reports 600 dpi; Fine mode differs only in dithering.
    color = settings.mono_color != MonoColor.MONO
    pjl = generate_pjl_header(
        resolution=600,
        color=color,
        # Only BRMonoColor=Auto sets the colour-adapt flag (printer_config_init).
        color_adapt=settings.mono_color == MonoColor.AUTO,
        # ECONOMODE is always OFF; toner-save is implemented through the
        # -TS_cache09.bin dither tables instead.
        economode=False,
        less_paper_curl=settings.improve_output == ImproveOutput.LESS_PAPER_CURL,
        fix_intensity=settings.improve_output == ImproveOutput.FIX_INTENSITY,
        apt_mode=(settings.resolution == Resolution.FINE),
        improve_gray=settings.improve_gray,
        ucrgcr=settings.enhance_black,
    )
    output.write(pjl)

    # XL2HB stream — one session wrapping all pages.
    w = XL2HBWriter(output)
    w.write_stream_header()
    w.write_begin_session()
    w.write_open_data_source()

    channels = _init_channels(settings)

    duplex = settings.duplex != DuplexMode.NONE
    if settings.reverse:
        _render_pages_reversed(pages, settings, channels, output, page_count)
    else:
        for index, (width, height, pixel_data) in enumerate(pages):
            _render_page(w, width, height, pixel_data, settings, channels, back_side=duplex and index % 2 == 1)

    w.write_close_data_source()
    w.write_end_session()

    output.write(generate_pjl_footer())

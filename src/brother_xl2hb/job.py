"""A print job: PJL, one XL2HB session, page order, duplex sides, reverse and skip blank.

`filter_pages` renders a whole job (simplex or duplex) inside one XL2HB
session, as the manufacturer's filter does; `filter_page` is the
single-page entry point. Both drive the HL-4150CDN unless given another
`PrinterModel`.
"""

import contextlib
import io
import logging
from collections.abc import Iterable, Iterator
from typing import BinaryIO

from brother_xl2hb.color.dither import (
    DitherChannel,
)
from brother_xl2hb.models import DEFAULT_MODEL
from brother_xl2hb.models.base import PrinterModel
from brother_xl2hb.protocol.xl2hb import XL2HBWriter
from brother_xl2hb.render.page import init_channels, render_page
from brother_xl2hb.render.rows import PageBuffer, PageData, collect_rows, is_blank_page
from brother_xl2hb.settings import DuplexMode, PrintSettings

logger = logging.getLogger(__name__)


def chain_peeked(
    first: tuple[int, int, PageData], rest: Iterator[tuple[int, int, PageData]]
) -> Iterator[tuple[int, int, PageData]]:
    """Yield a page taken off a job with `next()`, then the rest of the job.

    Unlike `itertools.chain([first], rest)`, which keeps its arguments and
    so the first page for the whole job, this lets go of `first` once it
    has been handed out. That matters for whole-page buffers (~100 MB for
    A4).

    Yields:
        `first`, then the pages of `rest`.
    """
    yield first
    del first
    yield from rest


def drop_blank_pages(pages: Iterable[tuple[int, int, PageData]]) -> Iterator[tuple[int, int, PageBuffer]]:
    """Skip the pure-white pages of a job (BRSkipBlank).

    Brother's filter drops them before sides are assigned, in simplex and
    duplex alike: the next page takes the blank one's place (verified
    against captures). Each page is collected into one buffer to be tested.

    Yields:
        The pages that are not blank, as whole-page buffers.
    """
    for width, height, data in pages:
        rows = collect_rows(data, width, height)
        # `rows` may be a view of `data`; drop both, blank or not, before
        # the next page is collected, so that only one whole page is held.
        del data
        if is_blank_page(rows):
            logger.info("skipping blank page")
        else:
            yield width, height, rows
        del rows


def filter_page(
    width: int,
    height: int,
    pixel_data: PageBuffer,
    settings: PrintSettings,
    output: BinaryIO,
    model: PrinterModel = DEFAULT_MODEL,
) -> None:
    """Convert PPM pixel data to XL2HB and write to output."""
    if settings.skip_blank and is_blank_page(pixel_data):
        return

    filter_pages([(width, height, pixel_data)], settings, output, model)

    _, paper_h = model.paper_sizes[settings.page_size]
    logger.debug("Processed %d lines, %dx%d input", paper_h, width, height)


def _render_pages(
    pages: Iterable[tuple[int, int, PageData]],
    settings: PrintSettings,
    channels: dict[str, DitherChannel],
    output: BinaryIO,
    model: PrinterModel,
) -> None:
    """Render pages in input order; with duplex every second page is a back side.

    Each page is rendered into memory (its compressed XL2HB, typically
    well under 10 MB) and written to `output` only once it is complete, so
    a failure while reading or rendering a page never leaves a half page
    in the printer data stream.
    """
    duplex = settings.duplex != DuplexMode.NONE
    page_buf = io.BytesIO()
    # A counter instead of enumerate(): enumerate keeps the last (index,
    # page) pair until the next page has been read, which would hold two
    # whole-page buffers at once.
    index = 0
    for width, height, pixel_data in pages:
        page_buf.seek(0)
        page_buf.truncate()
        render_page(
            XL2HBWriter(page_buf),
            width,
            height,
            pixel_data,
            settings,
            channels,
            model,
            flip=model.page_is_flipped(settings, duplex and index % 2 == 1),
        )
        # Release the page's raster before the next page is read; with
        # whole-page buffers (skip blank, flipped pages) two would add up.
        del pixel_data
        with page_buf.getbuffer() as page:
            output.write(page)
        index += 1  # noqa: SIM113 - see above, enumerate() would hold the previous page


def filter_pages(
    pages: Iterable[tuple[int, int, PageData]],
    settings: PrintSettings,
    output: BinaryIO,
    model: PrinterModel = DEFAULT_MODEL,
) -> None:
    """Render a job's pages inside a single XL2HB session.

    Pages are consumed lazily in input order; with duplex, every second
    page is a back side. Which pages go bottom-up with the back-side
    marker (long-edge back sides, `settings.reverse`) is the model's
    `page_is_flipped`.
    `settings.skip_blank` drops pure-white pages before sides are assigned.

    If reading or rendering a page fails, the exception propagates after
    the session has been closed and the PJL footer written. `output` then
    holds only the complete pages rendered before the failure.

    Args:
        pages: iterable of (width, height, pixel_data) tuples
        settings: PrintSettings of the job
        output: writable binary stream
        model: the printer to drive
    """
    settings = model.restrict(settings)
    if settings.skip_blank:
        # Idempotent: callers that drop blanks first to detect an empty job
        # hand over whole-page buffers, which are only viewed again here.
        pages = drop_blank_pages(pages)
    output.write(model.pjl_header(settings))

    # XL2HB stream — one session wrapping all pages.
    w = XL2HBWriter(output)
    w.write_stream_header()
    w.write_begin_session()
    w.write_open_data_source()

    try:
        channels = init_channels(settings, model)
        _render_pages(pages, settings, channels, output, model)
    except Exception:
        # Only whole pages have reached `output` (see _render_pages), so
        # the session can still be closed: the printer prints the pages it
        # got and ends the job instead of waiting for the rest of a page.
        # If `output` itself failed there is nothing more to do.
        with contextlib.suppress(OSError):
            _end_session(w, output, model)
        raise
    _end_session(w, output, model)


def _end_session(w: XL2HBWriter, output: BinaryIO, model: PrinterModel) -> None:
    """Close the data source and the session, then write the PJL footer."""
    w.write_close_data_source()
    w.write_end_session()
    output.write(model.pjl_footer())

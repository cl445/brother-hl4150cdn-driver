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

from brother_xl2hb.models import DEFAULT_MODEL
from brother_xl2hb.models.base import PrinterModel
from brother_xl2hb.protocol.xl2hb import XL2HBWriter
from brother_xl2hb.render.page import JobContext, render_page
from brother_xl2hb.render.rows import Page, PageBuffer, PageData, collect_rows, is_blank_page
from brother_xl2hb.settings import PrintSettings

logger = logging.getLogger(__name__)


def chain_peeked(first: Page, rest: Iterator[Page]) -> Iterator[Page]:
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


def drop_blank_pages(pages: Iterable[Page]) -> Iterator[tuple[int, int, PageBuffer]]:
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
    filter_pages([(width, height, pixel_data)], settings, output, model)

    _, paper_h = model.paper_sizes[settings.page_size]
    logger.debug("Processed %d lines, %dx%d input", paper_h, width, height)


def _render_pages(pages: Iterable[Page], job: JobContext, output: BinaryIO) -> None:
    """Render pages in input order, each as the model's `page_sends` say.

    A page sent more than once is collected into one buffer first, since
    streamed row blocks can be read only once. Each send is rendered into memory (its compressed XL2HB, typically
    well under 10 MB) and written to `output` only once it is complete, so
    a failure while reading or rendering a page never leaves a half page
    in the printer data stream.
    """
    page_buf = io.BytesIO()
    # A counter instead of enumerate(): enumerate keeps the last (index,
    # page) pair until the next page has been read, which would hold two
    # whole-page buffers at once.
    index = 0
    for width, height, pixel_data in pages:
        sends = job.model.page_sends(job.settings, index)
        data: PageData = collect_rows(pixel_data, width, height) if len(sends) > 1 else pixel_data
        for send in sends:
            page_buf.seek(0)
            page_buf.truncate()
            render_page(XL2HBWriter(page_buf), (width, height, data), job, send)
            with page_buf.getbuffer() as page:
                output.write(page)
        # Release the page's raster before the next page is read; with
        # whole-page buffers (skip blank, flipped pages) two would add up.
        del pixel_data, data
        index += 1  # noqa: SIM113 - see above, enumerate() would hold the previous page


def filter_pages(
    pages: Iterable[Page],
    settings: PrintSettings,
    output: BinaryIO,
    model: PrinterModel = DEFAULT_MODEL,
) -> None:
    """Render a job's pages inside a single XL2HB session.

    Pages are consumed lazily in input order. How each is sent (duplex
    sides, bottom-up for long-edge back sides and `settings.reverse`,
    copies) is the model's `page_sends`. `settings.skip_blank` drops
    pure-white pages before that; if that leaves no page, nothing is
    written at all.

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
        pages = drop_blank_pages(pages)
        first = next(pages, None)
        if first is None:
            logger.info("all pages blank, nothing to print")
            return
        pages = chain_peeked(first, pages)
        del first
    output.write(model.pjl_header(settings))

    # XL2HB stream — one session wrapping all pages.
    w = XL2HBWriter(output)
    w.write_stream_header()
    w.write_begin_session()
    w.write_open_data_source()

    try:
        _render_pages(pages, JobContext.create(settings, model), output)
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

"""Stream Ghostscript's PPM pages as blocks of cropped scanlines.

Ghostscript writes one P6 PPM per page to its stdout. Reading a whole
page (~100 MB at 600 dpi A4) before rendering it keeps a page-sized
buffer alive and serialises the two processes: gs waits on the full pipe
while Python renders, Python waits on gs while it rasterises.

`iter_ppm_pages` instead reads the stream on a background thread, a
block of scanlines at a time, cuts out the printable window and hands the
blocks over through a bounded queue. Blocking reads release the GIL, so gs
keeps rasterising while the main thread renders.
"""

import queue
import threading
from collections.abc import Iterator
from typing import BinaryIO

import numpy as np
import numpy.typing as npt

from ppm import read_ppm_header

_END_OF_PAGE = object()
_END_OF_STREAM = object()


class _ReaderFailure:
    """Carries an exception from the reader thread to the consumer."""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


def _get(q: "queue.Queue[object]") -> object:
    item = q.get()
    if isinstance(item, _ReaderFailure):
        raise item.exc
    return item


def _read_exact(stream: BinaryIO, buf: npt.NDArray[np.uint8]) -> None:
    """Fill `buf` from `stream`.

    Raises:
        ValueError: If the stream ends before `buf` is full.
    """
    view = memoryview(buf).cast("B")
    filled = 0
    while filled < len(view):
        n = stream.readinto(view[filled:])  # type: ignore[attr-defined]
        if not n:
            msg = f"PPM payload incomplete: got {filled} of {len(view)} bytes in block"
            raise ValueError(msg)
        filled += n


def _cropped_blocks(
    stream: BinaryIO,
    src_w: int,
    src_h: int,
    window: tuple[int, int, int, int],
    block_rows: int,
) -> Iterator[npt.NDArray[np.uint8]]:
    """Read one page's pixel data and yield the printable window in row blocks.

    `window` is (x_off, y_off, width, height). Rows and columns the render
    does not cover are white, so exactly `height` rows of `width * 3` bytes
    are yielded, matching a white-padded crop of the full page.
    """
    x_off, y_off, width, height = window
    crop_w = min(width, src_w - x_off)
    crop_h = min(height, src_h - y_off)
    row_bytes = width * 3
    emitted = 0

    for first in range(0, src_h, block_rows):
        rows = min(block_rows, src_h - first)
        buf = np.empty((rows, src_w * 3), dtype=np.uint8)
        _read_exact(stream, buf)
        lo = max(first, y_off)
        hi = min(first + rows, y_off + crop_h)
        if crop_w <= 0 or lo >= hi:
            continue
        part = buf[lo - first : hi - first, x_off * 3 : (x_off + crop_w) * 3]
        if crop_w < width:
            padded = np.full((hi - lo, row_bytes), 255, dtype=np.uint8)
            padded[:, : crop_w * 3] = part
            part = padded
        yield part
        emitted += hi - lo

    for first in range(emitted, height, block_rows):
        yield np.full((min(block_rows, height - first), row_bytes), 255, dtype=np.uint8)


def _read_pages(
    stream: BinaryIO,
    q: "queue.Queue[object]",
    window: tuple[int, int, int, int],
    block_rows: int,
) -> None:
    """Reader thread: push (src_w, src_h), row blocks and end markers onto `q`."""
    try:
        while (header := read_ppm_header(stream)) is not None:
            src_w, src_h, _maxval = header
            q.put((src_w, src_h))
            for block in _cropped_blocks(stream, src_w, src_h, window, block_rows):
                q.put(block)
            q.put(_END_OF_PAGE)
        q.put(_END_OF_STREAM)
    except BaseException as exc:  # noqa: BLE001 — re-raised in the consumer
        q.put(_ReaderFailure(exc))


class PageBlocks:
    """Row blocks of one streamed page; iterate once, in order.

    Each block is a (rows, width * 3) uint8 array whose rows are contiguous
    RGB scanlines (the block itself may be a strided view).
    """

    def __init__(self, q: "queue.Queue[object]") -> None:
        """Wrap the shared queue; the page ends at the next end-of-page marker."""
        self._q = q
        self._done = False

    def __iter__(self) -> Iterator[npt.NDArray[np.uint8]]:
        """Yield the page's remaining row blocks.

        Yields:
            (rows, width * 3) uint8 blocks.
        """
        while not self._done:
            item = _get(self._q)
            if item is _END_OF_PAGE:
                self._done = True
                return
            yield item  # type: ignore[misc]

    def drain(self) -> None:
        """Discard whatever the consumer did not read, up to the end of the page."""
        for _ in self:
            pass


def iter_ppm_pages(
    stream: BinaryIO,
    window: tuple[int, int, int, int],
    *,
    block_rows: int = 128,
    queue_blocks: int = 16,
) -> Iterator[tuple[int, int, PageBlocks]]:
    """Yield (src_width, src_height, blocks) for each PPM page in `stream`.

    `window` is (x_off, y_off, width, height) of the printable area; every
    page yields exactly `height` rows of `width` pixels. At most
    `queue_blocks` blocks are buffered ahead of the consumer. A page's
    unread blocks are discarded when the next page is requested.

    Yields:
        Source dimensions from the PPM header and the page's row blocks.
    """
    q: queue.Queue[object] = queue.Queue(maxsize=queue_blocks)
    reader = threading.Thread(target=_read_pages, args=(stream, q, window, block_rows), daemon=True)
    reader.start()
    while (item := _get(q)) is not _END_OF_STREAM:
        src_w, src_h = item  # type: ignore[misc]
        page = PageBlocks(q)
        yield src_w, src_h, page
        page.drain()

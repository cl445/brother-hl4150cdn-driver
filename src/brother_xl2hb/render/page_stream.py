"""Stream Ghostscript's PPM pages as blocks of cropped scanlines.

Ghostscript writes one P6 PPM per page to its stdout. Reading a whole
page (~100 MB at 600 dpi A4) before rendering it keeps a page-sized
buffer alive and serialises the two processes: gs waits on the full pipe
while Python renders, Python waits on gs while it rasterises.

`iter_ppm_pages` instead reads the stream on a background thread, a
block of scanlines at a time, cuts out the printable window and hands the
blocks over through a bounded queue. Blocking reads release the GIL, so gs
keeps rasterising while the main thread renders.

Closing the page iterator stops the reader thread, so a consumer that
gives up early does not leave it blocked on a full queue holding the
buffered blocks.
"""

import contextlib
import queue
import threading
from collections.abc import Iterator
from typing import BinaryIO, Self

import numpy as np
import numpy.typing as npt

from brother_xl2hb.render.ppm import read_ppm_header

_END_OF_PAGE = object()
_END_OF_STREAM = object()

# How often a reader blocked on a full queue checks whether it was stopped.
_PUT_POLL = 0.1


class _ReaderStoppedError(Exception):
    """The consumer closed the page iterator; ends the reader thread."""


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


def _put(q: "queue.Queue[object]", item: object, stop: threading.Event) -> None:
    """Put `item` on `q`, waiting for room until `stop` is set.

    Raises:
        _ReaderStoppedError: If `stop` is set.
    """
    while not stop.is_set():
        try:
            q.put(item, timeout=_PUT_POLL)
        except queue.Full:
            continue
        return
    raise _ReaderStoppedError


def _read_pages(
    stream: BinaryIO,
    q: "queue.Queue[object]",
    window: tuple[int, int, int, int],
    block_rows: int,
    stop: threading.Event,
) -> None:
    """Reader thread: push (src_w, src_h), row blocks and end markers onto `q`."""
    try:
        while (header := read_ppm_header(stream)) is not None:
            src_w, src_h, _maxval = header
            _put(q, (src_w, src_h), stop)
            for block in _cropped_blocks(stream, src_w, src_h, window, block_rows):
                _put(q, block, stop)
            _put(q, _END_OF_PAGE, stop)
        _put(q, _END_OF_STREAM, stop)
    except _ReaderStoppedError:
        pass
    except BaseException as exc:  # noqa: BLE001 — re-raised in the consumer
        with contextlib.suppress(_ReaderStoppedError):
            _put(q, _ReaderFailure(exc), stop)


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


class PageStream:
    """Iterator of (src_width, src_height, blocks), one per PPM page.

    See `iter_ppm_pages`. `close()` (or leaving a `with` block) stops the
    reader thread and drops the blocks it buffered.
    """

    def __init__(
        self,
        stream: BinaryIO,
        window: tuple[int, int, int, int],
        block_rows: int,
        queue_blocks: int,
    ) -> None:
        """Start the reader thread on `stream`."""
        self._q: queue.Queue[object] = queue.Queue(maxsize=queue_blocks)
        self._stop = threading.Event()
        self._page: PageBlocks | None = None
        self._ended = False
        reader = threading.Thread(
            target=_read_pages, args=(stream, self._q, window, block_rows, self._stop), name="ppm-reader", daemon=True
        )
        reader.start()

    def __iter__(self) -> Self:
        """Return the stream itself; it is iterated once.

        Returns:
            This stream.
        """
        return self

    def __next__(self) -> tuple[int, int, PageBlocks]:
        """Discard what is left of the previous page and start the next one.

        Returns:
            Source dimensions from the PPM header and the page's row blocks.

        Raises:
            StopIteration: At the end of the stream, or once closed.
        """
        if self._ended:
            raise StopIteration
        if self._page is not None:
            self._page.drain()
            self._page = None
        try:
            item = _get(self._q)
        except BaseException:
            self.close()
            raise
        if item is _END_OF_STREAM:
            self.close()
            raise StopIteration
        src_w, src_h = item  # type: ignore[misc]
        self._page = PageBlocks(self._q)
        return src_w, src_h, self._page

    def close(self) -> None:
        """Stop the reader thread and release the buffered blocks.

        The thread finishes the read it may be blocked in first, which
        ends at the latest when the writer closes the stream.
        """
        self._ended = True
        self._page = None
        self._stop.set()
        with contextlib.suppress(queue.Empty):
            while True:
                self._q.get_nowait()

    def __enter__(self) -> Self:
        """Use the stream in a `with` block that closes it.

        Returns:
            This stream.
        """
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the stream."""
        self.close()


def iter_ppm_pages(
    stream: BinaryIO,
    window: tuple[int, int, int, int],
    *,
    block_rows: int = 128,
    queue_blocks: int = 16,
) -> PageStream:
    """Iterate (src_width, src_height, blocks) for each PPM page in `stream`.

    `window` is (x_off, y_off, width, height) of the printable area; every
    page yields exactly `height` rows of `width` pixels. At most
    `queue_blocks` blocks are buffered ahead of the consumer. A page's
    unread blocks are discarded when the next page is requested.

    A consumer that stops early closes the returned stream, so that the
    reader thread ends instead of blocking on a full queue that holds the
    buffered blocks.

    Returns:
        The page iterator, also a context manager that closes it.
    """
    return PageStream(stream, window, block_rows, queue_blocks)

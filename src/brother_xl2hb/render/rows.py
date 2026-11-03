"""RGB page input: row views, whole-page collection, line numbering and the blank test."""

from collections.abc import Buffer, Iterable, Iterator

import numpy as np
import numpy.typing as npt

PageBuffer = bytes | npt.NDArray[np.uint8]
"""A whole RGB page as packed bytes or as a (height, width, 3) uint8 array.

The array may be a strided view (e.g. the printable window of a larger
render) as long as each row is contiguous.
"""


RowBlocks = Iterable[npt.NDArray[np.uint8]]
"""A page streamed as (rows, width * 3) uint8 blocks with contiguous rows.

Consumed once, top to bottom (see `page_stream.iter_ppm_pages`).
"""


LineBlocks = Iterable[tuple[npt.NDArray[np.uint8], int]]
"""Row blocks, each with the page line of its first row.

What the scanline encoders take: the line selects the dither rows. A page
sent top to bottom numbers its blocks from 0 (`numbered_blocks`); a
flipped one hands over each block mirrored, with the line its last row
ends up on.
"""


PageData = PageBuffer | RowBlocks

Page = tuple[int, int, PageData]
"""(width, height, pixel data) of one page of a job."""


def page_rows(pixel_data: Buffer, width: int, height: int) -> npt.NDArray[np.uint8]:
    """View an RGB page as (height, width * 3) rows without copying.

    Returns:
        uint8 array whose rows are contiguous RGB scanlines.
    """
    if isinstance(pixel_data, np.ndarray):
        return pixel_data.reshape(height, width * 3)
    return np.frombuffer(pixel_data, dtype=np.uint8, count=width * height * 3).reshape(height, width * 3)


BUFFER_TYPES = (bytes, bytearray, memoryview, np.ndarray)


def collect_rows(pixel_data: PageData, width: int, height: int) -> npt.NDArray[np.uint8]:
    """Return the page as one (height, width * 3) array.

    A view for packed bytes and arrays; streamed row blocks are copied into
    a single page-sized buffer. Only needed where the whole page must be
    seen at once: a page sent more than once in different ways, and
    skip-blank with collated copies.

    Returns:
        uint8 array whose rows are contiguous RGB scanlines.

    Raises:
        ValueError: If streamed blocks do not add up to `height` rows.
    """
    if isinstance(pixel_data, BUFFER_TYPES):
        return page_rows(pixel_data, width, height)
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


def numbered_blocks(blocks: RowBlocks) -> Iterator[tuple[npt.NDArray[np.uint8], int]]:
    """Number row blocks top to bottom, from page line 0.

    Yields:
        (block, page line of its first row).
    """
    line = 0
    for block in blocks:
        yield block, line
        line += block.shape[0]


class BlankTracker:
    """Row blocks passed through unchanged, noting whether all of them are pure white.

    `is_blank_page` for a page that is streamed rather than collected: the
    page renders while it is read, and `finish` tells afterwards whether it
    was blank. The test stops at the first block with ink.
    """

    def __init__(self, blocks: RowBlocks) -> None:
        """Wrap `blocks`; `blank` stays True until a block has a byte below 255."""
        self._blocks = iter(blocks)
        self.blank = True

    def __iter__(self) -> "BlankTracker":
        """Return the tracker itself; it is iterated once.

        Returns:
            This tracker.
        """
        return self

    def __next__(self) -> npt.NDArray[np.uint8]:
        """Return the next block, testing it while the page still looks blank.

        Returns:
            The next row block.
        """
        block = next(self._blocks)
        if self.blank and block.size and int(block.min()) != 255:
            self.blank = False
        return block

    def finish(self) -> bool:
        """Read the blocks the consumer left over while the page still looks blank.

        Returns:
            True if every byte of the page is 255.
        """
        while self.blank and next(self, None) is not None:
            pass
        return self.blank


def white_rows(block: npt.NDArray[np.uint8]) -> npt.NDArray[np.bool_]:
    """Return which rows of `block` are pure white (all bytes 255).

    White rows carry no ink whatever the settings: apply_input_remap_rgb
    preserves (255,255,255), saturation and vivid leave the gray axis
    untouched, and every colour table maps white to zero ink.
    """
    if not block.shape[1]:
        return np.ones(block.shape[0], dtype=np.bool_)
    return block.min(axis=1) == 255


def is_blank_page(pixel_data: PageBuffer) -> bool:
    """Return True if every byte of the page is 255 (pure white).

    Works on packed bytes and on strided array views without copying.
    """
    arr = pixel_data if isinstance(pixel_data, np.ndarray) else np.frombuffer(pixel_data, dtype=np.uint8)
    return arr.size == 0 or int(arr.min()) == 255

"""RGB page input: row views, whole-page collection, blank test and the bottom-up flip."""

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


PageData = PageBuffer | RowBlocks


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
    seen at once (long-edge back pages, skip-blank).

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


def white_rows(block: npt.NDArray[np.uint8]) -> npt.NDArray[np.bool_]:
    """Return which rows of `block` are pure white (all bytes 255).

    White rows carry no ink whatever the settings: apply_input_remap_rgb
    preserves (255,255,255), saturation and vivid leave the gray axis
    untouched, and every colour table maps white to zero ink.
    """
    if not block.shape[1]:
        return np.ones(block.shape[0], dtype=np.bool_)
    return block.min(axis=1) == 255


def iter_rows(blocks: RowBlocks) -> Iterator[tuple[npt.NDArray[np.uint8], bool]]:
    """Yield (scanline, is_pure_white) for every row of every block.

    The white test runs vectorised per block.

    Yields:
        Contiguous RGB scanline view and whether all its bytes are 255.
    """
    for block in blocks:
        yield from zip(block, white_rows(block).tolist(), strict=True)


def is_blank_page(pixel_data: PageBuffer) -> bool:
    """Return True if every byte of the page is 255 (pure white).

    Works on packed bytes and on strided array views without copying.
    """
    arr = pixel_data if isinstance(pixel_data, np.ndarray) else np.frombuffer(pixel_data, dtype=np.uint8)
    return arr.size == 0 or int(arr.min()) == 255


def flip_vertical(rows: npt.NDArray[np.uint8], paper_h: int) -> npt.NDArray[np.uint8]:
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

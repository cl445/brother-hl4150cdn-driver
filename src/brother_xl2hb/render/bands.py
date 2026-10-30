"""Whole bands in the native kernels (`_band_fast`, `_fine_fast`), on the render threads."""

import logging
import os
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from functools import cache
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from brother_xl2hb import _native
from brother_xl2hb.color import color_lut
from brother_xl2hb.color.dither import (
    DitherChannel,
)
from brother_xl2hb.render.lines import ColourSetup, LineCodes, adjust_rgb, fine_overrun_rows
from brother_xl2hb.render.rows import RowBlocks, page_rows, white_rows

_band_fast = _native.load("_band_fast")
_fine_fast = _native.load("_fine_fast")

logger = logging.getLogger(__name__)


_EMPTY_TABLE = np.empty(0, dtype=np.uint8)


# Scanlines per band handed to the native kernel; also the unit of work
# for the render threads.
_BAND_ROWS = 128


class KernelColour(NamedTuple):
    """Colour arguments of `render_band`: the inverse LUT, or the grid to interpolate."""

    inverse: npt.NDArray[np.uint8]
    grid: npt.NDArray[np.int32]
    weights: npt.NDArray[np.uint8]
    black: npt.NDArray[np.int32]


_EMPTY_INT = np.empty(0, dtype=np.int32)


def band_kernel_colour(
    table: color_lut.ColorTable, channels: dict[str, DitherChannel], *, is_fine: bool
) -> KernelColour | None:
    """Return the colour arguments for `render_band`, or None if it cannot render the page.

    The kernels cover Normal mode with threshold-matrix dither channels
    (`render_band`) and Fine mode with BRCD nibble tables
    (`render_band_fine`); each needs its native module. Both gather through
    the installed inverse LUT of `table` when there is one and interpolate
    the grid otherwise. `render_band_fine` takes the same colour arguments.

    Returns:
        The kernel's colour arguments, or None for the per-line path.
    """
    if is_fine:
        if _fine_fast is None or any(channel.nibble_levels is None for channel in channels.values()):
            return None
    elif _band_fast is None or any(channel.threshold_matrix is None for channel in channels.values()):
        return None
    inverse = color_lut.inverse_lut(table)
    if inverse is not None:
        return KernelColour(inverse.reshape(-1), _EMPTY_INT, _EMPTY_TABLE, _EMPTY_INT)
    return KernelColour(_EMPTY_TABLE, *color_lut.interp_arrays(table))


# Environment variables for the thread count, in order: the current name and
# the one of releases up to 1.2.0.
_RENDER_THREADS_VARS = ("BRXL2HB_RENDER_THREADS", "BRHL4150CDN_RENDER_THREADS")


def render_threads() -> int:
    """Number of band render threads.

    `BRXL2HB_RENDER_THREADS` (formerly `BRHL4150CDN_RENDER_THREADS`, still
    read) overrides the default of one thread per spare core, at most
    three; 0 renders on the calling thread.

    Returns:
        Thread count, >= 0.
    """
    default = max(1, min(3, (os.cpu_count() or 1) - 1))
    name = next((var for var in _RENDER_THREADS_VARS if var in os.environ), None)
    if name is None:
        return default
    configured = os.environ[name]
    try:
        return max(0, int(configured))
    except ValueError:
        logger.warning("ignoring non-numeric %s=%r, using %d threads", name, configured, default)
        return default


@cache
def _band_executor(workers: int) -> ThreadPoolExecutor:
    """Return the process-wide pool of `workers` render threads."""
    return ThreadPoolExecutor(max_workers=workers, thread_name_prefix="brxl2hb-band")


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


def encode_lines_banded(
    blocks: RowBlocks,
    width: int,
    sw: int,
    colour: ColourSetup,
    channels: dict[str, DitherChannel],
    kernel_colour: KernelColour,
    *,
    is_fine: bool,
) -> Iterator[LineCodes]:
    """Like `encode_lines`, but whole bands at a time in the native kernel.

    The kernel releases the GIL, so bands render in parallel on the
    render threads; results are yielded strictly in page order.
    `kernel_colour` comes from `band_kernel_colour`.

    Yields:
        Encoded planes of each input row, top to bottom.

    Raises:
        RuntimeError: If the kernel for `is_fine` is not loaded.
    """
    native = _fine_fast if is_fine else _band_fast
    if native is None:
        msg = "encode_lines_banded needs the native kernel; check band_kernel_colour first"
        raise RuntimeError(msg)
    # Prepare the dither tables up front; the render threads only read them.
    if is_fine:
        tables = tuple(np.ascontiguousarray(channels[c].nibble_levels) for c in "KCMY")
    else:
        tables = tuple(channels[c].tiled_thresholds(sw) for c in "KCMY")

    def encode(band: npt.NDArray[np.uint8], first_line: int) -> list[LineCodes]:
        n = band.shape[0]
        skip = white_rows(band).view(np.uint8)
        rgb = band
        if colour.adjusts_rgb:
            rgb = page_rows(adjust_rgb(np.ascontiguousarray(band), n * width, colour), width, n)
        lengths = np.empty((n, 4), dtype=np.int32)
        if is_fine:
            overrun = fine_overrun_rows(band, width)
            data = native.render_band_fine(rgb, skip, overrun, width, first_line, sw, *kernel_colour, *tables, lengths)
        else:
            data = native.render_band(rgb, skip, width, first_line, sw, *kernel_colour, *tables, lengths)
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

    workers = render_threads()
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

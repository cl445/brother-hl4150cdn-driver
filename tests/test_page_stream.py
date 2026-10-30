"""page_stream: threaded PPM reader that yields the printable window in row blocks."""

import io
import threading

import numpy as np
import pytest

from brother_xl2hb.models import HL4150CDN
from brother_xl2hb.render.page_stream import iter_ppm_pages
from brother_xl2hb.render.rows import collect_rows

_TARGET_W, _TARGET_H = HL4150CDN.paper_sizes["A4"]
_WINDOW = (100, 100, _TARGET_W, _TARGET_H)


def _ppm(pixel_data: bytes, width: int, height: int) -> bytes:
    return b"P6\n%d %d\n255\n" % (width, height) + pixel_data


def _reference_crop(pixel_data, src_width, src_height, target_w, target_h):
    """The original per-row crop: printable window at (100, 100), white-padded."""
    crop_w = min(target_w, src_width - 100)
    crop_h = min(target_h, src_height - 100)
    if crop_w <= 0 or crop_h <= 0:
        return b"\xff" * (target_w * target_h * 3)
    rows = []
    for y in range(crop_h):
        start = (y + 100) * src_width * 3 + 100 * 3
        row = pixel_data[start : start + crop_w * 3]
        if crop_w < target_w:
            row += b"\xff" * ((target_w - crop_w) * 3)
        rows.append(row)
    rows.extend([b"\xff" * (target_w * 3)] * (target_h - len(rows)))
    return b"".join(rows)


def _random_page(width, height, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, width * height * 3, dtype=np.uint8).tobytes()


@pytest.mark.parametrize(
    ("src_width", "src_height"),
    [(4958, 7016), (4700, 7016), (4958, 6000), (4000, 5000), (90, 7016), (4958, 50)],
)
def test_page_matches_reference_crop(src_width, src_height):
    pixel_data = _random_page(src_width, src_height, src_width + src_height)
    pages = iter_ppm_pages(io.BytesIO(_ppm(pixel_data, src_width, src_height)), _WINDOW, block_rows=97)
    w, h, blocks = next(pages)
    assert (w, h) == (src_width, src_height)
    got = collect_rows(blocks, _TARGET_W, _TARGET_H)
    assert got.tobytes() == _reference_crop(pixel_data, src_width, src_height, _TARGET_W, _TARGET_H)
    assert next(pages, None) is None


def test_multiple_pages_and_skipped_page():
    """An unread page is drained, so the next page starts at the right byte."""
    sizes = [(4958, 7016), (4958, 7016), (4800, 7100)]
    datas = [_random_page(w, h, i) for i, (w, h) in enumerate(sizes)]
    stream = io.BytesIO(b"".join(_ppm(d, w, h) for d, (w, h) in zip(datas, sizes, strict=True)))

    seen = []
    for index, (w, h, blocks) in enumerate(iter_ppm_pages(stream, _WINDOW, block_rows=500, queue_blocks=2)):
        if index == 1:
            next(iter(blocks))  # read one block, leave the rest unread
            continue
        got = collect_rows(blocks, _TARGET_W, _TARGET_H).tobytes()
        seen.append((index, got == _reference_crop(datas[index], w, h, _TARGET_W, _TARGET_H)))
    assert seen == [(0, True), (2, True)]


def test_truncated_page_raises_in_consumer():
    data = _random_page(4958, 7016, 1)
    stream = io.BytesIO(_ppm(data, 4958, 7016)[: -4958 * 3 * 10])
    _w, _h, blocks = next(iter_ppm_pages(stream, _WINDOW))
    with pytest.raises(ValueError, match="incomplete"):
        collect_rows(blocks, _TARGET_W, _TARGET_H)


def test_collect_rows_rejects_blocks_short_of_the_page():
    blocks = iter([np.zeros((3, 8 * 3), np.uint8), np.zeros((2, 8 * 3), np.uint8)])
    with pytest.raises(ValueError, match="page has 5 rows, expected 8"):
        collect_rows(blocks, 8, 8)


def test_empty_stream_yields_nothing():
    assert list(iter_ppm_pages(io.BytesIO(b""), _WINDOW)) == []


def test_closing_the_pages_stops_the_reader_thread():
    """A consumer that gives up must not leave the reader blocked on a full queue."""
    width, height = 200, 300
    data = b"".join(_ppm(_random_page(width, height, i), width, height) for i in range(30))
    stream = io.BytesIO(data)
    before = set(threading.enumerate())
    pages = iter_ppm_pages(stream, (0, 0, width, height), block_rows=10, queue_blocks=2)
    _w, _h, blocks = next(pages)
    next(iter(blocks))
    (reader,) = set(threading.enumerate()) - before

    pages.close()
    reader.join(timeout=10)
    assert not reader.is_alive()
    assert stream.tell() < len(data)


def test_reader_failure_after_close_does_not_block():
    """A reader that fails while the consumer is gone exits instead of waiting for room."""
    width, height = 200, 300
    stream = io.BytesIO(_ppm(_random_page(width, height, 0), width, height)[:-10])
    before = set(threading.enumerate())
    pages = iter_ppm_pages(stream, (0, 0, width, height), block_rows=10, queue_blocks=1)
    next(pages)
    (reader,) = set(threading.enumerate()) - before
    pages.close()
    reader.join(timeout=10)
    assert not reader.is_alive()


def test_failure_while_draining_a_page_ends_the_stream():
    """A reader failure met while skipping the rest of a page closes the stream."""
    width, height = 200, 300
    stream = io.BytesIO(_ppm(_random_page(width, height, 0), width, height)[:-10])
    with iter_ppm_pages(stream, (0, 0, width, height), block_rows=10, queue_blocks=1) as pages:
        next(pages)
        with pytest.raises(ValueError, match="incomplete"):
            next(pages)
        with pytest.raises(StopIteration):
            next(pages)

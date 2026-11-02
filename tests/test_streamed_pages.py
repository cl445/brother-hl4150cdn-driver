"""Flipped and skip-blank pages render from the row stream, without a whole-page buffer."""

import io

import numpy as np
import pytest

from brother_xl2hb import job
from brother_xl2hb.job import filter_pages
from brother_xl2hb.models import HL4150CDN
from brother_xl2hb.render import bands
from brother_xl2hb.render.page import _flipped_lines, _line_encoder, init_channels
from brother_xl2hb.render.rows import BlankTracker, numbered_blocks
from brother_xl2hb.settings import DuplexMode, MonoColor, PrintSettings, Resolution
from fixture_utils import assert_matches_fixture
from test_full_pipeline import _A4_H, _A4_W, _as_row_blocks, _duplex_test_pages

_W, _PAPER_H = 600, 500


def _rows(height: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = np.full((height, _W * 3), 255, np.uint8)
    rows[20:140] = rng.integers(0, 256, (120, _W * 3), dtype=np.uint8)
    rows[200:260, : _W * 3 // 2] = 30
    rows[height - 5 :] = 90  # ink on the last rows: they end up at the top, or past the paper
    return rows


def _blocks(rows: np.ndarray, size: int):
    return (rows[i : i + size] for i in range(0, rows.shape[0], size))


_ENCODERS = {
    "normal": (PrintSettings(), True),
    "normal_per_line": (PrintSettings(), False),
    "mono": (PrintSettings(mono_color=MonoColor.MONO), True),
    "mono_per_line": (PrintSettings(mono_color=MonoColor.MONO), False),
    "fine": (PrintSettings(resolution=Resolution.FINE), True),
}


@pytest.mark.parametrize("name", list(_ENCODERS))
@pytest.mark.parametrize("height", [_PAPER_H - 130, _PAPER_H, _PAPER_H + 40])
def test_flipped_lines_match_encoding_the_mirrored_page(name, height, monkeypatch):
    """Encoding blocks mirrored as they stream equals encoding the mirrored, white-padded page."""
    settings, native = _ENCODERS[name]
    if not native:
        monkeypatch.setattr(bands, "_band_fast", None)
        monkeypatch.setattr(bands, "_fine_fast", None)
    monkeypatch.setenv("BRXL2HB_RENDER_THREADS", "2")
    channels = init_channels(settings, HL4150CDN)
    encode = _line_encoder(settings, HL4150CDN, channels, _W, _W)
    rows = _rows(height, seed=height)

    mirrored = np.full((_PAPER_H, _W * 3), 255, np.uint8)
    kept = min(height, _PAPER_H)
    mirrored[:kept] = rows[:kept]
    expected = list(encode(numbered_blocks([mirrored[::-1]])))

    got = list(_flipped_lines(encode, _blocks(rows, 97), height, _PAPER_H))
    assert [tuple(map(bytes, line)) for line in got] == [tuple(map(bytes, line)) for line in expected]


def test_flipped_lines_report_short_pages():
    settings = PrintSettings()
    encode = _line_encoder(settings, HL4150CDN, init_channels(settings, HL4150CDN), _W, _W)
    with pytest.raises(ValueError, match="page has 300 rows, expected 400"):
        list(_flipped_lines(encode, _blocks(_rows(300, seed=1), 128), 400, _PAPER_H))


def test_streamed_flipped_and_skip_blank_pages_are_not_collected(monkeypatch):
    """Long-edge back sides and skip blank render streamed pages without a whole-page buffer."""
    pages = _duplex_test_pages()[:2]
    settings = PrintSettings(duplex=DuplexMode.NO_TUMBLE, skip_blank=True)
    expected = io.BytesIO()
    filter_pages(pages, settings, expected)

    def no_collect(*args):
        msg = "page collected"
        raise AssertionError(msg)

    monkeypatch.setattr(job, "collect_rows", no_collect)
    out = io.BytesIO()
    filter_pages([(w, h, _as_row_blocks(data, w, h, 128)) for w, h, data in pages], settings, out)
    assert out.getvalue() == expected.getvalue()


@pytest.mark.parametrize("block_rows", [128, 1000])
def test_streamed_skip_blank_in_duplex_matches_brother_capture(block_rows):
    """Streamed pages are tested for blank while they render; the next page takes a blank page's side."""
    a, b, c, _ = _duplex_test_pages()
    white = (_A4_W, _A4_H, b"\xff" * (_A4_W * _A4_H * 3))
    pages = [(w, h, _as_row_blocks(data, w, h, block_rows)) for w, h, data in (a, white, b, c)]
    out = io.BytesIO()
    filter_pages(pages, PrintSettings(duplex=DuplexMode.NO_TUMBLE, skip_blank=True), out)
    assert_matches_fixture("skip_blank_long_edge", out.getvalue())


def _white_blocks(height: int, *, ink_row: int | None = None):
    band = np.full((128, _A4_W * 3), 255, np.uint8)
    for lo in range(0, height, 128):
        block = band[: min(128, height - lo)]
        if ink_row is not None and lo <= ink_row < lo + block.shape[0]:
            block = block.copy()
            block[ink_row - lo, :30] = 0
        yield block


def test_streamed_blank_pages_only_write_nothing():
    out = io.BytesIO()
    pages = [(_A4_W, _A4_H, _white_blocks(_A4_H)) for _ in range(3)]
    filter_pages(pages, PrintSettings(skip_blank=True), out)
    assert out.getvalue() == b""


def test_ink_past_the_paper_keeps_a_streamed_page():
    """Like is_blank_page, every source row counts, also those the render never reads."""
    height = _A4_H + 20
    out = io.BytesIO()
    filter_pages([(_A4_W, height, _white_blocks(height, ink_row=height - 3))], PrintSettings(skip_blank=True), out)
    assert out.getvalue()  # an all-white page writes nothing


def test_blank_tracker():
    blocks = [np.full((2, 6), 255, np.uint8), np.full((2, 6), 255, np.uint8), np.zeros((2, 6), np.uint8)]
    tracker = BlankTracker(blocks)
    assert next(tracker) is blocks[0]
    assert tracker.blank
    assert not tracker.finish()  # reads on until the block with ink
    assert BlankTracker(blocks[:2]).finish()
    assert BlankTracker([]).finish()

"""Copies and Collate from Brother's rc file: byte-exact against brhl4150cdnfilter.

Uncollated, the original sends each page `Copies` times in a row, every
send with PageCopies = Copies, and counts duplex sides per send. With
Collate=ON it sends each page once with PageCopies 1. Captures from the
original under i386 emulation, with the pages of `_pages` on Br3x5.
"""

import io

import numpy as np
import pytest

from brother_xl2hb.job import filter_pages
from brother_xl2hb.models import HL4150CDN
from brother_xl2hb.models.base import PageSend
from brother_xl2hb.settings import DuplexMode, PageSize, PrintSettings
from fixture_utils import assert_matches_fixture

_COLOURS = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 0, 0)]


def _pages(n: int, blank: tuple[int, ...] = ()) -> list[tuple[int, int, bytes]]:
    """`n` full-height Br3x5 pages, asymmetric in x and y; the pages in `blank` stay white."""
    w, h = HL4150CDN.paper_sizes["Br3x5"]
    pages = []
    for i in range(n):
        page = np.full((h, w, 3), 255, np.uint8)
        if i not in blank:
            page[100 + i * 50 : 160 + i * 50, 100:900] = (0, 0, 0)
            page[2000:2080, 800 + i * 100 : 1500] = _COLOURS[i % 4]
        pages.append((w, h, page.tobytes()))
    return pages


def _run_rc(tmp_path, pages, rc: str) -> bytes:
    rc_file = tmp_path / "brhl4150cdnrc"
    rc_file.write_text(f"[hl4150cdn]\nPageSize=Br3x5\n{rc}")
    out = io.BytesIO()
    filter_pages(pages, HL4150CDN.settings_from_rc_file(str(rc_file)), out)
    return out.getvalue()


@pytest.mark.parametrize(
    ("fixture", "n_pages", "blank", "rc"),
    [
        ("copies3_long_edge", 3, (), "Copies=3\nDuplex=DuplexNoTumble\n"),
        (
            "copies2_fine_reverse_short_edge",
            3,
            (),
            "Copies=2\nDuplex=DuplexTumble\nBRResolution=Fine\nBRReverse=ON\n",
        ),
        (
            "copies2_skip_blank_reverse_long_edge",
            4,
            (1,),
            "Copies=2\nDuplex=DuplexNoTumble\nBRSkipBlank=ON\nBRReverse=ON\n",
        ),
    ],
)
def test_uncollated_copies_match_original(tmp_path, fixture, n_pages, blank, rc):
    assert_matches_fixture(fixture, _run_rc(tmp_path, _pages(n_pages, blank), rc))


def test_collated_copies_send_each_page_once(tmp_path):
    """Collate=ON leaves copies to the caller: the original's output is that of Copies=1."""
    pages = _pages(3)
    collated = _run_rc(tmp_path, pages, "Copies=3\nCollate=ON\nDuplex=DuplexNoTumble\n")
    single = _run_rc(tmp_path, pages, "Copies=1\nDuplex=DuplexNoTumble\n")
    assert collated == single


def test_rc_collate_needs_exact_on(tmp_path):
    """Brother's filter takes only "ON"; any other spelling leaves copies uncollated."""
    rc_file = tmp_path / "brhl4150cdnrc"
    for value, collate in [("ON", True), ("On", False), ("OFF", False)]:
        rc_file.write_text(f"[hl4150cdn]\nCollate={value}\n")
        assert HL4150CDN.settings_from_rc_file(str(rc_file)).collate is collate


@pytest.mark.parametrize(
    ("duplex", "reverse", "index", "flips"),
    [
        (DuplexMode.NONE, False, 1, (False, False, False)),
        (DuplexMode.NONE, True, 0, (True, True, True)),
        (DuplexMode.NO_TUMBLE, False, 0, (False, True, False)),
        (DuplexMode.NO_TUMBLE, False, 1, (True, False, True)),  # sides 3, 4, 5
        (DuplexMode.NO_TUMBLE, True, 0, (True, False, True)),
        (DuplexMode.TUMBLE, False, 1, (False, False, False)),
    ],
)
def test_page_sends_repeat_each_page_with_sides_per_send(duplex, reverse, index, flips):
    settings = PrintSettings(duplex=duplex, reverse=reverse, copies=3)
    assert HL4150CDN.page_sends(settings, index) == tuple(PageSend(flip=f, copies=3) for f in flips)


def test_many_copies_render_each_distinct_send_once(monkeypatch):
    """Copies=1000 repeats bytes instead of rendering the page 1000 times."""
    import brother_xl2hb.job as job_module

    calls = []
    real = job_module.render_page

    def counting(*args):
        calls.append(args[-1])
        real(*args)

    monkeypatch.setattr(job_module, "render_page", counting)
    w, h = 64, HL4150CDN.paper_sizes["Postcard"][1]
    page = (w, h, b"\x00" * (w * h * 3))
    settings = PrintSettings(page_size=PageSize.POSTCARD, duplex=DuplexMode.NO_TUMBLE, copies=1000)
    out = io.BytesIO()
    filter_pages([page], settings, out)
    assert sorted(send.flip for send in calls) == [False, True]
    assert out.getvalue().count(b"\xf8\x31\x44") == 1000  # PageCopies attribute + EndPage

"""Fine mode (600x2400 dpi, APTMODE=ON4) byte-exact against brhl4150cdnfilter.

Each case renders a mostly white page (A4 unless the case sets another of
the 21 paper sizes) with a short band of content and
compares the whole XL2HB job with a capture of the original filter for
the same page and rc settings (`FINE_CASES`, captured with
scripts/fine_diff.py's Docker runner).
"""

import io

import numpy as np
import pytest

from brother_xl2hb.job import filter_page, filter_pages
from brother_xl2hb.models import HL4150CDN
from brother_xl2hb.settings import DuplexMode, MonoColor, PageSize, PrintSettings, Resolution
from fixture_utils import assert_matches_fixture

TOP = 100


def fine_test_page(kind: str, lines: int, page_size: str = "A4") -> np.ndarray:
    """Page of `page_size`, white except for `lines` rows of `kind` from row TOP.

    Returns:
        (height, width, 3) uint8 RGB page.
    """
    width, height = HL4150CDN.paper_sizes[page_size]
    page = np.full((height, width, 3), 255, np.uint8)
    band = page[TOP : TOP + lines]
    if kind == "hramp":
        # Black on the left, white on the right: the row ends in a no-ink run.
        band[:] = (np.arange(width) * 256 // width).astype(np.uint8)[None, :, None]
    elif kind == "bars":
        colours = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0)]
        colours.append((128, 128, 128))
        for i in range(lines):
            band[i] = colours[i * len(colours) // lines]
    elif kind == "noise":
        band[:] = np.random.default_rng(0).integers(0, 256, band.shape, dtype=np.uint8)
    else:
        raise ValueError(kind)
    return page


# fixture name -> (page kind, rows of content, PrintSettings overrides, rc overrides)
FINE_CASES = {
    "fine_hramp": ("hramp", 16, {}, []),
    "fine_bars": ("bars", 24, {}, []),
    "fine_noise": ("noise", 8, {}, []),
    "fine_toner_save": ("hramp", 8, {"toner_save": True}, ["TonerSaveMode=ON"]),
    "fine_mono": ("bars", 16, {"mono_color": MonoColor.MONO}, ["BRMonoColor=Mono"]),
    "fine_brightness_n20": ("hramp", 8, {"brightness": -20}, ["Brightness=-20"]),
    # Widths that are not a multiple of 8 pixels: rows are padded to 8 (Letter 4900 -> 4904).
    "fine_letter": ("bars", 16, {"page_size": PageSize.LETTER}, ["PageSize=Letter"]),
    "fine_postcard": ("hramp", 8, {"page_size": PageSize("Postcard")}, ["PageSize=Postcard"]),
}
# Every other paper size: bars across the full width, so the last pixel of
# each row carries ink (fine_<size>, e.g. fine_envmonarch).
FINE_CASES.update(
    {
        f"fine_{size.lower()}": ("bars", 16, {"page_size": PageSize(size)}, [f"PageSize={size}"])
        for size in sorted(HL4150CDN.paper_sizes)
        if size not in {"A4", "Letter", "Postcard"}
    }
)

# Full-width solid bands on an A4 page: (first row, end row, RGB).
SOLID_BAND_CASES = {
    "fine_cyan100": (3000, 3100, (0, 255, 255)),
    "fine_gray50": (1000, 2000, (128, 128, 128)),
}

# Normal mode, same pages: negative brightness rounds its shift down.
NORMAL_CASES = {
    "hramp_brightness_n20": ("hramp", 8, {"brightness": -20}, ["Brightness=-20"]),
    "hramp_brightness_n5": ("hramp", 8, {"brightness": -5}, ["Brightness=-5"]),
}


def _render(kind: str, lines: int, overrides: dict, *, fine: bool) -> bytes:
    resolution = Resolution.FINE if fine else Resolution.NORMAL
    settings = PrintSettings(resolution=resolution, **overrides)
    page = fine_test_page(kind, lines, settings.page_size)
    out = io.BytesIO()
    filter_page(page.shape[1], page.shape[0], page, settings, out)
    return out.getvalue()


@pytest.mark.parametrize("name", FINE_CASES)
def test_fine_page_matches_original(name):
    kind, lines, overrides, _ = FINE_CASES[name]
    assert_matches_fixture(name, _render(kind, lines, overrides, fine=True))


@pytest.mark.parametrize("name", SOLID_BAND_CASES)
def test_fine_solid_band_matches_original(name):
    first, end, rgb = SOLID_BAND_CASES[name]
    page = fine_test_page("bars", 0)
    page[first:end] = rgb
    out = io.BytesIO()
    filter_page(page.shape[1], page.shape[0], page, PrintSettings(resolution=Resolution.FINE), out)
    assert_matches_fixture(name, out.getvalue())


@pytest.mark.parametrize("name", NORMAL_CASES)
def test_normal_negative_brightness_matches_original(name):
    kind, lines, overrides, _ = NORMAL_CASES[name]
    assert_matches_fixture(name, _render(kind, lines, overrides, fine=False))


def test_fine_reverse_long_edge_job_matches_original():
    """Reverse with long-edge duplex in Fine mode: pages 1 and 3 are flipped, page 2 is not."""
    pages = [fine_test_page(kind, 16) for kind in ("hramp", "bars", "noise")]
    settings = PrintSettings(resolution=Resolution.FINE, duplex=DuplexMode.NO_TUMBLE, reverse=True)
    out = io.BytesIO()
    filter_pages([(page.shape[1], page.shape[0], page) for page in pages], settings, out)
    assert_matches_fixture("fine_reverse3_long_edge", out.getvalue())

"""Fine mode (600x2400 dpi, APTMODE=ON4) byte-exact against brhl4150cdnfilter.

Each case renders a mostly white A4 page with a short band of content and
compares the whole XL2HB job with a capture of the original filter for
the same page and rc settings (`FINE_CASES`, captured with
scripts/fine_diff.py's Docker runner).
"""

import io

import numpy as np
import pytest

from fixture_utils import assert_matches_fixture
from pipeline import filter_page
from settings import MonoColor, PrintSettings, Resolution

A4_W, A4_H = 4760, 6812
TOP = 100


def fine_test_page(kind: str, lines: int) -> np.ndarray:
    """A4 page, white except for `lines` rows of `kind` from row TOP.

    Returns:
        (A4_H, A4_W, 3) uint8 RGB page.
    """
    page = np.full((A4_H, A4_W, 3), 255, np.uint8)
    band = page[TOP : TOP + lines]
    if kind == "hramp":
        # Black on the left, white on the right: the row ends in a no-ink run.
        band[:] = (np.arange(A4_W) * 256 // A4_W).astype(np.uint8)[None, :, None]
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
}

# Normal mode, same pages: negative brightness rounds its shift down.
NORMAL_CASES = {
    "hramp_brightness_n20": ("hramp", 8, {"brightness": -20}, ["Brightness=-20"]),
    "hramp_brightness_n5": ("hramp", 8, {"brightness": -5}, ["Brightness=-5"]),
}


def _render(kind: str, lines: int, overrides: dict, *, fine: bool) -> bytes:
    resolution = Resolution.FINE if fine else Resolution.NORMAL
    settings = PrintSettings(resolution=resolution, **overrides)
    out = io.BytesIO()
    filter_page(A4_W, A4_H, fine_test_page(kind, lines), settings, out)
    return out.getvalue()


@pytest.mark.parametrize("name", FINE_CASES)
def test_fine_page_matches_original(name):
    kind, lines, overrides, _ = FINE_CASES[name]
    assert_matches_fixture(name, _render(kind, lines, overrides, fine=True))


@pytest.mark.parametrize("name", NORMAL_CASES)
def test_normal_negative_brightness_matches_original(name):
    kind, lines, overrides, _ = NORMAL_CASES[name]
    assert_matches_fixture(name, _render(kind, lines, overrides, fine=False))

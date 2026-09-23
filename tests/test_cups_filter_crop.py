"""crop_page in the CUPS filter script: numpy slicing vs. the row-by-row reference."""

import importlib.machinery
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_FILTER = Path(__file__).resolve().parent.parent / "cups" / "brhl4150cdn-filter"


@pytest.fixture(scope="module")
def cups_filter():
    loader = importlib.machinery.SourceFileLoader("brhl4150cdn_filter", str(_FILTER))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def _reference_crop(pixel_data, src_width, src_height, target_w, target_h):
    """The original per-row implementation."""
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


@pytest.mark.parametrize(
    ("src_width", "src_height"),
    [(4958, 7016), (4700, 7016), (4958, 6000), (4000, 5000), (90, 7016), (4958, 50)],
)
def test_crop_matches_reference(cups_filter, src_width, src_height):
    target_w, target_h = cups_filter.PAPER_SIZES["A4"]
    rng = np.random.default_rng(src_width + src_height)
    pixel_data = rng.integers(0, 256, src_width * src_height * 3, dtype=np.uint8).tobytes()
    w, h, data = cups_filter.crop_page(pixel_data, src_width, src_height, "A4")
    assert (w, h) == (target_w, target_h)
    assert data == _reference_crop(pixel_data, src_width, src_height, target_w, target_h)

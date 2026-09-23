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
    assert data.shape == (target_h, target_w, 3)
    assert data.tobytes() == _reference_crop(pixel_data, src_width, src_height, target_w, target_h)


def test_crop_is_a_view_when_render_covers_printable_area(cups_filter):
    target_w, target_h = cups_filter.PAPER_SIZES["A4"]
    src_width, src_height = target_w + 200, target_h + 200
    pixel_data = bytes(src_width * src_height * 3)
    _, _, data = cups_filter.crop_page(pixel_data, src_width, src_height, "A4")
    assert np.shares_memory(data, np.frombuffer(pixel_data, dtype=np.uint8))


@pytest.mark.parametrize(
    ("dsc", "expected"),
    [
        (b"%!PS-Adobe-3.0\n%%Pages: 7\n%%EndComments\n", 7),
        (b"%!PS-Adobe-3.0\n%%Pages: (atend)\n%%EndComments\n", None),
    ],
)
def test_count_ps_pages_reads_dsc(cups_filter, tmp_path, dsc, expected):
    body = b"showpage\n" * 3
    trailer = b"%%Trailer\n%%Pages: 3\n%%EOF\n" if expected is None else b"%%EOF\n"
    path = tmp_path / "job.ps"
    path.write_bytes(dsc + body + trailer)
    assert cups_filter.count_ps_pages(str(path)) == (expected or 3)


def test_count_ps_pages_falls_back_to_ghostscript(cups_filter, tmp_path):
    if cups_filter.shutil.which("gs") is None:
        pytest.skip("Ghostscript not installed")
    path = tmp_path / "job.ps"
    path.write_bytes(b"%!PS\n" + b"newpath 10 10 moveto 20 20 lineto stroke showpage\n" * 5)
    assert cups_filter.count_ps_pages(str(path)) == 5

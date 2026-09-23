"""Helpers in the CUPS filter script (loaded from cups/brhl4150cdn-filter)."""

import importlib.machinery
import importlib.util
from pathlib import Path

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

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


_PPD = _FILTER.parent / "brhl4150cdn.ppd"


def _ppd_paper_dimensions() -> dict[str, tuple[int, int]]:
    """PaperDimension of every PageSize in the PPD, in points."""
    dims = {}
    for line in _PPD.read_text(encoding="latin-1").splitlines():
        if line.startswith("*PaperDimension "):
            name = line.split()[1].split("/")[0]
            width, height = line.split('"')[1].split()
            dims[name] = (round(float(width)), round(float(height)))
    return dims


def test_paper_points_match_the_ppd(cups_filter):
    """Ghostscript renders the page at the size the PPD advertises, for every PageSize."""
    from settings import PageSize

    ppd = _ppd_paper_dimensions()
    assert set(ppd) == {size.value for size in PageSize}
    assert ppd == cups_filter.PAPER_POINTS


@pytest.mark.parametrize("page_size", sorted(_ppd_paper_dimensions()))
def test_gs_command_sets_the_page_size_in_points(cups_filter, monkeypatch, page_size):
    monkeypatch.setattr(cups_filter, "find_gs", lambda: "gs")
    cmd = cups_filter.build_gs_command("job.ps", page_size)
    width, height = _ppd_paper_dimensions()[page_size]
    assert f"-dDEVICEWIDTHPOINTS={width}" in cmd
    assert f"-dDEVICEHEIGHTPOINTS={height}" in cmd
    assert not any(arg.startswith("-sPAPERSIZE") for arg in cmd)
    assert cmd[-1] == "job.ps"


@pytest.mark.parametrize("page_size", ["Executive", "Env10", "EnvMonarch", "A4"])
def test_ghostscript_renders_the_requested_size(cups_filter, tmp_path, page_size):
    """Executive, Com-10 and Monarch came out as Letter while gs got paper names."""
    if cups_filter.shutil.which("gs") is None:
        pytest.skip("Ghostscript not installed")
    job = tmp_path / "job.ps"
    job.write_bytes(b"%!PS\nshowpage\n")
    proc = cups_filter.subprocess.Popen(
        cups_filter.build_gs_command(str(job), page_size),
        stdout=cups_filter.subprocess.PIPE,
        stderr=cups_filter.subprocess.DEVNULL,
    )
    try:
        header = proc.stdout.read(512)
    finally:
        proc.kill()
        proc.wait()
    fields = [line for line in header.split(b"\n")[:4] if not line.startswith(b"#")]
    width, height = (int(v) for v in fields[1].split())
    w_pt, h_pt = cups_filter.PAPER_POINTS[page_size]
    assert (width, height) == (round(w_pt * 600 / 72), round(h_pt * 600 / 72))

"""Helpers in the CUPS filter (brother_xl2hb.cups_filter)."""

from pathlib import Path

import pytest

from brother_xl2hb import cups_filter as cups_filter_module
from brother_xl2hb.models import HL4150CDN


@pytest.fixture(scope="module")
def cups_filter():
    return cups_filter_module


_PPD = Path(__file__).resolve().parent.parent / "cups" / "brhl4150cdn.ppd"


def _ppd_paper_dimensions() -> dict[str, tuple[int, int]]:
    """PaperDimension of every PageSize in the PPD, in points."""
    dims = {}
    for line in _PPD.read_text(encoding="latin-1").splitlines():
        if line.startswith("*PaperDimension "):
            name = line.split()[1].split("/")[0]
            width, height = line.split('"')[1].split()
            dims[name] = (round(float(width)), round(float(height)))
    return dims


@pytest.mark.parametrize("page_size", sorted(_ppd_paper_dimensions()))
def test_gs_command_sets_the_page_size_in_points(cups_filter, monkeypatch, page_size):
    monkeypatch.setattr(cups_filter, "find_gs", lambda: "gs")
    cmd = cups_filter.build_gs_command("job.ps", page_size, HL4150CDN)
    width, height = _ppd_paper_dimensions()[page_size]
    assert f"-dDEVICEWIDTHPOINTS={width}" in cmd
    assert f"-dDEVICEHEIGHTPOINTS={height}" in cmd
    assert not any(arg.startswith("-sPAPERSIZE") for arg in cmd)
    assert cmd[-1] == "job.ps"


@pytest.mark.parametrize("page_size", ["Executive", "Env10", "EnvMonarch", "A4"])
def test_ghostscript_renders_the_requested_size(cups_filter, tmp_path, page_size):
    """Ghostscript renders each page size at its PPD dimensions, not as Letter."""
    if cups_filter.shutil.which("gs") is None:
        pytest.skip("Ghostscript not installed")
    job = tmp_path / "job.ps"
    job.write_bytes(b"%!PS\nshowpage\n")
    proc = cups_filter.subprocess.Popen(
        cups_filter.build_gs_command(str(job), page_size, HL4150CDN),
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
    w_pt, h_pt = HL4150CDN.paper_points[page_size]
    assert (width, height) == (round(w_pt * 600 / 72), round(h_pt * 600 / 72))

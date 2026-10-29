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
    """Executive, Com-10 and Monarch came out as Letter while gs got paper names."""
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


def test_gs_command_without_collated_copies_ends_with_the_job(cups_filter, monkeypatch):
    monkeypatch.setattr(cups_filter, "find_gs", lambda: "gs")
    plain = cups_filter.build_gs_command("-", "A4", HL4150CDN)
    assert plain == cups_filter.build_gs_command("-", "A4", HL4150CDN, collate_copies=False)
    assert plain[-1] == "-"
    assert "-c" not in plain


def test_gs_command_with_collated_copies_reruns_the_job_file(cups_filter, monkeypatch):
    monkeypatch.setattr(cups_filter, "find_gs", lambda: "gs")
    job = "/tmp/br job (1).ps"
    cmd = cups_filter.build_gs_command(job, "A4", HL4150CDN, collate_copies=True)
    assert f"--permit-file-read={job}" in cmd
    assert cmd[-3:-1] == ["-c", cups_filter.DUPLEX_COPIES_PROLOGUE]
    # The path goes in as a hex string, so parentheses need no escaping.
    assert cmd[-1] == cups_filter.DUPLEX_COPIES_RUN.format(job=job.encode().hex())
    assert cmd[-1].startswith(f"<{job.encode().hex()}> run ")
    assert job not in cmd[:-4]


def test_gs_command_refuses_collated_copies_from_stdin(cups_filter, monkeypatch):
    monkeypatch.setattr(cups_filter, "find_gs", lambda: "gs")
    with pytest.raises(ValueError, match="as a file"):
        cups_filter.build_gs_command("-", "A4", HL4150CDN, collate_copies=True)


# Duplex copies through a real Ghostscript. Page n of a test job is filled
# with gray n/10, so the order of the rendered pages can be read off their
# centres ("B" for a blank page).


def _test_job(pages: int, setup: str = "") -> bytes:
    lines = ["%!PS-Adobe-3.0", f"%%Pages: {pages}", "%%EndComments", "%%BeginSetup", setup, "%%EndSetup"]
    for n in range(1, pages + 1):
        lines += [f"%%Page: {n} {n}", f"{n / 10} setgray 0 0 1000 1000 rectfill showpage"]
    return "\n".join([*lines, "%%EOF", ""]).encode()


def _page_order(ppm: bytes) -> str:
    order = []
    pos = 0
    while pos < len(ppm):
        fields = []
        while len(fields) < 4:
            while ppm[pos : pos + 1].isspace() or ppm[pos : pos + 1] == b"#":
                pos = ppm.index(b"\n", pos) + 1 if ppm[pos : pos + 1] == b"#" else pos + 1
            end = pos
            while not ppm[end : end + 1].isspace():
                end += 1
            fields.append(ppm[pos:end])
            pos = end
        pos += 1
        width, height = int(fields[1]), int(fields[2])
        level = round(ppm[pos + 3 * (width * (height // 2) + width // 2)] * 10 / 255)
        order.append("B" if level == 10 else str(level))
        pos += 3 * width * height
    return " ".join(order)


def _render(cups_filter, job_path: Path, *, collate_copies: bool) -> str:
    cmd = cups_filter.build_gs_command(str(job_path), "Postcard", HL4150CDN, collate_copies=collate_copies)
    cmd[cmd.index("-r600")] = "-r10"
    result = cups_filter.subprocess.run(cmd, capture_output=True, check=True, timeout=60)
    return _page_order(result.stdout)


# How pstops asks for uncollated copies, in the job's setup.
_NUM_COPIES = (
    "{n} /languagelevel where{{pop languagelevel 2 ge}}{{false}}ifelse\n"
    "{{1 dict begin/NumCopies exch def currentdict end setpagedevice}}\n"
    "{{userdict/#copies 3 -1 roll put}}ifelse"
)


@pytest.mark.parametrize(
    ("pages", "copies", "expected"),
    [
        (3, 1, "1 2 3"),
        (3, 2, "1 2 3 B 1 2 3"),
        (3, 3, "1 2 3 B 1 2 3 B 1 2 3"),
        (4, 2, "1 2 3 4 1 2 3 4"),
        (1, 3, "1 B 1 B 1"),
    ],
)
def test_ghostscript_collates_duplex_copies(cups_filter, tmp_path, pages, copies, expected):
    if cups_filter.shutil.which("gs") is None:
        pytest.skip("Ghostscript not installed")
    job = tmp_path / "job (copies).ps"
    job.write_bytes(_test_job(pages, _NUM_COPIES.format(n=copies)))
    uncollated = " ".join(str(n) for n in range(1, pages + 1) for _ in range(copies))
    assert _render(cups_filter, job, collate_copies=False) == uncollated
    assert _render(cups_filter, job, collate_copies=True) == expected


@pytest.mark.parametrize("setup", ["", "<</NumCopies null>> setpagedevice", "<</NumCopies 1>> setpagedevice"])
def test_ghostscript_keeps_single_copies(cups_filter, tmp_path, setup):
    if cups_filter.shutil.which("gs") is None:
        pytest.skip("Ghostscript not installed")
    job = tmp_path / "job.ps"
    job.write_bytes(_test_job(3, setup))
    assert _render(cups_filter, job, collate_copies=True) == "1 2 3"


def _find_pstops() -> Path | None:
    for directory in ("/usr/lib/cups/filter", "/usr/libexec/cups/filter"):
        pstops = Path(directory) / "pstops"
        if pstops.is_file():
            return pstops
    return None


@pytest.mark.parametrize("sides", ["one-sided", "two-sided-long-edge"])
@pytest.mark.parametrize("collate", [True, False])
@pytest.mark.parametrize(("pages", "copies"), [(3, 2), (3, 3), (4, 2)])
def test_pstops_copies_through_ghostscript(cups_filter, tmp_path, sides, collate, pages, copies):
    """The PostScript path with the system's pstops and this PPD, as print_job runs it."""
    pstops = _find_pstops()
    if pstops is None or cups_filter.shutil.which("gs") is None:
        pytest.skip("pstops or Ghostscript not installed")
    source = tmp_path / "source.ps"
    source.write_bytes(_test_job(pages))
    job = tmp_path / "job.ps"
    with job.open("wb") as out:
        cups_filter.subprocess.run(
            [str(pstops), "1", "user", "title", str(copies), f"sides={sides} Collate={collate}", str(source)],
            stdout=out,
            stderr=cups_filter.subprocess.DEVNULL,
            env={**cups_filter.os.environ, "PPD": str(_PPD)},
            check=True,
            timeout=60,
        )
    duplex = sides != "one-sided"
    order = _render(cups_filter, job, collate_copies=duplex).split()
    document = [str(n) for n in range(1, pages + 1)]
    if not duplex:
        # Unchanged: Ghostscript repeats each page for uncollated copies.
        assert order == (document * copies if collate else [n for n in document for _ in range(copies)])
        return
    # Each copy starts on a new sheet; pstops' collated copies also pad the last one.
    sheets = document + ["B"] * (pages % 2)
    assert order in (sheets * copies, (sheets * copies)[: len(sheets) * copies - pages % 2])

"""End-to-end runs of the CUPS filter's main() against a fake Ghostscript.

The filter runs as CUPS runs it, as a subprocess with the CUPS argument
vector. A stand-in `gs` first on PATH writes PPM pages (or fails) as the
test asks and records how it was called.
"""

import io
import json
import os
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from pipeline import filter_pages
from settings import PrintSettings
from xl2hb import OP_END_SESSION, generate_pjl_footer

_FILTER = Path(__file__).resolve().parent.parent / "cups" / "brhl4150cdn-filter"

# Postcard keeps the rendered pages small.
_OPTIONS = "PageSize=Postcard"
_JOB = b"%!PS-Adobe-3.0\n%%Pages: 2\n%%EndComments\nshowpage\nshowpage\n%%EOF\n"

_FAKE_GS = textwrap.dedent(
    """\
    #!{python}
    import json, os, sys

    args = sys.argv[1:]
    source = args[-1]
    job = sys.stdin.buffer.read() if source == "-" else open(source, "rb").read()
    with open(os.environ["FAKE_GS_RECORD"], "w") as record:
        json.dump({{"args": args, "job": job.decode("latin-1")}}, record)

    mode = os.environ.get("FAKE_GS_MODE", "ok")
    if mode == "fail_start":
        sys.stderr.write("Error: /undefined in foo\\nOperand stack:\\n")
        sys.exit(1)

    points = dict(a[2:].split("=") for a in args if a.startswith("-dDEVICE") and "POINTS=" in a)
    width = round(int(points["DEVICEWIDTHPOINTS"]) * 600 / 72)
    height = round(int(points["DEVICEHEIGHTPOINTS"]) * 600 / 72)
    header = b"P6\\n%d %d\\n255\\n" % (width, height)
    row = bytes([200, 120, 40]) * width
    out = sys.stdout.buffer
    for _ in range(int(os.environ.get("FAKE_GS_PAGES", "1"))):
        out.write(header)
        for _ in range(height):
            out.write(row)
    if mode == "truncate":
        # A second page that breaks off halfway, as when gs dies mid-job.
        out.write(header)
        for _ in range(height // 2):
            out.write(row)
        out.flush()
        sys.stderr.write("Error: /ioerror in --showpage--\\n")
        sys.exit(3)
    """
)


@pytest.fixture
def fake_gs(tmp_path):
    """Put a fake `gs` first on PATH; return the environment for the filter."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gs = bin_dir / "gs"
    gs.write_text(_FAKE_GS.format(python=sys.executable))
    gs.chmod(gs.stat().st_mode | stat.S_IXUSR)
    env = {k: v for k, v in os.environ.items() if k != "PPD"}
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    env["FAKE_GS_RECORD"] = str(tmp_path / "gs-call.json")
    return env


def _run(env, *, options=_OPTIONS, copies="1", file=None, stdin=_JOB, **fake):
    argv = [sys.executable, str(_FILTER), "42", "alice", "My Title", copies, options]
    if file is not None:
        argv.append(str(file))
    env = {**env, **{f"FAKE_GS_{key.upper()}": str(value) for key, value in fake.items()}}
    return subprocess.run(argv, input=stdin, capture_output=True, env=env, timeout=120, check=False)


def _gs_call(env):
    return json.loads(Path(env["FAKE_GS_RECORD"]).read_text())


def test_reads_the_job_from_stdin(fake_gs):
    result = _run(fake_gs)
    assert result.returncode == 0, result.stderr.decode()
    call = _gs_call(fake_gs)
    assert call["args"][-1] == "-"
    assert call["job"].encode("latin-1") == _JOB
    assert result.stdout.startswith(b"\x1b%-12345X@PJL")
    assert result.stdout.endswith(bytes([OP_END_SESSION]) + generate_pjl_footer())
    assert b"INFO: brhl4150cdn-filter: job=42 user=alice title='My Title' copies=1" in result.stderr


def test_reads_the_job_from_the_file_argument(fake_gs, tmp_path):
    job = tmp_path / "job.ps"
    job.write_bytes(_JOB)
    from_file = _run(fake_gs, file=job, stdin=b"")
    assert from_file.returncode == 0, from_file.stderr.decode()
    assert _gs_call(fake_gs)["args"][-1] == str(job)
    assert from_file.stdout == _run(fake_gs).stdout


def test_copies_argument_does_not_reach_the_job(fake_gs):
    # Ghostscript already renders each page `copies` times (pstops sets
    # NumCopies); a PageCopies on top would multiply them again.
    one, two = _run(fake_gs), _run(fake_gs, copies="2")
    assert two.returncode == 0
    assert b"copies=2" in two.stderr
    assert one.stdout == two.stdout


def test_usage_error_without_cups_arguments(fake_gs):
    result = subprocess.run(
        [sys.executable, str(_FILTER), "42"], capture_output=True, env=fake_gs, timeout=60, check=False
    )
    assert result.returncode == 1
    assert b"ERROR: brhl4150cdn-filter: Usage:" in result.stderr
    assert result.stdout == b""


def test_reverse_long_edge_streams_the_job_from_stdin(fake_gs):
    """Reverse turns pages in place, so the job needs neither spooling nor a page count."""
    result = _run(fake_gs, options=f"{_OPTIONS} Duplex=DuplexNoTumble BRReverse=ON", pages=2)
    assert result.returncode == 0, result.stderr.decode()
    assert _gs_call(fake_gs)["args"][-1] == "-"
    assert result.stdout.endswith(bytes([OP_END_SESSION]) + generate_pjl_footer())


def test_gs_failure_at_start(fake_gs):
    result = _run(fake_gs, mode="fail_start")
    assert result.returncode == 1
    assert result.stdout == b""
    assert b"ERROR: brhl4150cdn-filter: GS: Error: /undefined in foo" in result.stderr
    assert b"ERROR: brhl4150cdn-filter: Ghostscript exited with status 1" in result.stderr
    assert b"Traceback" not in result.stderr


def test_gs_failure_mid_job_sends_only_complete_pages(fake_gs):
    """A page that breaks off is dropped; the job still ends with EndSession and UEL."""
    complete = _run(fake_gs, pages=1)
    assert complete.returncode == 0

    broken = _run(fake_gs, pages=1, mode="truncate")
    assert broken.returncode == 1
    assert broken.stdout == complete.stdout
    stderr = broken.stderr.decode()
    assert "ERROR: brhl4150cdn-filter: Ghostscript failed with exit status 3" in stderr
    assert "ERROR: brhl4150cdn-filter: GS: Error: /ioerror in --showpage--" in stderr
    assert "while reading page 2 from Ghostscript" in stderr
    assert "Traceback" not in stderr


def test_gs_failure_mid_job_in_reverse_order_sends_only_complete_pages(fake_gs):
    options = f"{_OPTIONS} BRReverse=ON"
    complete = _run(fake_gs, options=options, pages=1)
    assert complete.returncode == 0
    broken = _run(fake_gs, options=options, pages=1, mode="truncate")
    assert broken.returncode == 1
    assert broken.stdout == complete.stdout


def test_fine_resolution_prints_in_fine_mode(fake_gs):
    normal = _run(fake_gs)
    fine = _run(fake_gs, options=f"{_OPTIONS} BRResolution=600x2400dpi")
    assert fine.returncode == 0
    assert b"resolution=Fine" in fine.stderr
    assert b"@PJL SET APTMODE=ON4" in fine.stdout
    assert b"@PJL SET APTMODE=OFF" in normal.stdout


def test_ppd_offers_normal_and_fine():
    ppd = (_FILTER.parent / "brhl4150cdn.ppd").read_text(encoding="latin-1")
    choices = [line for line in ppd.splitlines() if line.startswith("*BRResolution ")]
    assert choices == [
        '*BRResolution 600dpi/Normal (600 x 600dpi): \t"          "',
        '*BRResolution 600x2400dpi/Fine (2400 dpi class): \t"          "',
    ]


@pytest.mark.parametrize(("value", "expected"), [("2", 2), ("0", 0), ("-1", 0)])
def test_render_threads_from_environment(monkeypatch, value, expected):
    import pipeline

    monkeypatch.setenv("BRHL4150CDN_RENDER_THREADS", value)
    assert pipeline._render_threads() == expected


def test_non_numeric_render_threads_fall_back_to_the_default(monkeypatch, caplog):
    import pipeline

    monkeypatch.delenv("BRHL4150CDN_RENDER_THREADS", raising=False)
    default = pipeline._render_threads()
    monkeypatch.setenv("BRHL4150CDN_RENDER_THREADS", "four")
    assert pipeline._render_threads() == default
    assert "non-numeric BRHL4150CDN_RENDER_THREADS='four'" in caplog.text


def test_filter_pages_closes_the_session_when_a_page_breaks_off():
    """Only complete pages are written; the session and PJL job still end."""
    import numpy as np

    width, height = 64, 32
    good = bytes([90, 60, 30]) * (width * height)

    def pages():
        yield width, height, good
        yield width, height, iter([np.zeros((5, width * 3), np.uint8)])  # 5 of 32 rows
        yield width, height, good

    one_page = io.BytesIO()
    filter_pages([(width, height, good)], PrintSettings(), one_page)
    out = io.BytesIO()
    with pytest.raises(ValueError, match="page ended after 5 rows"):
        filter_pages(pages(), PrintSettings(), out)
    assert out.getvalue() == one_page.getvalue()


def test_filter_pages_does_not_mask_a_broken_output():
    class Broken(io.BytesIO):
        def write(self, data):
            raise BrokenPipeError

    with pytest.raises(BrokenPipeError):
        filter_pages([(8, 4, bytes(96))], PrintSettings(), Broken())

"""End-to-end runs of the CUPS filter's main() against a fake Ghostscript.

A stand-in `gs` first on PATH writes PPM pages (or fails) as the test
asks and records how it was called. Most tests call main() in this
process with the CUPS argument vector, the job on file descriptor 0 (which
Ghostscript inherits) and stdout replaced; a few run the filter script as
CUPS does, as a subprocess, to check what reaches CUPS's error log.
"""

import contextlib
import io
import json
import logging
import os
import stat
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import NamedTuple

import pytest

from brother_xl2hb import cups_filter
from brother_xl2hb.job import filter_pages
from brother_xl2hb.models import HL4150CDN
from brother_xl2hb.protocol.pjl import generate_pjl_footer
from brother_xl2hb.protocol.xl2hb import OP_END_SESSION
from brother_xl2hb.settings import PrintSettings

_FILTER = Path(__file__).resolve().parent.parent / "cups" / "brhl4150cdn-filter"

# Postcard keeps the rendered pages small.
_OPTIONS = "PageSize=Postcard"
_JOB = b"%!PS-Adobe-3.0\n%%Pages: 2\n%%EndComments\nshowpage\nshowpage\n%%EOF\n"
_JOB_END = bytes([OP_END_SESSION]) + generate_pjl_footer()

_FAKE_GS = textwrap.dedent(
    """\
    #!{python}
    import json, os, sys

    args = sys.argv[1:]
    source = args[-1]
    job = sys.stdin.buffer.read() if source == "-" else open(source, "rb").read()
    with open(os.environ["FAKE_GS_RECORD"], "w") as record:
        json.dump({{"args": args, "job": job.decode("latin-1"), "pid": os.getpid()}}, record)

    mode = os.environ.get("FAKE_GS_MODE", "ok")
    if mode == "fail_start":
        sys.stderr.write("Error: /undefined in foo\\nOperand stack:\\n")
        sys.exit(1)

    points = dict(a[2:].split("=") for a in args if a.startswith("-dDEVICE") and "POINTS=" in a)
    width = round(int(points["DEVICEWIDTHPOINTS"]) * 600 / 72)
    height = round(int(points["DEVICEHEIGHTPOINTS"]) * 600 / 72)
    header = b"P6\\n%d %d\\n255\\n" % (width, height)
    rgb = [int(v) for v in os.environ.get("FAKE_GS_RGB", "200,120,40").split(",")]
    row = bytes(rgb) * width
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
    if mode == "fail_end":
        # Every page written, then a failure at the end of the job.
        out.flush()
        sys.stderr.write("Error: /invalidfont in --findfont--\\n")
        sys.exit(2)
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


def _is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class _Stdout:
    """Stand-in for sys.stdout: the filter writes to its binary `buffer`."""

    def __init__(self, buffer: io.BytesIO) -> None:
        self.buffer = buffer


class Result(NamedTuple):
    """Exit status, what main() wrote to stdout, and its log."""

    returncode: int | str | None
    stdout: bytes
    log: str


@contextlib.contextmanager
def _stdin_fd(path: Path):
    """Point file descriptor 0, which Ghostscript inherits, at `path`."""
    saved = os.dup(0)
    try:
        with path.open("rb") as job:
            os.dup2(job.fileno(), 0)
        yield
    finally:
        os.dup2(saved, 0)
        os.close(saved)


@pytest.fixture
def run_main(fake_gs, monkeypatch, tmp_path, caplog):
    """Call `cups_filter.main` in this process as CUPS would run the filter.

    Keywords: `options`, `copies`, `file` (the job file argument), `stdin`,
    `argv` (replaces the CUPS arguments), `output` (stdout's buffer) and
    FAKE_GS_* settings for the fake gs, e.g. `pages=2` or `mode="truncate"`.
    """
    monkeypatch.setenv("PATH", fake_gs["PATH"])
    monkeypatch.setenv("FAKE_GS_RECORD", fake_gs["FAKE_GS_RECORD"])
    monkeypatch.delenv("PPD", raising=False)
    caplog.set_level(logging.INFO)

    def run(*, options=_OPTIONS, copies="1", file=None, stdin=_JOB, argv=None, output=None, **fake) -> Result:
        for key in ("MODE", "PAGES", "RGB"):
            monkeypatch.delenv(f"FAKE_GS_{key}", raising=False)
        for key, value in fake.items():
            monkeypatch.setenv(f"FAKE_GS_{key.upper()}", str(value))
        if argv is None:
            argv = ["42", "alice", "My Title", copies, options] + ([str(file)] if file is not None else [])
        monkeypatch.setattr(sys, "argv", [str(_FILTER), *argv])
        buffer = io.BytesIO() if output is None else output
        monkeypatch.setattr(sys, "stdout", _Stdout(buffer))
        stdin_path = tmp_path / "stdin"
        stdin_path.write_bytes(stdin)
        caplog.clear()
        with _stdin_fd(stdin_path), pytest.raises(SystemExit) as exit_info:
            cups_filter.main(HL4150CDN)
        return Result(exit_info.value.code, buffer.getvalue(), caplog.text)

    return run


# ---------------------------------------------------------------------------
# The filter script as CUPS runs it
# ---------------------------------------------------------------------------


def test_reads_the_job_from_stdin(fake_gs):
    result = _run(fake_gs)
    assert result.returncode == 0, result.stderr.decode()
    call = _gs_call(fake_gs)
    assert call["args"][-1] == "-"
    assert call["job"].encode("latin-1") == _JOB
    assert result.stdout.startswith(b"\x1b%-12345X@PJL")
    assert result.stdout.endswith(_JOB_END)
    assert b"INFO: brhl4150cdn-filter: job=42 user=alice title='My Title' copies=1" in result.stderr


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


# ---------------------------------------------------------------------------
# main() in this process
# ---------------------------------------------------------------------------


def test_main_reads_the_job_from_stdin(run_main, fake_gs):
    result = run_main()
    assert result.returncode == 0, result.log
    call = _gs_call(fake_gs)
    assert call["args"][-1] == "-"
    assert call["job"].encode("latin-1") == _JOB
    assert result.stdout.startswith(b"\x1b%-12345X@PJL")
    assert result.stdout.endswith(_JOB_END)
    assert "job=42 user=alice title='My Title' copies=1" in result.log
    assert "total pages: 1" in result.log
    assert "done" in result.log


def test_reads_the_job_from_the_file_argument(run_main, fake_gs, tmp_path):
    job = tmp_path / "job.ps"
    job.write_bytes(_JOB)
    from_file = run_main(file=job, stdin=b"")
    assert from_file.returncode == 0, from_file.log
    assert _gs_call(fake_gs)["args"][-1] == str(job)
    assert from_file.stdout == run_main().stdout


def test_copies_argument_does_not_reach_the_job(run_main):
    # Ghostscript already renders each page `copies` times (pstops sets
    # NumCopies); a PageCopies on top would multiply them again.
    one, two = run_main(), run_main(copies="2")
    assert two.returncode == 0
    assert "copies=2" in two.log
    assert one.stdout == two.stdout


def test_usage_error_without_cups_arguments(run_main):
    result = run_main(argv=["42"])
    assert result.returncode == 1
    assert "Usage:" in result.log
    assert result.stdout == b""


def test_job_options_override_the_ppd_defaults(run_main, monkeypatch, tmp_path):
    ppd = tmp_path / "queue.ppd"
    ppd.write_text('*PPD-Adobe: "4.3"\n*DefaultBRResolution: 600x2400dpi\n*DefaultPageSize: A4\n')
    monkeypatch.setenv("PPD", str(ppd))
    from_ppd = run_main()
    assert from_ppd.returncode == 0
    assert "page_size=Postcard" in from_ppd.log
    assert "resolution=Fine" in from_ppd.log
    assert b"@PJL SET APTMODE=ON4" in from_ppd.stdout

    from_job = run_main(options=f"{_OPTIONS} BRResolution=600dpi")
    assert "resolution=Normal" in from_job.log
    assert b"@PJL SET APTMODE=OFF" in from_job.stdout


def test_reverse_long_edge_streams_the_job_from_stdin(run_main, fake_gs):
    """Reverse turns pages in place, so the job needs neither spooling nor a page count."""
    result = run_main(options=f"{_OPTIONS} Duplex=DuplexNoTumble BRReverse=ON", pages=2)
    assert result.returncode == 0, result.log
    assert _gs_call(fake_gs)["args"][-1] == "-"
    assert result.stdout.endswith(_JOB_END)
    assert "total pages: 2" in result.log


def test_skip_blank_job_of_blank_pages_prints_nothing(run_main):
    result = run_main(options=f"{_OPTIONS} BRSkipBlank=ON", pages=2, rgb="255,255,255")
    assert result.returncode == 0, result.log
    assert result.stdout == b""
    assert "total pages: 2" in result.log


def test_gs_failure_at_start(run_main):
    result = run_main(mode="fail_start")
    assert result.returncode == 1
    assert result.stdout == b""
    assert "GS: Error: /undefined in foo" in result.log
    assert "Ghostscript exited with status 1" in result.log


def test_gs_failure_after_the_last_page(run_main):
    """The pages are sent, but the job fails with Ghostscript's status."""
    result = run_main(pages=1, mode="fail_end")
    assert result.returncode == 1
    assert result.stdout.endswith(_JOB_END)
    assert "GS: Error: /invalidfont in --findfont--" in result.log
    assert "Ghostscript exited with status 2" in result.log


def test_no_pages_from_gs_is_an_error(run_main):
    result = run_main(pages=0)
    assert result.returncode == 1
    assert result.stdout == b""
    assert "No pages produced by Ghostscript" in result.log


def test_gs_failure_mid_job_in_reverse_order_sends_only_complete_pages(run_main):
    options = f"{_OPTIONS} BRReverse=ON"
    complete = run_main(options=options, pages=1)
    assert complete.returncode == 0
    broken = run_main(options=options, pages=1, mode="truncate")
    assert broken.returncode == 1
    assert broken.stdout == complete.stdout
    assert "Ghostscript failed with exit status 3" in broken.log
    assert "while reading page 2 from Ghostscript" in broken.log


def test_broken_output_aborts_the_job_and_stops_gs(run_main, fake_gs, monkeypatch):
    """CUPS's backend went away: gs, still writing, gets the grace period and is killed."""

    class Broken(io.BytesIO):
        def write(self, data):
            raise BrokenPipeError

    monkeypatch.setattr(cups_filter, "GS_EXIT_GRACE", 0.1)
    result = run_main(pages=3, output=Broken())
    assert result.returncode == 1
    assert "job aborted" in result.log
    assert "BrokenPipeError" in result.log
    assert not _is_running(_gs_call(fake_gs)["pid"])


def test_interrupt_kills_gs(run_main, fake_gs):
    class Interrupted(io.BytesIO):
        def write(self, data):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_main(pages=3, output=Interrupted())
    assert not _is_running(_gs_call(fake_gs)["pid"])


def test_fine_resolution_prints_in_fine_mode(run_main):
    normal = run_main()
    fine = run_main(options=f"{_OPTIONS} BRResolution=600x2400dpi")
    assert fine.returncode == 0
    assert "resolution=Fine" in fine.log
    assert b"@PJL SET APTMODE=ON4" in fine.stdout
    assert b"@PJL SET APTMODE=OFF" in normal.stdout


def test_find_gs_tries_the_usual_prefixes(monkeypatch):
    """CUPS runs filters with a minimal PATH."""
    monkeypatch.setattr(cups_filter.shutil, "which", lambda _name: None)
    monkeypatch.setattr(Path, "is_file", lambda self: str(self) == "/usr/local/bin/gs")
    assert cups_filter.find_gs() == "/usr/local/bin/gs"


def test_missing_gs_exits(monkeypatch, caplog):
    monkeypatch.setattr(cups_filter.shutil, "which", lambda _name: None)
    monkeypatch.setattr(Path, "is_file", lambda _self: False)
    with pytest.raises(SystemExit) as exit_info:
        cups_filter.find_gs()
    assert exit_info.value.code == 1
    assert "Ghostscript (gs) not found in PATH" in caplog.text


def test_ppd_offers_normal_and_fine():
    ppd = (_FILTER.parent / "brhl4150cdn.ppd").read_text(encoding="latin-1")
    choices = [line for line in ppd.splitlines() if line.startswith("*BRResolution ")]
    assert choices == [
        '*BRResolution 600dpi/Normal (600 x 600dpi): \t"          "',
        '*BRResolution 600x2400dpi/Fine (2400 dpi class): \t"          "',
    ]


@pytest.mark.parametrize(("value", "expected"), [("2", 2), ("0", 0), ("-1", 0)])
def test_render_threads_from_environment(monkeypatch, value, expected):
    from brother_xl2hb.render import bands

    monkeypatch.setenv("BRXL2HB_RENDER_THREADS", value)
    assert bands.render_threads() == expected


def test_non_numeric_render_threads_fall_back_to_the_default(monkeypatch, caplog):
    from brother_xl2hb.render import bands

    monkeypatch.delenv("BRXL2HB_RENDER_THREADS", raising=False)
    default = bands.render_threads()
    monkeypatch.setenv("BRXL2HB_RENDER_THREADS", "four")
    assert bands.render_threads() == default
    assert "non-numeric BRXL2HB_RENDER_THREADS='four'" in caplog.text


def test_render_threads_from_the_old_variable(monkeypatch):
    from brother_xl2hb.render import bands

    monkeypatch.delenv("BRXL2HB_RENDER_THREADS", raising=False)
    monkeypatch.setenv("BRHL4150CDN_RENDER_THREADS", "2")
    assert bands.render_threads() == 2
    monkeypatch.setenv("BRXL2HB_RENDER_THREADS", "1")
    assert bands.render_threads() == 1


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

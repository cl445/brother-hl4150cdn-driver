"""End-to-end runs of the CUPS filter's main() against a fake Ghostscript.

The filter runs as CUPS runs it, as a subprocess with the CUPS argument
vector. A stand-in `gs` first on PATH writes PPM pages (or fails) as the
test asks and records how it was called.
"""

import io
import json
import os
import signal
import stat
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from brother_xl2hb.job import filter_pages
from brother_xl2hb.protocol.pjl import generate_pjl_footer
from brother_xl2hb.protocol.xl2hb import OP_END_SESSION
from brother_xl2hb.settings import PageSize, PrintSettings

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
    if "FAKE_GS_PID" in os.environ:
        with open(os.environ["FAKE_GS_PID"], "w") as pid:
            pid.write(str(os.getpid()))
    pages = int(os.environ.get("FAKE_GS_PAGES", "1"))
    page = 0
    while mode == "endless" or page < pages:
        if page and mode == "stray_newline":
            out.write(b"\\n")
        out.write(header)
        for _ in range(height):
            out.write(row)
        page += 1
    if mode == "linger":
        # Output done, but the process does not exit.
        out.flush()
        os.close(1)
        import time
        time.sleep(600)
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


def test_stray_whitespace_between_pages_keeps_the_rest_of_the_job(fake_gs):
    """A newline between two PPMs used to end the job after page 1 and hang on gs."""
    plain = _run(fake_gs, pages=2)
    stray = _run(fake_gs, pages=2, mode="stray_newline")
    assert stray.returncode == 0, stray.stderr.decode()
    assert stray.stdout == plain.stdout
    assert b"total pages: 2" in stray.stderr


def _wait_for(predicate, timeout=60.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.05)


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def test_sigterm_ends_the_job_after_the_last_complete_page(fake_gs, tmp_path):
    """CUPS cancels with SIGTERM: gs is killed and the session closed after whole pages."""
    out_path, err_path, pid_path = tmp_path / "out", tmp_path / "err", tmp_path / "gs.pid"
    job_path = tmp_path / "job.ps"
    job_path.write_bytes(_JOB)
    env = {**fake_gs, "FAKE_GS_MODE": "endless", "FAKE_GS_PID": str(pid_path)}
    argv = [sys.executable, str(_FILTER), "42", "alice", "My Title", "1", _OPTIONS, str(job_path)]
    with out_path.open("wb") as out, err_path.open("wb") as err:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=err, env=env)
        try:
            _wait_for(lambda: b"page 2:" in err_path.read_bytes())
            proc.send_signal(signal.SIGTERM)
            assert proc.wait(timeout=60) == 0
        finally:
            proc.kill()
            proc.wait()

    stderr = err_path.read_text()
    assert "INFO: brhl4150cdn-filter: job canceled" in stderr
    assert "Traceback" not in stderr
    assert _gone(int(pid_path.read_text()))

    output = out_path.read_bytes()
    assert output.endswith(bytes([OP_END_SESSION]) + generate_pjl_footer())
    assert output in [_run(fake_gs, pages=n).stdout for n in (1, 2, 3)]


def test_gs_that_does_not_exit_after_its_output_is_killed(fake_gs, tmp_path, monkeypatch, caplog):
    from brother_xl2hb import cups_filter

    for key, value in fake_gs.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("FAKE_GS_MODE", "linger")
    monkeypatch.setenv("FAKE_GS_PID", str(tmp_path / "gs.pid"))
    monkeypatch.setattr(cups_filter, "GS_EXIT_TIMEOUT", 0.5)
    job = tmp_path / "job.ps"
    job.write_bytes(_JOB)
    out = io.BytesIO()

    assert cups_filter.print_job(str(job), PrintSettings(page_size=PageSize.POSTCARD), out) == 1
    assert "Ghostscript did not exit after its output ended" in caplog.text
    assert out.getvalue().endswith(bytes([OP_END_SESSION]) + generate_pjl_footer())
    assert _gone(int((tmp_path / "gs.pid").read_text()))


class _SignalingOutput(io.BytesIO):
    """Gets SIGTERM in the middle of its first write."""

    def write(self, data):
        written = super().write(data[:2])
        if self.tell() == 2:
            os.kill(os.getpid(), signal.SIGTERM)
        return written + super().write(data[2:])


def test_sigterm_during_a_write_cancels_once_the_write_is_done():
    from brother_xl2hb.cups_filter import CancelOnSigterm, JobCanceledError

    previous = signal.getsignal(signal.SIGTERM)
    target = _SignalingOutput()
    with CancelOnSigterm(target) as output:
        with pytest.raises(JobCanceledError):
            output.write(b"whole page")
        assert target.getvalue() == b"whole page"
        # Canceled once: the shutdown writes go through, a second SIGTERM is ignored.
        os.kill(os.getpid(), signal.SIGTERM)
        output.write(b" footer")
    assert target.getvalue() == b"whole page footer"
    assert signal.getsignal(signal.SIGTERM) == previous


def test_sigterm_between_writes_cancels_at_once():
    from brother_xl2hb.cups_filter import CancelOnSigterm, JobCanceledError

    def sigterm_while_rendering():
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(5)

    with CancelOnSigterm(io.BytesIO()), pytest.raises(JobCanceledError):
        sigterm_while_rendering()


def test_reader_thread_ends_when_the_output_breaks():
    from brother_xl2hb import cups_filter
    from brother_xl2hb.models import HL4150CDN

    class Broken(io.BytesIO):
        def write(self, data):
            raise BrokenPipeError

    w_pt, h_pt = HL4150CDN.paper_points["Postcard"]
    width, height = round(w_pt * 600 / 72), round(h_pt * 600 / 72)
    page = b"P6\n%d %d\n255\n" % (width, height) + bytes(width * height * 3)
    before = set(threading.enumerate())
    with pytest.raises(BrokenPipeError):
        cups_filter.send_pages(io.BytesIO(page * 2), PrintSettings(page_size=PageSize.POSTCARD), Broken(), HL4150CDN)
    for reader in set(threading.enumerate()) - before:
        reader.join(timeout=10)
        assert not reader.is_alive()

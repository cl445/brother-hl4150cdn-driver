"""Tests for the command-line entry point (brother_xl2hb.cli)."""

import io
import sys

import pytest

from brother_xl2hb import cli
from brother_xl2hb.render.pipeline import filter_pages
from brother_xl2hb.settings import PrintSettings
from fixture_utils import assert_matches_fixture


class _Stdio:
    """Stand-in for sys.stdin / sys.stdout with a binary `buffer`."""

    def __init__(self, data: bytes = b"") -> None:
        self.buffer = io.BytesIO(data)


def _ppm(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    return f"P6\n{width} {height}\n255\n".encode() + bytes(rgb) * (width * height)


def _run_cli(monkeypatch, argv: list[str], stdin: bytes) -> bytes:
    stdout = _Stdio()
    monkeypatch.setattr(sys, "stdin", _Stdio(stdin))
    monkeypatch.setattr(sys, "stdout", stdout)
    cli.main(argv)
    return stdout.buffer.getvalue()


def test_cli_renders_every_page_of_the_stream(monkeypatch):
    """A concatenated PPM stream becomes one session with all its pages."""
    pages = [(8, 4, (0, 0, 0)), (8, 4, (255, 0, 0)), (8, 4, (0, 255, 255))]
    stream = b"".join(_ppm(w, h, rgb) for w, h, rgb in pages)

    out = _run_cli(monkeypatch, [], stream)

    expected = io.BytesIO()
    filter_pages(
        [(w, h, bytes(rgb) * (w * h)) for w, h, rgb in pages],
        PrintSettings(),
        expected,
    )
    assert out == expected.getvalue()


def test_cli_single_page_is_unchanged(monkeypatch):
    out = _run_cli(monkeypatch, [], _ppm(8, 4, (0, 0, 0)))
    expected = io.BytesIO()
    filter_pages([(8, 4, bytes(3) * 32)], PrintSettings(), expected)
    assert out == expected.getvalue()


def test_cli_empty_stdin_fails(monkeypatch):
    with pytest.raises(SystemExit) as excinfo:
        _run_cli(monkeypatch, [], b"")
    assert excinfo.value.code == 1


def test_cli_missing_rc_file_fails(monkeypatch, tmp_path, capsys):
    missing = tmp_path / "no-such-rc"
    with pytest.raises(SystemExit) as excinfo:
        _run_cli(monkeypatch, ["--rc", str(missing)], _ppm(8, 4, (0, 0, 0)))
    assert excinfo.value.code != 0
    assert f"RC file not found: {missing}" in capsys.readouterr().err


def test_cli_reads_existing_rc_file(monkeypatch, tmp_path):
    rc = tmp_path / "brhl4150cdnrc"
    rc.write_text("[hl4150cdn]\nBRMonoColor=Mono\n")
    out = _run_cli(monkeypatch, ["--rc", str(rc)], _ppm(8, 4, (0, 0, 0)))
    expected = io.BytesIO()
    filter_pages([(8, 4, bytes(3) * 32)], PrintSettings.from_rc_file(str(rc)), expected)
    assert out == expected.getvalue()
    default = io.BytesIO()
    filter_pages([(8, 4, bytes(3) * 32)], PrintSettings(), default)
    assert out != default.getvalue()


def test_cli_fine_help_says_mode_is_not_verified_on_the_printer(monkeypatch, capsys):
    monkeypatch.setenv("COLUMNS", "500")  # no line wrapping inside words
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    assert "not verified on the printer" in help_text


def test_cli_skips_blank_pages_in_duplex(monkeypatch, tmp_path):
    """Brother's filter drops blank pages in duplex too; compared with its capture."""
    from test_full_pipeline import _duplex_test_pages

    a, b, c, _ = _duplex_test_pages()
    white = (a[0], a[1], b"\xff" * len(a[2]))
    stream = b"".join(b"P6\n%d %d\n255\n" % (w, h) + data for w, h, data in (a, white, b, c))
    rc = tmp_path / "brhl4150cdnrc"
    rc.write_text("[hl4150cdn]\nBRSkipBlank=ON\n")
    out = _run_cli(monkeypatch, ["--rc", str(rc), "--duplex", "long"], stream)
    assert_matches_fixture("skip_blank_long_edge", out)


def test_cli_prints_nothing_when_every_page_is_blank(monkeypatch, tmp_path):
    rc = tmp_path / "brhl4150cdnrc"
    rc.write_text("[hl4150cdn]\nBRSkipBlank=ON\nDuplex=DuplexNoTumble\n")
    assert _run_cli(monkeypatch, ["--rc", str(rc)], _ppm(8, 4, (255, 255, 255)) * 3) == b""

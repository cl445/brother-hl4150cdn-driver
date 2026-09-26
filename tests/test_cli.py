"""Tests for the command-line entry point (src/cli.py)."""

import io
import sys

import pytest

import cli
from pipeline import filter_pages
from settings import PrintSettings


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


def test_cli_fine_help_marks_mode_experimental(monkeypatch, capsys):
    monkeypatch.setenv("COLUMNS", "500")  # no line wrapping inside words
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    assert "experimental" in help_text
    assert "not byte-identical" in help_text

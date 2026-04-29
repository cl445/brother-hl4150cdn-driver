"""
PPM reader tests.

Tests P6 binary PPM parsing used as input to the filter pipeline.
"""

import io

import pytest

from brfilter import read_ppm


def _make_ppm(width: int, height: int, maxval: int = 255, pixels: bytes | None = None) -> bytes:
    """Build a minimal P6 PPM binary."""
    header = f"P6\n{width} {height}\n{maxval}\n".encode("ascii")
    if pixels is None:
        pixels = bytes(width * height * 3)
    return header + pixels


class TestPPMReaderValid:
    def test_basic_1x1_black(self):
        data = _make_ppm(1, 1, pixels=bytes(3))
        result = read_ppm(io.BytesIO(data))
        assert result is not None
        w, h, m, pix = result
        assert (w, h, m) == (1, 1, 255)
        assert pix == bytes(3)

    def test_basic_1x1_white(self):
        data = _make_ppm(1, 1, pixels=bytes([255, 255, 255]))
        result = read_ppm(io.BytesIO(data))
        assert result is not None
        _w, _h, _m, pix = result
        assert pix == bytes([255, 255, 255])

    def test_small_image(self):
        pixels = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255])
        data = _make_ppm(2, 2, pixels=pixels)
        result = read_ppm(io.BytesIO(data))
        assert result is not None
        w, h, _m, pix = result
        assert (w, h) == (2, 2)
        assert pix == pixels

    def test_maxval_65535_rejected(self):
        """16-bit maxval should be rejected (only 8-bit PPM supported)."""
        header = b"P6\n1 1\n65535\n"
        pixels = bytes(3)
        data = header + pixels
        with pytest.raises(ValueError, match="maxval=65535"):
            read_ppm(io.BytesIO(data))

    def test_comments_in_header(self):
        header = b"P6\n# This is a comment\n2 2\n255\n"
        pixels = bytes(2 * 2 * 3)
        data = header + pixels
        result = read_ppm(io.BytesIO(data))
        assert result is not None
        w, h, _m, _pix = result
        assert (w, h) == (2, 2)

    def test_dimensions_on_separate_lines(self):
        """Width and height on separate lines."""
        header = b"P6\n4\n3\n255\n"
        pixels = bytes(4 * 3 * 3)
        data = header + pixels
        result = read_ppm(io.BytesIO(data))
        assert result is not None
        w, h, _m, _pix = result
        assert (w, h) == (4, 3)

    def test_printer_resolution_image(self):
        """4760x10 — verify dimensions at printer width."""
        width, height = 4760, 10
        pixels = bytes(width * height * 3)
        header = f"P6\n{width} {height}\n255\n".encode("ascii")
        data = header + pixels
        result = read_ppm(io.BytesIO(data))
        assert result is not None
        w, h, _m, _pix = result
        assert (w, h) == (width, height)


class TestPPMReaderErrors:
    def test_wrong_magic(self):
        with pytest.raises(ValueError, match="Not PPM P6"):
            read_ppm(io.BytesIO(b"P5\n1 1\n255\n\x00"))

    def test_incomplete_data(self):
        header = b"P6\n10 10\n255\n"
        data = header + bytes(10)  # way too short for 10x10
        with pytest.raises(ValueError, match="incomplete"):
            read_ppm(io.BytesIO(data))

    def test_empty_input(self):
        assert read_ppm(io.BytesIO(b"")) is None

    def test_whitespace_only_input(self):
        assert read_ppm(io.BytesIO(b"\n")) is None


class TestPPMReaderConcatenated:
    """Test reading concatenated PPM streams (e.g. from Ghostscript)."""

    def test_two_pages(self):
        ppm1 = _make_ppm(2, 2, pixels=bytes([255, 0, 0] * 4))
        ppm2 = _make_ppm(2, 2, pixels=bytes([0, 255, 0] * 4))
        stream = io.BytesIO(ppm1 + ppm2)

        result1 = read_ppm(stream)
        assert result1 is not None
        w, h, _m, pix1 = result1
        assert (w, h) == (2, 2)
        assert pix1 == bytes([255, 0, 0] * 4)

        result2 = read_ppm(stream)
        assert result2 is not None
        _, _, _, pix2 = result2
        assert pix2 == bytes([0, 255, 0] * 4)

        # EOF after second page
        assert read_ppm(stream) is None

    def test_three_pages_eof(self):
        pages = [_make_ppm(1, 1, pixels=bytes([i, i, i])) for i in range(3)]
        stream = io.BytesIO(b"".join(pages))

        for i in range(3):
            result = read_ppm(stream)
            assert result is not None
            _, _, _, pix = result
            assert pix == bytes([i, i, i])

        assert read_ppm(stream) is None

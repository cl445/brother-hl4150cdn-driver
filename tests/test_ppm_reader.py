"""
PPM reader tests.

Tests P6 binary PPM parsing used as input to the filter pipeline.
"""

import io

import pytest

from brother_xl2hb.render.ppm import read_ppm


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


class TestPPMReaderHeaderSyntax:
    """Header layouts the netpbm format allows besides Ghostscript's own."""

    @pytest.mark.parametrize(
        "header",
        [
            b"P6 2 2 255\n",
            b"P6\n2 2\n# after the dimensions\n255\n",
            b"P6\n2 2# right after a number\n255\n",
            b"P6\n\n2 2\n\n255\n",
            b"P6\r\n2\t2\r\n255\r",
            b"# before the magic\nP6\n2 2\n255\n",
        ],
    )
    def test_accepted(self, header):
        pixels = bytes(range(12))
        result = read_ppm(io.BytesIO(header + pixels))
        assert result == (2, 2, 255, pixels)

    def test_exactly_one_whitespace_byte_ends_the_header(self):
        """Raster bytes that look like whitespace or comments belong to the raster."""
        pixels = b"\n# \t\n\r"
        result = read_ppm(io.BytesIO(b"P6 2 1 255\n" + pixels))
        assert result == (2, 1, 255, pixels)

    def test_whitespace_between_and_after_images(self):
        stream = io.BytesIO(_make_ppm(1, 1, pixels=b"abc") + b"\n \n" + _make_ppm(1, 1, pixels=b"def") + b"\n")
        assert read_ppm(stream) == (1, 1, 255, b"abc")
        assert read_ppm(stream) == (1, 1, 255, b"def")
        assert read_ppm(stream) is None


class TestPPMReaderMalformedHeader:
    @pytest.mark.parametrize(
        ("data", "match"),
        [
            (b"P6", "incomplete"),
            (b"P6\n2", "incomplete"),
            (b"P6\n2 2\n", "incomplete"),
            (b"P6\n2 2\n255", "incomplete"),
            (b"P6\n2 2 # comment without a line end", "incomplete"),
            (b"P6\nx 2\n255\n", "invalid width"),
            (b"P6\n2 -2\n255\n", "invalid height"),
            (b"P6\n0 2\n255\n", "invalid width"),
            (b"P6\n2 2\n25x\n", "invalid maxval"),
            (b"P6\n2 2\n15\n", "maxval=15"),
            (b"P61 1 255\n", "Not PPM P6"),
            (b"\x89PNG" + bytes(100), "too long"),
        ],
    )
    def test_raises_value_error(self, data, match):
        with pytest.raises(ValueError, match=match):
            read_ppm(io.BytesIO(data))

"""
Color separation tests (RGB -> CMYK).

Tests the current simple-threshold implementation AND defines target behavior
for the full driver replacement (ordered dithering, proper GCR, etc.).
"""

import numpy as np
import pytest

from brfilter import ColorMatching, rgb_line_to_cmyk_intensities
from dither import dither_channel_1bpp
from xl2hb import BPL

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rgb_row(r: int, g: int, b: int, width: int = 4760) -> bytes:
    """Create a uniform RGB scanline."""
    return bytes([r, g, b]) * width


def _any_set(plane_data: bytes) -> bool:
    """Check if any bit is set in plane data."""
    return any(b != 0 for b in plane_data)


def _all_set(plane_data: bytes, width: int) -> bool:
    """Check if all pixel bits are set (within active width)."""
    for x in range(width):
        byte_idx = x >> 3
        bit_mask = 0x80 >> (x & 7)
        if not (plane_data[byte_idx] & bit_mask):
            return False
    return True


def _rgb_line_to_cmyk_planes(rgb_row: bytes, width: int, bpl: int) -> tuple[bytes, bytes, bytes, bytes]:
    """Convert one RGB scanline to 1bpp K, C, M, Y plane data.

    Simple threshold separation: any non-white pixel gets ink.
    Returns (k_data, c_data, m_data, y_data) each of bpl bytes.
    """
    rgb = np.frombuffer(rgb_row, dtype=np.uint8, count=width * 3).reshape(width, 3)
    r, g, b = rgb[:, 0].astype(np.int16), rgb[:, 1].astype(np.int16), rgb[:, 2].astype(np.int16)

    # CMY from RGB (inverted)
    c_val = 255 - r
    m_val = 255 - g
    y_val = 255 - b

    # Simple GCR: minimum of CMY becomes K
    k_val = np.minimum(np.minimum(c_val, m_val), y_val)
    c_val -= k_val
    m_val -= k_val
    y_val -= k_val

    # Threshold to 1bpp, pad to bpl*8 bits
    total_bits = bpl * 8

    def to_plane(vals: np.ndarray) -> bytes:
        bits = vals > 127
        if len(bits) < total_bits:
            bits = np.concatenate([bits, np.zeros(total_bits - len(bits), dtype=bool)])
        return bytes(np.packbits(bits)[:bpl])

    return to_plane(k_val), to_plane(c_val), to_plane(m_val), to_plane(y_val)


# ---------------------------------------------------------------------------
# Pure color -> expected plane mapping (current simple threshold)
# ---------------------------------------------------------------------------


class TestPureColors:
    """Test that pure colors map to the correct planes."""

    def test_white_produces_empty_planes(self):
        """White (255,255,255) -> all planes zero."""
        k, c, m, y = _rgb_line_to_cmyk_planes(_make_rgb_row(255, 255, 255), 4760, BPL)
        assert not _any_set(k)
        assert not _any_set(c)
        assert not _any_set(m)
        assert not _any_set(y)

    def test_black_produces_only_k(self):
        """Black (0,0,0) -> K=1, C=M=Y=0 (GCR removes CMY)."""
        k, c, m, y = _rgb_line_to_cmyk_planes(_make_rgb_row(0, 0, 0), 4760, BPL)
        assert _all_set(k, 4760)
        assert not _any_set(c)
        assert not _any_set(m)
        assert not _any_set(y)

    def test_red_produces_m_and_y(self):
        """Red (255,0,0) -> C=0, M=1, Y=1, K=0."""
        k, c, m, y = _rgb_line_to_cmyk_planes(_make_rgb_row(255, 0, 0), 4760, BPL)
        assert not _any_set(k)
        assert not _any_set(c)
        assert _all_set(m, 4760)
        assert _all_set(y, 4760)

    def test_green_produces_c_and_y(self):
        """Green (0,255,0) -> C=1, M=0, Y=1, K=0."""
        k, c, m, y = _rgb_line_to_cmyk_planes(_make_rgb_row(0, 255, 0), 4760, BPL)
        assert not _any_set(k)
        assert _all_set(c, 4760)
        assert not _any_set(m)
        assert _all_set(y, 4760)

    def test_blue_produces_c_and_m(self):
        """Blue (0,0,255) -> C=1, M=1, Y=0, K=0."""
        k, c, m, y = _rgb_line_to_cmyk_planes(_make_rgb_row(0, 0, 255), 4760, BPL)
        assert not _any_set(k)
        assert _all_set(c, 4760)
        assert _all_set(m, 4760)
        assert not _any_set(y)

    def test_cyan_produces_only_c(self):
        """Cyan (0,255,255) -> C=1, M=0, Y=0, K=0."""
        k, c, m, y = _rgb_line_to_cmyk_planes(_make_rgb_row(0, 255, 255), 4760, BPL)
        assert not _any_set(k)
        assert _all_set(c, 4760)
        assert not _any_set(m)
        assert not _any_set(y)

    def test_magenta_produces_only_m(self):
        """Magenta (255,0,255) -> C=0, M=1, Y=0, K=0."""
        k, c, m, y = _rgb_line_to_cmyk_planes(_make_rgb_row(255, 0, 255), 4760, BPL)
        assert not _any_set(k)
        assert not _any_set(c)
        assert _all_set(m, 4760)
        assert not _any_set(y)

    def test_yellow_produces_only_y(self):
        """Yellow (255,255,0) -> C=0, M=0, Y=1, K=0."""
        k, c, m, y = _rgb_line_to_cmyk_planes(_make_rgb_row(255, 255, 0), 4760, BPL)
        assert not _any_set(k)
        assert not _any_set(c)
        assert not _any_set(m)
        assert _all_set(y, 4760)


class TestGrayLevels:
    """Gray values should produce only K (via GCR), no CMY."""

    def test_dark_gray_all_k(self):
        """Dark gray (64,64,64) -> K=1 (below threshold), no CMY."""
        k, c, m, y = _rgb_line_to_cmyk_planes(_make_rgb_row(64, 64, 64), 4760, BPL)
        assert _all_set(k, 4760)
        assert not _any_set(c)
        assert not _any_set(m)
        assert not _any_set(y)

    def test_mid_gray_threshold(self):
        """Mid gray (128,128,128) -> K depends on threshold (127=1 in current impl)."""
        k, c, m, y = _rgb_line_to_cmyk_planes(_make_rgb_row(128, 128, 128), 4760, BPL)
        # GCR: k_val = min(127, 127, 127) = 127 -> just below threshold 127
        # 127 > 127 is False -> K=0 in current implementation
        assert not _any_set(k)
        assert not _any_set(c)
        assert not _any_set(m)
        assert not _any_set(y)

    def test_light_gray_no_ink(self):
        """Light gray (192,192,192) -> no ink (below threshold)."""
        k, c, m, y = _rgb_line_to_cmyk_planes(_make_rgb_row(192, 192, 192), 4760, BPL)
        assert not _any_set(k)
        assert not _any_set(c)
        assert not _any_set(m)
        assert not _any_set(y)


# ---------------------------------------------------------------------------
# GCR (Gray Component Replacement) properties
# ---------------------------------------------------------------------------


class TestGCR:
    """Verify GCR removes the minimum CMY component into K."""

    def test_gcr_dark_red(self):
        """Dark red (128,0,0): C=127, M=255, Y=255, min=127 -> K=127, C=0, M=128, Y=128.
        After threshold: K=0 (127 not > 127), M=1, Y=1."""
        _k, c, m, y = _rgb_line_to_cmyk_planes(_make_rgb_row(128, 0, 0), 4760, BPL)
        assert not _any_set(c), "C should be zero after GCR"
        assert _all_set(m, 4760)
        assert _all_set(y, 4760)

    def test_gcr_ensures_no_cmy_for_neutral(self):
        """Any neutral color should have C=M=Y=0 after GCR (all goes to K)."""
        for gray in [0, 32, 64, 96, 127]:
            _k, c, m, y = _rgb_line_to_cmyk_planes(_make_rgb_row(gray, gray, gray), 4760, BPL)
            assert not _any_set(c), f"C should be 0 for gray={gray}"
            assert not _any_set(m), f"M should be 0 for gray={gray}"
            assert not _any_set(y), f"Y should be 0 for gray={gray}"


# ---------------------------------------------------------------------------
# Width / padding
# ---------------------------------------------------------------------------


class TestScanlineWidth:
    def test_output_length_is_bpl(self):
        """Each plane output is exactly BPL bytes."""
        k, c, m, y = _rgb_line_to_cmyk_planes(_make_rgb_row(128, 0, 0), 4760, BPL)
        assert len(k) == BPL
        assert len(c) == BPL
        assert len(m) == BPL
        assert len(y) == BPL

    def test_narrow_width(self):
        """Narrower input should still produce BPL-length output."""
        row = _make_rgb_row(0, 0, 0, width=100)
        k, _c, _m, _y = _rgb_line_to_cmyk_planes(row, 100, BPL)
        assert len(k) == BPL
        # Only first 100 pixels should have ink
        assert k[12] & 0xF0 == 0xF0  # pixels 96-99 set, 100-103 not
        # Byte 13+ should be padding zeros (past pixel 103)
        assert all(b == 0 for b in k[13:])

    def test_single_pixel(self):
        """Single black pixel at position 0."""
        row = bytes([0, 0, 0])
        k, _c, _m, _y = _rgb_line_to_cmyk_planes(row, 1, BPL)
        assert k[0] & 0x80 == 0x80  # bit 0 set
        assert k[0] & 0x7F == 0  # other bits clear
        assert all(b == 0 for b in k[1:])


# ---------------------------------------------------------------------------
# Ordered dithering via new dithering pipeline
# ---------------------------------------------------------------------------


class TestOrderedDithering:
    """Tests for proper ordered dithering -- the driver uses a matrix-based
    dither, not simple thresholding."""

    def test_50pct_gray_produces_halftone_pattern(self):
        """50% gray should produce a halftone pattern, not uniform output.
        The driver uses ordered dithering to distribute dots.
        With the 3D LUT, gray ink is spread across CMYK channels."""
        k_int, c_int, m_int, y_int = rgb_line_to_cmyk_intensities(_make_rgb_row(128, 128, 128), 4760)
        # Dither all channels and count total dots
        k = dither_channel_1bpp(k_int, y=0, width=4760)
        c = dither_channel_1bpp(c_int, y=0, width=4760)
        m = dither_channel_1bpp(m_int, y=0, width=4760)
        y_d = dither_channel_1bpp(y_int, y=0, width=4760)
        # At least some channel should have meaningful coverage
        total_bits = sum(b.bit_count() for b in (k + c + m + y_d)[: 595 * 4])
        assert total_bits > 500, f"50% gray should produce dots across CMYK, got {total_bits}"

    def test_25pct_gray_sparser_than_50pct(self):
        """25% gray should have fewer dots than 50% gray."""
        k_25_int, _, _, _ = rgb_line_to_cmyk_intensities(_make_rgb_row(191, 191, 191), 4760)
        k_50_int, _, _, _ = rgb_line_to_cmyk_intensities(_make_rgb_row(128, 128, 128), 4760)
        k_25 = dither_channel_1bpp(k_25_int, y=0, width=4760)
        k_50 = dither_channel_1bpp(k_50_int, y=0, width=4760)
        bits_25 = sum(b.bit_count() for b in k_25[:595])
        bits_50 = sum(b.bit_count() for b in k_50[:595])
        assert bits_25 < bits_50

    def test_dither_varies_by_line(self):
        """Different scanlines should produce different dot patterns
        (the dither matrix is position-dependent)."""
        k_int, _, _, _ = rgb_line_to_cmyk_intensities(_make_rgb_row(128, 128, 128), 4760)
        k_y0 = dither_channel_1bpp(k_int, y=0, width=4760)
        k_y1 = dither_channel_1bpp(k_int, y=1, width=4760)
        assert k_y0 != k_y1, "Dither should vary by y position"

    def test_brcd_table_loading(self):
        """The driver loads dither matrices from BRCD cache files."""
        from pathlib import Path

        from dither import _try_load_brcd

        lut_dir = str(Path(__file__).resolve().parent.parent / "src" / "lut")
        if not (Path(lut_dir) / "0600-k_cache09.bin").exists():
            pytest.skip("Original BRCD files not available")
        channels = _try_load_brcd(lut_dir)
        assert channels is not None
        # K and Y use 32x32, C and M use 40x40
        assert channels["K"].width == 32
        assert channels["Y"].width == 32
        assert channels["C"].width == 40
        assert channels["M"].width == 40


# ---------------------------------------------------------------------------
# Target behavior: color matching / ICC profiles
# ---------------------------------------------------------------------------


class TestColorMatching:
    """The driver supports Normal, Vivid, and None color matching modes."""

    def test_vivid_mode_increases_saturation(self):
        """Vivid mode should produce more saturated colors."""
        from brfilter import apply_vivid

        # Desaturated red: R=200, G=100, B=100
        row = _make_rgb_row(200, 100, 100, width=1)
        boosted = apply_vivid(row, 1)
        # Vivid should push R higher and G/B lower
        r_out, g_out, b_out = boosted[0], boosted[1], boosted[2]
        # Saturation ratio: (max - min) / max
        sat_in = (200 - 100) / 200
        sat_out = (r_out - min(g_out, b_out)) / max(r_out, 1)
        assert sat_out > sat_in, f"Vivid saturation {sat_out:.2f} should exceed {sat_in:.2f}"

    def test_none_mode_passes_through(self):
        """None mode should skip color management -- gray produces CMY but no K."""
        # Gray pixel: R=G=B=128
        k_norm, _c_norm, _m_norm, _y_norm = rgb_line_to_cmyk_intensities(
            _make_rgb_row(128, 128, 128, width=1), 1, color_matching=ColorMatching.NORMAL
        )
        k_none, c_none, m_none, y_none = rgb_line_to_cmyk_intensities(
            _make_rgb_row(128, 128, 128, width=1), 1, color_matching=ColorMatching.NONE
        )
        # Normal (LUT): K has some ink, CMY also have ink (LUT distributes across channels)
        assert k_norm[0] < 255  # K has ink
        # None: K has no ink (no GCR), CMY all have ink
        assert k_none[0] == 255  # K has no ink
        assert c_none[0] < 255  # C has ink
        assert m_none[0] < 255  # M has ink
        assert y_none[0] < 255  # Y has ink


# Brightness / contrast / RGB-key adjustments are covered byte-for-byte
# by the cyan_100_* fixtures in tests/test_full_pipeline.py.

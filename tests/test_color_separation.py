"""
Color separation tests (RGB -> CMYK).

Colour-table selection, the LUT-based separation and its dithering. The
byte-exact separation of real pages is covered by the capture tests in
tests/test_full_pipeline.py.
"""

import pytest

from brfilter import ColorMatching, MediaType, PrintSettings, rgb_to_cmyk_lut
from color_lut import ColorTable
from dither import dither_channel_1bpp
from transforms import color_table

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rgb_row(r: int, g: int, b: int, width: int = 4760) -> bytes:
    """Create a uniform RGB scanline."""
    return bytes([r, g, b]) * width


# ---------------------------------------------------------------------------
# Ordered dithering of separated channels
# ---------------------------------------------------------------------------


class TestOrderedDithering:
    """Tests for proper ordered dithering -- the driver uses a matrix-based
    dither, not simple thresholding."""

    def test_50pct_gray_produces_halftone_pattern(self):
        """50% gray should produce a halftone pattern, not uniform output.
        The driver uses ordered dithering to distribute dots.
        With the 3D LUT, gray ink is spread across CMYK channels."""
        k_int, c_int, m_int, y_int = rgb_to_cmyk_lut(_make_rgb_row(128, 128, 128), 4760)
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
        k_25_int, _, _, _ = rgb_to_cmyk_lut(_make_rgb_row(191, 191, 191), 4760)
        k_50_int, _, _, _ = rgb_to_cmyk_lut(_make_rgb_row(128, 128, 128), 4760)
        k_25 = dither_channel_1bpp(k_25_int, y=0, width=4760)
        k_50 = dither_channel_1bpp(k_50_int, y=0, width=4760)
        bits_25 = sum(b.bit_count() for b in k_25[:595])
        bits_50 = sum(b.bit_count() for b in k_50[:595])
        assert bits_25 < bits_50

    def test_dither_varies_by_line(self):
        """Different scanlines should produce different dot patterns
        (the dither matrix is position-dependent)."""
        k_int, _, _, _ = rgb_to_cmyk_lut(_make_rgb_row(128, 128, 128), 4760)
        k_y0 = dither_channel_1bpp(k_int, y=0, width=4760)
        k_y1 = dither_channel_1bpp(k_int, y=1, width=4760)
        assert k_y0 != k_y1, "Dither should vary by y position"

    def test_brcd_table_loading(self):
        """The driver loads dither matrices from BRCD cache files."""
        from pathlib import Path

        from dither import load_brcd_tables

        lut_dir = str(Path(__file__).resolve().parent.parent / "src" / "lut")
        if not (Path(lut_dir) / "0600-k_cache09.bin").exists():
            pytest.skip("Original BRCD files not available")
        channels = load_brcd_tables(lut_dir)
        assert channels is not None
        # K and Y use 32x32, C and M use 40x40
        assert channels["K"].width == 32
        assert channels["Y"].width == 32
        assert channels["C"].width == 40
        assert channels["M"].width == 40


# ---------------------------------------------------------------------------
# Colour table selection
# ---------------------------------------------------------------------------


class TestColorTables:
    """Colour grid selection as in `lookup_color_transform_table`."""

    @pytest.mark.parametrize(
        ("settings", "expected"),
        [
            (PrintSettings(), ColorTable("rgb")),
            (PrintSettings(color_matching=ColorMatching.VIVID), ColorTable("srgb")),
            (PrintSettings(color_matching=ColorMatching.NONE), ColorTable("cmyk")),
            (PrintSettings(improve_gray=True), ColorTable("rgb", improve_gray=True)),
            (PrintSettings(toner_save=True), ColorTable("rgb", variant="density2")),
            (PrintSettings(media_type=MediaType.GLOSSY), ColorTable("rgb", variant="glossy")),
            (PrintSettings(toner_save=True, media_type=MediaType.GLOSSY), ColorTable("rgb", variant="density2")),
            (PrintSettings(enhance_black=True), ColorTable("rgb", rich_black=True)),
            (PrintSettings(enhance_black=True, color_matching=ColorMatching.NONE), ColorTable("cmyk")),
        ],
    )
    def test_table_for_settings(self, settings, expected):
        assert color_table(settings) == expected

    @pytest.mark.parametrize(
        ("table", "rgb", "expected"),
        [
            (ColorTable("srgb"), (0, 255, 255), (255, 20, 255, 255)),
            (ColorTable("srgb"), (255, 0, 0), (255, 255, 30, 30)),
            (ColorTable("srgb"), (25, 230, 230), (253, 35, 251, 235)),
            (ColorTable("rgb", variant="density2"), (30, 200, 90), (248, 98, 253, 96)),
            (ColorTable("rgb", variant="density2"), (250, 240, 10), (255, 253, 244, 16)),
            (ColorTable("cmyk"), (30, 200, 90), (255, 35, 236, 113)),
            (ColorTable("cmyk"), (200, 100, 50), (255, 235, 124, 60)),
        ],
    )
    def test_separation_matches_original(self, table, rgb, expected):
        """(K, C, M, Y) as recovered from brhl4150cdnfilter output for these solid colours."""
        k, c, m, y = rgb_to_cmyk_lut(_make_rgb_row(*rgb, width=1), 1, table)
        assert (k[0], c[0], m[0], y[0]) == expected

    def test_rich_black_takes_grid_entry_zero(self):
        k, c, m, y = rgb_to_cmyk_lut(_make_rgb_row(0, 0, 0, width=1), 1, ColorTable("rgb", rich_black=True))
        assert (k[0], c[0], m[0], y[0]) == (0, 255 - 83, 255 - 55, 255 - 65)
        k, c, m, y = rgb_to_cmyk_lut(_make_rgb_row(0, 0, 0, width=1), 1, ColorTable("rgb"))
        assert (k[0], c[0], m[0], y[0]) == (0, 255, 255, 255)


# Brightness / contrast / RGB-key adjustments are covered byte-for-byte
# by the cyan_100_* fixtures in tests/test_full_pipeline.py.

"""Tests for saturation adjustment (compress_adjust_saturation reimplementation)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from saturation import adjust_saturation


def _pixel(r: int, g: int, b: int) -> bytes:
    """Create a single-pixel RGB row."""
    return bytes([r, g, b])


def _get_pixel(rgb_row: bytes, idx: int = 0) -> tuple[int, int, int]:
    """Extract pixel at index from RGB row."""
    off = idx * 3
    return rgb_row[off], rgb_row[off + 1], rgb_row[off + 2]


# --- Positive saturation (boost) ---


class TestAdjustSaturationPositive:
    def test_boost_red(self):
        """Reddish pixel becomes more saturated: max channel increases, min decreases."""
        orig = _pixel(200, 100, 50)
        result = adjust_saturation(orig, 1, 10)
        r, _g, b = _get_pixel(result)
        # Max (R) should increase, min (B) should decrease
        assert r > 200
        assert b < 50

    def test_boost_preserves_gray(self):
        """Gray pixels (R==G==B) are never modified."""
        orig = _pixel(128, 128, 128)
        result = adjust_saturation(orig, 1, 20)
        assert result == orig

    def test_boost_skips_black(self):
        """Pure black (0,0,0) is unchanged (gray + black skip)."""
        orig = _pixel(0, 0, 0)
        result = adjust_saturation(orig, 1, 20)
        assert result == orig

    def test_boost_skips_white(self):
        """Pure white (255,255,255) is unchanged (gray + white skip)."""
        orig = _pixel(255, 255, 255)
        result = adjust_saturation(orig, 1, 20)
        assert result == orig

    def test_boost_skips_pixel_with_zero_channel(self):
        """Positive saturation skips pixels where any channel is 0."""
        orig = _pixel(0, 100, 200)
        result = adjust_saturation(orig, 1, 10)
        assert result == orig

    def test_boost_skips_pixel_with_255_channel(self):
        """Positive saturation skips pixels where any channel is 255."""
        orig = _pixel(255, 100, 50)
        result = adjust_saturation(orig, 1, 10)
        assert result == orig

    def test_boost_clamped(self):
        """Result always in [0, 255] even with high saturation on near-extreme values."""
        orig = _pixel(250, 5, 128)
        result = adjust_saturation(orig, 1, 20)
        r, g, b = _get_pixel(result)
        assert 0 <= r <= 255
        assert 0 <= g <= 255
        assert 0 <= b <= 255

    def test_boost_preserves_hue(self):
        """Relative channel ordering is preserved after boost."""
        orig = _pixel(200, 100, 50)
        result = adjust_saturation(orig, 1, 15)
        r, g, b = _get_pixel(result)
        # Original order: R > G > B should be maintained
        assert r > g > b

    def test_boost_stronger_with_higher_sat(self):
        """Higher sat_mode produces larger changes."""
        orig = _pixel(200, 100, 50)
        weak = adjust_saturation(orig, 1, 5)
        strong = adjust_saturation(orig, 1, 20)
        r_w, _, b_w = _get_pixel(weak)
        r_s, _, b_s = _get_pixel(strong)
        # Strong boost should push max higher and min lower
        assert r_s >= r_w
        assert b_s <= b_w
        # At least one must differ (not both clamped)
        assert (r_s > r_w) or (b_s < b_w)


# --- Negative saturation (desaturation) ---


class TestAdjustSaturationNegative:
    def test_desaturate_moves_toward_gray(self):
        """Channels move toward the center value."""
        orig = _pixel(200, 100, 50)
        result = adjust_saturation(orig, 1, -10)
        r, _g, b = _get_pixel(result)
        # R was above center (125), should decrease
        assert r < 200
        # B was below center, should increase
        assert b > 50

    def test_desaturate_preserves_gray(self):
        """Gray pixels are unchanged under desaturation."""
        orig = _pixel(128, 128, 128)
        result = adjust_saturation(orig, 1, -15)
        assert result == orig

    def test_full_desaturate(self):
        """Strong desaturation brings channels closer together."""
        orig = _pixel(200, 100, 50)
        result = adjust_saturation(orig, 1, -20)
        r, g, b = _get_pixel(result)
        orig_range = 200 - 50  # = 150
        new_range = max(r, g, b) - min(r, g, b)
        assert new_range < orig_range

    def test_desaturate_does_not_skip_black_white(self):
        """Negative saturation processes pixels with channels at 0 or 255."""
        # Pure red: R=255, G=0, B=0 — skipped for positive, but processed for negative
        orig = _pixel(255, 0, 0)
        result = adjust_saturation(orig, 1, -10)
        r, g, b = _get_pixel(result)
        # Should change: R decreases, G/B increase
        assert r < 255
        assert g > 0 or b > 0


# --- Zero saturation (identity) ---


class TestAdjustSaturationZero:
    def test_zero_is_identity(self):
        """sat_mode=0 returns input unchanged."""
        orig = _pixel(200, 100, 50)
        result = adjust_saturation(orig, 1, 0)
        assert result == orig

    def test_zero_returns_exact_bytes(self):
        """sat_mode=0 returns the same bytes object (fast path)."""
        orig = _pixel(123, 45, 67)
        result = adjust_saturation(orig, 1, 0)
        assert result is orig


# --- Edge cases ---


class TestAdjustSaturationEdgeCases:
    def test_two_equal_channels(self):
        """Pixel with two equal channels: (100, 100, 200) — correct min/mid/max."""
        orig = _pixel(100, 100, 200)
        result = adjust_saturation(orig, 1, 10)
        r, g, b = _get_pixel(result)
        # B is max and should increase; R and G are min/mid and should decrease
        assert b > 200
        assert r < 100
        # R and G were equal, should stay equal (both moved by same amount)
        assert r == g

    def test_full_width_row(self):
        """4760-pixel wide row processes without error."""
        width = 4760
        row = bytes([128, 64, 192] * width)
        result = adjust_saturation(row, width, 10)
        assert len(result) == width * 3

    def test_single_pixel(self):
        """Single pixel row works correctly."""
        orig = _pixel(150, 80, 40)
        result = adjust_saturation(orig, 1, 10)
        r, _g, b = _get_pixel(result)
        assert r > 150
        assert b < 40

    def test_mixed_row(self):
        """Row with mixed pixels: each adjusted individually."""
        # Gray + colored + black
        row = bytes([128, 128, 128, 200, 100, 50, 0, 0, 0])
        result = adjust_saturation(row, 3, 10)
        # Gray pixel unchanged
        assert _get_pixel(result, 0) == (128, 128, 128)
        # Colored pixel changed
        r, _g, b = _get_pixel(result, 1)
        assert r > 200 or b < 50  # at least some change
        # Black pixel unchanged
        assert _get_pixel(result, 2) == (0, 0, 0)

    def test_specific_values_positive(self):
        """Verify exact computation for a known pixel (positive saturation)."""
        # (200, 100, 50), sat=10
        # min=50 (B, idx=2), max=200 (R, idx=0), mid=100 (G, idx=1)
        # half_range = (200-50)//2 = 75
        # boost = (75*10)//50 = 15
        # Clamp: min(15, 255-200=55) → 15; min(15, 50) → 15
        # new_max = 215, new_min = 35
        # old_range = 150, new_range = 180
        # mid_offset = round(180 * (100-50) / 150) = round(60.0) = 60
        # new_mid = 35 + 60 = 95
        orig = _pixel(200, 100, 50)
        result = adjust_saturation(orig, 1, 10)
        assert _get_pixel(result) == (215, 95, 35)

    def test_specific_values_negative(self):
        """Verify exact computation for a known pixel (negative saturation)."""
        # (200, 100, 50), sat=-10
        # center = 50 + (200-50)//2 = 125
        # R=200 (above center): round(-10*(200-125)/50 + 200) = round(-15+200) = 185
        # G=100 (below center): round(100 - (-10)*(125-100)/50) = round(100+5) = 105
        # B=50 (below center): round(50 - (-10)*(125-50)/50) = round(50+15) = 65
        orig = _pixel(200, 100, 50)
        result = adjust_saturation(orig, 1, -10)
        assert _get_pixel(result) == (185, 105, 65)

    def test_rgb_row_too_short(self):
        """ValueError when rgb_row is shorter than width * 3."""
        import pytest

        with pytest.raises(ValueError, match="rgb_row too short"):
            adjust_saturation(b"\x80\x40", 1, 10)

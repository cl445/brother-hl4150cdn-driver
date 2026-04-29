"""
Tests for tone curve utilities (gamma loading, spline interpolation, tone curve builder).
"""

import numpy as np
import pytest

from tone_curve import (
    apply_tone_curve,
    build_tone_curve,
    cubic_spline_interpolate,
    load_gamma_curve,
)

# ---------------------------------------------------------------------------
# TestGammaCurveLoading
# ---------------------------------------------------------------------------


class TestGammaCurveLoading:
    """Tests for load_gamma_curve()."""

    def test_load_curve_0(self):
        curve = load_gamma_curve(0)
        assert curve.shape == (256,)
        assert curve.dtype == np.uint8

    def test_load_curve_1(self):
        curve = load_gamma_curve(1)
        assert curve.shape == (256,)
        assert curve.dtype == np.uint8

    def test_curve_endpoints(self):
        """Both gamma curves map 0→0 and 255→255."""
        for cid in (0, 1):
            curve = load_gamma_curve(cid)
            assert curve[0] == 0, f"curve {cid}: index 0 should map to 0"
            assert curve[255] == 255, f"curve {cid}: index 255 should map to 255"

    def test_curve_monotonic(self):
        """Gamma curves should be monotonically non-decreasing."""
        for cid in (0, 1):
            curve = load_gamma_curve(cid)
            for i in range(1, 256):
                assert curve[i] >= curve[i - 1], (
                    f"curve {cid}: not monotonic at index {i}: {curve[i - 1]} -> {curve[i]}"
                )

    def test_curve_0_vs_1_differ(self):
        """The two gamma curves should produce different values."""
        c0 = load_gamma_curve(0)
        c1 = load_gamma_curve(1)
        assert not np.array_equal(c0, c1)

    def test_curve_nonlinear(self):
        """Gamma curves should not be identity (some mid-values differ)."""
        for cid in (0, 1):
            curve = load_gamma_curve(cid)
            identity = np.arange(256, dtype=np.uint8)
            assert not np.array_equal(curve, identity), f"curve {cid} is identity — expected gamma correction"

    def test_invalid_curve_id(self):
        """Loading a non-existent curve should raise."""
        with pytest.raises((FileNotFoundError, ValueError)):
            load_gamma_curve(99)


# ---------------------------------------------------------------------------
# TestCubicSplineInterpolate
# ---------------------------------------------------------------------------


class TestCubicSplineInterpolate:
    """Tests for cubic_spline_interpolate()."""

    def test_two_points_linear(self):
        """Two control points [0, 255] → linear ramp."""
        cp = np.array([0.0, 255.0], dtype=np.float64)
        result = cubic_spline_interpolate(cp, output_count=256)
        assert len(result) == 256
        # Should be close to identity
        expected = np.linspace(0, 255, 256)
        np.testing.assert_allclose(result, expected, atol=1.0)

    def test_passes_through_control_points(self):
        """Output at knot x-positions should approximate input values."""
        cp = np.array([0.0, 50.0, 200.0, 255.0], dtype=np.float64)
        result = cubic_spline_interpolate(cp, output_count=256)
        n = len(cp)
        for i, val in enumerate(cp):
            x = round(255.0 * i / (n - 1))
            assert abs(result[x] - val) < 2.0, f"at x={x}: expected ~{val}, got {result[x]}"

    def test_monotonic_input_monotonic_output(self):
        """Strictly increasing control points → non-decreasing output."""
        cp = np.array([0.0, 30.0, 100.0, 180.0, 255.0], dtype=np.float64)
        result = cubic_spline_interpolate(cp, output_count=256)
        for i in range(1, len(result)):
            assert result[i] >= result[i - 1] - 0.5, (
                f"output not monotonic at index {i}: {result[i - 1]:.2f} -> {result[i]:.2f}"
            )

    def test_clamped_to_range(self):
        """All output values should be in [0, output_count-1]."""
        cp = np.array([0.0, 300.0, -50.0, 255.0], dtype=np.float64)
        result = cubic_spline_interpolate(cp, output_count=256)
        assert np.all(result >= 0.0)
        assert np.all(result <= 255.0)

    def test_identity_like(self):
        """Evenly spaced [0, 85, 170, 255] → near-identity curve."""
        cp = np.array([0.0, 85.0, 170.0, 255.0], dtype=np.float64)
        result = cubic_spline_interpolate(cp, output_count=256)
        identity = np.arange(256, dtype=np.float64)
        np.testing.assert_allclose(result, identity, atol=2.0)

    def test_smooth_curve(self):
        """5+ control points → no jumps > threshold between adjacent entries."""
        cp = np.array([0.0, 20.0, 80.0, 180.0, 230.0, 255.0], dtype=np.float64)
        result = cubic_spline_interpolate(cp, output_count=256)
        diffs = np.abs(np.diff(result))
        assert np.max(diffs) < 10.0, f"max adjacent jump = {np.max(diffs):.2f}"

    def test_many_control_points(self):
        """17 control points (matches driver's usage) works correctly."""
        cp = np.linspace(0, 255, 17)
        result = cubic_spline_interpolate(cp, output_count=256)
        assert len(result) == 256
        # Should be near-identity
        identity = np.arange(256, dtype=np.float64)
        np.testing.assert_allclose(result, identity, atol=2.0)


# ---------------------------------------------------------------------------
# TestBuildToneCurve
# ---------------------------------------------------------------------------


class TestBuildToneCurve:
    """Tests for build_tone_curve()."""

    def test_no_adjustments_no_gamma_is_identity(self):
        """All params zero/None → identity LUT."""
        lut = build_tone_curve(brightness=0, contrast=0, gamma_select=None)
        assert lut.shape == (256,)
        assert lut.dtype == np.uint8
        expected = np.arange(256, dtype=np.uint8)
        np.testing.assert_array_equal(lut, expected)

    def test_no_adjustments_with_gamma(self):
        """gamma_select=0 with no brightness/contrast → equals gamma curve 0."""
        lut = build_tone_curve(brightness=0, contrast=0, gamma_select=0)
        gamma = load_gamma_curve(0)
        np.testing.assert_array_equal(lut, gamma)

    def test_brightness_positive_lightens(self):
        """Positive brightness → higher values (less ink)."""
        lut = build_tone_curve(brightness=10, contrast=0, gamma_select=None)
        identity = np.arange(256, dtype=np.uint8)
        # Mid-range values should increase
        assert lut[128] > identity[128]

    def test_brightness_negative_darkens(self):
        """Negative brightness → lower values (more ink)."""
        lut = build_tone_curve(brightness=-10, contrast=0, gamma_select=None)
        identity = np.arange(256, dtype=np.uint8)
        # Mid-range values should decrease
        assert lut[128] < identity[128]

    def test_contrast_expands_range(self):
        """Positive contrast → midtones stay, extremes diverge."""
        lut = build_tone_curve(brightness=0, contrast=20, gamma_select=None)
        # Values below 128 should get darker (lower)
        assert lut[64] < 64
        # Values above 128 should get lighter (higher)
        assert lut[192] > 192

    def test_gamma_applied_after_contrast(self):
        """Verify composition order: brightness → contrast → gamma."""
        # Build with known brightness and gamma
        lut_with_gamma = build_tone_curve(brightness=5, contrast=10, gamma_select=0)
        lut_no_gamma = build_tone_curve(brightness=5, contrast=10, gamma_select=None)
        gamma = load_gamma_curve(0)
        # The gamma-applied LUT should equal gamma[no-gamma-lut]
        expected = gamma[lut_no_gamma]
        np.testing.assert_array_equal(lut_with_gamma, expected)

    def test_endpoints_clamped(self):
        """Output should always be in [0, 255] regardless of settings."""
        for b in (-20, -10, 0, 10, 20):
            for c in (-20, -10, 0, 10, 20):
                lut = build_tone_curve(brightness=b, contrast=c, gamma_select=None)
                assert lut.min() >= 0
                assert lut.max() <= 255


# ---------------------------------------------------------------------------
# TestApplyToneCurve
# ---------------------------------------------------------------------------


class TestApplyToneCurve:
    """Tests for apply_tone_curve()."""

    def test_identity_curve_no_change(self):
        """Identity LUT should preserve input."""
        identity = np.arange(256, dtype=np.uint8)
        data = bytes(range(256))
        k, c, m, y = apply_tone_curve(data, data, data, data, identity)
        assert k == data
        assert c == data
        assert m == data
        assert y == data

    def test_invert_curve(self):
        """Reversed LUT should invert values."""
        invert = np.arange(255, -1, -1, dtype=np.uint8)
        data = bytes(range(256))
        k, c, _m, _y = apply_tone_curve(data, data, data, data, invert)
        expected = bytes(range(255, -1, -1))
        assert k == expected
        assert c == expected

    def test_four_channels_independent(self):
        """Each channel should be processed independently with the same curve."""
        lut = build_tone_curve(brightness=5, contrast=0, gamma_select=None)
        k_in = bytes([0, 50, 100, 200])
        c_in = bytes([10, 60, 110, 210])
        m_in = bytes([20, 70, 120, 220])
        y_in = bytes([30, 80, 130, 230])
        k_out, c_out, m_out, y_out = apply_tone_curve(k_in, c_in, m_in, y_in, lut)
        # Each output byte should equal lut[input_byte]
        for i in range(4):
            assert k_out[i] == lut[k_in[i]]
            assert c_out[i] == lut[c_in[i]]
            assert m_out[i] == lut[m_in[i]]
            assert y_out[i] == lut[y_in[i]]

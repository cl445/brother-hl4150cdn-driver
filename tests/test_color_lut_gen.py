"""Tests for parametric color data generation.

Validates that generated LUT, interpolation tables, and gamma curves
match the original binary data within acceptable tolerances.
"""

import io
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from color_lut_gen import (
    generate_gamma_curve,
    generate_interp_tables,
    generate_rgb_default_lut,
    generate_srgb_default_lut,
)

# Data directory containing original binary tables
_DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "color_data"


# ---------------------------------------------------------------------------
# Original data loaders (for comparison)
# ---------------------------------------------------------------------------
def _load_original_lut() -> np.ndarray:
    """Load the original binary LUT as (4913, 4) int32 [C, M, Y, K]."""
    data = (_DATA_DIR / "rgb_default_lut.bin").read_bytes()
    packed = np.frombuffer(data, dtype=np.int32).reshape(-1, 2)
    unpacked = np.empty((4913, 4), dtype=np.int32)
    unpacked[:, 0] = packed[:, 0] & 0xFFFF
    unpacked[:, 1] = packed[:, 0] >> 16
    unpacked[:, 2] = packed[:, 1] & 0xFFFF
    unpacked[:, 3] = packed[:, 1] >> 16
    return unpacked


# ---------------------------------------------------------------------------
# LUT generation tests
# ---------------------------------------------------------------------------
class TestLUTGeneration:
    """Test that generated LUT matches original within tolerances."""

    @pytest.fixture
    def original_lut(self):
        return _load_original_lut()

    @pytest.fixture
    def generated_lut(self):
        return generate_rgb_default_lut()

    def test_shape(self, generated_lut):
        assert generated_lut.shape == (4913, 4)

    def test_dtype(self, generated_lut):
        assert generated_lut.dtype == np.int32

    def test_value_range(self, generated_lut):
        assert generated_lut.min() >= 0
        assert generated_lut.max() <= 255

    @pytest.mark.parametrize(("ch", "name"), [(0, "C"), (1, "M"), (2, "Y"), (3, "K")])
    def test_mean_error_per_channel(self, original_lut, generated_lut, ch, name):
        """Mean error per channel should be well below 3."""
        err = np.abs(generated_lut[:, ch] - original_lut[:, ch])
        assert err.mean() < 3, f"{name}: mean_err={err.mean():.2f}"

    @pytest.mark.parametrize(("ch", "name"), [(0, "C"), (1, "M"), (2, "Y"), (3, "K")])
    def test_p95_error_per_channel(self, original_lut, generated_lut, ch, name):
        """95th percentile error should be below 8."""
        err = np.abs(generated_lut[:, ch] - original_lut[:, ch])
        p95 = np.percentile(err, 95)
        assert p95 < 8, f"{name}: p95={p95:.1f}"

    def test_white_point(self, original_lut, generated_lut):
        """White point (R=G=B=255) should be close to original [0,0,0,0]."""
        white_idx = 16 * 289 + 16 * 17 + 16  # = 4912
        err = np.abs(generated_lut[white_idx] - original_lut[white_idx])
        assert err.max() <= 5, f"White point error: {err.tolist()}"

    def test_black_point(self, original_lut, generated_lut):
        """Black point (R=G=B=0) should be close to original.

        The black point is a known singularity in the parametric fit
        (transition to pure-K black), so higher error is acceptable.
        """
        err = np.abs(generated_lut[0] - original_lut[0])
        assert err.max() <= 20, f"Black point error: {err.tolist()}"

    def test_neutral_axis_k_dominant(self, generated_lut):
        """Along neutral axis (R=G=B), K should be the dominant channel."""
        for i in range(1, 16):
            idx = i * 289 + i * 17 + i  # r=g=b=i
            cmyk = generated_lut[idx]
            # For dark neutrals, K should be significant
            # Note: original LUT has K=0 starting at grid point 10
            if i < 10:
                assert cmyk[3] > 0, f"Grid ({i},{i},{i}): K should be > 0"


# ---------------------------------------------------------------------------
# Interpolation table tests
# ---------------------------------------------------------------------------
class TestInterpTables:
    """Test that generated interpolation tables match original exactly."""

    def test_exact_match(self):
        """Generated tables should match original byte-for-byte."""
        original = np.frombuffer((_DATA_DIR / "interp_tables.bin").read_bytes(), dtype=np.uint8).reshape(17, 17 * 17, 9)
        generated = generate_interp_tables()
        assert generated.shape == (17, 289, 9)
        np.testing.assert_array_equal(generated, original)

    def test_total_weight_positive(self):
        """All total weights should be positive."""
        tables = generate_interp_tables()
        assert np.all(tables[:, :, 0] > 0)

    def test_weights_sum_to_total(self):
        """Corner weights should sum to the total weight."""
        tables = generate_interp_tables()
        totals = tables[:, :, 0].astype(np.int32)
        sums = tables[:, :, 1:9].astype(np.int32).sum(axis=2)
        np.testing.assert_array_equal(totals, sums)

    def test_corner_weights_only_four_nonzero(self):
        """Tetrahedral interpolation uses at most 4 non-zero weights."""
        tables = generate_interp_tables()
        for b_f in range(17):
            for row in range(289):
                weights = tables[b_f, row, 1:9]
                nonzero = np.count_nonzero(weights)
                assert nonzero <= 4, (
                    f"b={b_f} row={row}: {nonzero} non-zero weights (tetrahedral should have at most 4)"
                )


# ---------------------------------------------------------------------------
# Gamma curve tests
# ---------------------------------------------------------------------------
class TestGammaCurves:
    """Test that generated gamma curves match originals exactly."""

    @pytest.mark.parametrize("curve_id", [0, 1])
    def test_exact_match(self, curve_id):
        """Generated curve should match original byte-for-byte."""
        original = np.frombuffer((_DATA_DIR / f"gamma_curve_{curve_id}.bin").read_bytes(), dtype=np.uint8)
        generated = generate_gamma_curve(curve_id)
        assert generated.shape == (256,)
        assert generated.dtype == np.uint8
        np.testing.assert_array_equal(generated, original)

    def test_invalid_curve_id(self):
        with pytest.raises(ValueError, match="Invalid gamma curve_id"):
            generate_gamma_curve(2)

    @pytest.mark.parametrize("curve_id", [0, 1])
    def test_monotonic(self, curve_id):
        """Gamma curves should be monotonically non-decreasing."""
        curve = generate_gamma_curve(curve_id)
        diffs = np.diff(curve.astype(np.int16))
        assert np.all(diffs >= 0), "Gamma curve is not monotonic"

    @pytest.mark.parametrize("curve_id", [0, 1])
    def test_endpoints(self, curve_id):
        """Gamma curve should map 0->0 and 255->255."""
        curve = generate_gamma_curve(curve_id)
        assert curve[0] == 0
        assert curve[255] == 255


# ---------------------------------------------------------------------------
# sRGB LUT generation tests
# ---------------------------------------------------------------------------
def _load_original_srgb_lut() -> np.ndarray:
    """Load the original sRGB binary LUT as (4913, 4) int32 [C, M, Y, K]."""
    data = (_DATA_DIR / "srgb_default_lut.bin").read_bytes()
    packed = np.frombuffer(data, dtype=np.int32).reshape(-1, 2)
    unpacked = np.empty((4913, 4), dtype=np.int32)
    unpacked[:, 0] = packed[:, 0] & 0xFFFF
    unpacked[:, 1] = packed[:, 0] >> 16
    unpacked[:, 2] = packed[:, 1] & 0xFFFF
    unpacked[:, 3] = packed[:, 1] >> 16
    return unpacked


class TestSRGBLUTGeneration:
    """Test that generated sRGB LUT matches original within tolerances."""

    @pytest.fixture
    def original_lut(self):
        return _load_original_srgb_lut()

    @pytest.fixture
    def generated_lut(self):
        return generate_srgb_default_lut()

    def test_shape(self, generated_lut):
        assert generated_lut.shape == (4913, 4)

    def test_dtype(self, generated_lut):
        assert generated_lut.dtype == np.int32

    def test_value_range(self, generated_lut):
        assert generated_lut.min() >= 0
        assert generated_lut.max() <= 255

    @pytest.mark.parametrize(("ch", "name"), [(0, "C"), (1, "M"), (2, "Y"), (3, "K")])
    def test_mean_error_per_channel(self, original_lut, generated_lut, ch, name):
        """Mean error per channel should be well below 3."""
        err = np.abs(generated_lut[:, ch] - original_lut[:, ch])
        assert err.mean() < 3, f"{name}: mean_err={err.mean():.2f}"

    @pytest.mark.parametrize(("ch", "name"), [(0, "C"), (1, "M"), (2, "Y"), (3, "K")])
    def test_p95_error_per_channel(self, original_lut, generated_lut, ch, name):
        """95th percentile error should be below 8."""
        err = np.abs(generated_lut[:, ch] - original_lut[:, ch])
        p95 = np.percentile(err, 95)
        assert p95 < 8, f"{name}: p95={p95:.1f}"

    @pytest.mark.parametrize(("ch", "name"), [(0, "C"), (1, "M"), (2, "Y"), (3, "K")])
    def test_max_error_per_channel(self, original_lut, generated_lut, ch, name):
        """Max error per channel should not exceed 5."""
        err = np.abs(generated_lut[:, ch] - original_lut[:, ch])
        assert err.max() <= 5, f"{name}: max_err={err.max()}"

    def test_white_point(self, original_lut, generated_lut):
        """White point (R=G=B=255) should be [0,0,0,0]."""
        white_idx = 4912
        err = np.abs(generated_lut[white_idx] - original_lut[white_idx])
        assert err.max() <= 5, f"White point error: {err.tolist()}"

    def test_black_point(self, original_lut, generated_lut):
        """Black point (R=G=B=0) should be close to original."""
        err = np.abs(generated_lut[0] - original_lut[0])
        assert err.max() <= 20, f"Black point error: {err.tolist()}"


# ---------------------------------------------------------------------------
# Fallback integration tests
# ---------------------------------------------------------------------------
class TestFallbackIntegration:
    """Test that the fallback mechanism works when binary files are missing."""

    def test_color_lut_fallback(self):
        """color_lut module should fall back to generation when binaries missing."""
        import color_lut

        # Clear the LRU cache
        color_lut._load_data.cache_clear()

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch.object(color_lut, "_DATA_DIR", Path(tmpdir)),
        ):
            lut, interp = color_lut._load_data()
            assert lut.shape == (4913, 4)
            assert interp.shape == (17, 289, 9)

        # Restore cache
        color_lut._load_data.cache_clear()

    def test_tone_curve_fallback(self):
        """tone_curve module should fall back to generation when binaries missing."""
        import tone_curve

        # Clear cache
        tone_curve._gamma_cache.clear()

        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(tone_curve, "_DATA_DIR", Path(tmpdir)):
            curve = tone_curve.load_gamma_curve(0)
            assert curve.shape == (256,)
            assert curve.dtype == np.uint8

        # Restore cache
        tone_curve._gamma_cache.clear()

    def test_pipeline_without_binaries(self):
        """Full pipeline should work without binary data files."""
        import color_lut
        import tone_curve

        color_lut._load_data.cache_clear()
        tone_curve._gamma_cache.clear()

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch.object(color_lut, "_DATA_DIR", Path(tmpdir)),
            mock.patch.object(tone_curve, "_DATA_DIR", Path(tmpdir)),
        ):
            from brfilter import PrintSettings, filter_page

            w, h = 100, 10
            pixels = bytes([128, 0, 0]) * (w * h)  # Red pixels
            out = io.BytesIO()
            filter_page(w, h, pixels, PrintSettings(), out)
            result = out.getvalue()
            # Should produce valid XL2HB output
            assert result.startswith(b"\x1b%-12345X")
            assert b") BROTHER XL2HB" in result

        color_lut._load_data.cache_clear()
        tone_curve._gamma_cache.clear()

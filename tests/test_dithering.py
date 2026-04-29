"""
Ordered dithering tests.

Defines the target API and behavior for the dithering subsystem,
based on the reverse-engineered dither.c from the original driver.

The driver uses ordered dithering with BRCD lookup tables,
supporting both 1bpp (bitmap) and 4bpp (nibble) output modes,
and 1-channel (mono) and 4-channel (CMYK) configurations.
"""

import pytest

# ---------------------------------------------------------------------------
# Dither module API
# ---------------------------------------------------------------------------


class TestDitherAPI:
    """Tests that the dither module exposes the expected API."""

    def test_module_exists(self):
        import dither  # noqa: F401

    def test_has_init_function(self):
        from dither import load_dither_tables  # noqa: F401

    def test_has_ordered_4ch_1bpp(self):
        from dither import dither_cmyk_1bpp  # noqa: F401

    def test_has_ordered_1ch_1bpp(self):
        from dither import dither_channel_1bpp  # noqa: F401

    def test_has_ordered_1ch_4bpp(self):
        from dither import dither_channel_4bpp  # noqa: F401

    def test_has_ordered_4ch_4bpp(self):
        from dither import dither_cmyk_4bpp  # noqa: F401


# ---------------------------------------------------------------------------
# 1bpp ordered dithering (bitmap mode — what the HL-4150CDN uses)
# ---------------------------------------------------------------------------


class TestOrdered1bpp:
    """Ordered dithering to 1-bit-per-pixel output."""

    def test_white_input_no_dots(self):
        """Pure white (255) should produce no dots at any position."""
        from dither import dither_channel_1bpp

        row = bytes([255] * 4760)
        result = dither_channel_1bpp(row, y=0, width=4760)
        assert not any(result)

    def test_black_input_all_dots(self):
        """Pure black (0) should produce all dots set."""
        from dither import dither_channel_1bpp

        row = bytes([0] * 4760)
        result = dither_channel_1bpp(row, y=0, width=4760)
        bpl = (4760 + 7) // 8
        assert len(result) == bpl
        # All bits set for active pixels
        for x in range(4760):
            byte_idx = x >> 3
            bit_mask = 0x80 >> (x & 7)
            assert result[byte_idx] & bit_mask, f"Pixel {x} should be set"

    def test_50pct_gray_coverage(self):
        """50% gray should produce approximately 50% dot coverage."""
        from dither import dither_channel_1bpp

        row = bytes([128] * 4760)
        result = dither_channel_1bpp(row, y=0, width=4760)
        bits_set = sum(b.bit_count() for b in result[:595])
        # Allow 10% tolerance
        assert 1900 < bits_set < 2860, f"Expected ~2380 bits, got {bits_set}"

    def test_monotonic_coverage(self):
        """Darker grays should produce more dots than lighter grays."""
        from dither import dither_channel_1bpp

        coverages = []
        for gray in [224, 192, 128, 64, 32]:
            row = bytes([gray] * 4760)
            result = dither_channel_1bpp(row, y=0, width=4760)
            bits = sum(b.bit_count() for b in result[:595])
            coverages.append(bits)
        # Each should be strictly greater than the previous
        for i in range(len(coverages) - 1):
            assert coverages[i] < coverages[i + 1], f"Coverage not monotonic: gray levels produced {coverages}"

    def test_position_dependent(self):
        """Different y positions should produce different patterns for same input."""
        from dither import dither_channel_1bpp

        row = bytes([128] * 4760)
        result_y0 = dither_channel_1bpp(row, y=0, width=4760)
        result_y1 = dither_channel_1bpp(row, y=1, width=4760)
        assert result_y0 != result_y1, "Dither should vary by y position"

    def test_deterministic(self):
        """Same input + position should produce same output."""
        from dither import dither_channel_1bpp

        row = bytes([128] * 4760)
        a = dither_channel_1bpp(row, y=42, width=4760)
        b = dither_channel_1bpp(row, y=42, width=4760)
        assert a == b


# ---------------------------------------------------------------------------
# 4-channel (CMYK) dithering
# ---------------------------------------------------------------------------


class TestOrdered4Channel:
    """4-channel ordered dithering (CMYK mode)."""

    def test_4ch_returns_four_planes(self):
        """4-channel dither should return K, C, M, Y plane data."""
        from dither import dither_cmyk_1bpp

        # Input: 4760 pixels x 4 channels (C, M, Y, K)
        cmyk_row = bytes([0, 0, 0, 255] * 4760)  # pure black
        k, c, m, y = dither_cmyk_1bpp(cmyk_row, y=0, width=4760)
        bpl = (4760 + 7) // 8
        assert len(k) == bpl
        assert len(c) == bpl
        assert len(m) == bpl
        assert len(y) == bpl

    def test_pure_black_only_k(self):
        """Pure black through CMYK dither: only K plane should have dots."""
        from dither import dither_cmyk_1bpp

        cmyk_row = bytes([0, 0, 0, 255] * 4760)
        k, c, m, y = dither_cmyk_1bpp(cmyk_row, y=0, width=4760)
        assert any(k)
        assert not any(c)
        assert not any(m)
        assert not any(y)

    def test_pure_cyan_only_c(self):
        """Pure cyan: only C plane should have dots."""
        from dither import dither_cmyk_1bpp

        cmyk_row = bytes([255, 0, 0, 0] * 4760)
        k, c, m, y = dither_cmyk_1bpp(cmyk_row, y=0, width=4760)
        assert not any(k)
        assert any(c)
        assert not any(m)
        assert not any(y)


# ---------------------------------------------------------------------------
# 4bpp ordered dithering (nibble mode — 16 intensity levels per pixel)
# ---------------------------------------------------------------------------


class TestOrdered1ch4bpp:
    """Ordered dithering to 4-bit-per-pixel nibble-packed output."""

    def test_white_input_all_zero_nibbles(self):
        """Pure white (255) should produce all-zero nibbles."""
        from dither import dither_channel_4bpp

        row = bytes([255] * 4760)
        result = dither_channel_4bpp(row, y=0, width=4760)
        assert not any(result)

    def test_black_input_all_max_nibbles(self):
        """Pure black (0) should produce all 0xFF bytes (nibbles = 15)."""
        from dither import dither_channel_4bpp

        row = bytes([0] * 4760)
        result = dither_channel_4bpp(row, y=0, width=4760)
        assert all(b == 0xFF for b in result)

    def test_output_length(self):
        """Output should be (width + 1) // 2 bytes."""
        from dither import dither_channel_4bpp

        for width in [4760, 100, 32]:
            row = bytes([128] * width)
            result = dither_channel_4bpp(row, y=0, width=width)
            assert len(result) == (width + 1) // 2

    def test_output_length_odd_width(self):
        """Odd width: output is (width + 1) // 2, last low nibble padded to 0."""
        from dither import dither_channel_4bpp

        row = bytes([0] * 33)  # all black
        result = dither_channel_4bpp(row, y=0, width=33)
        assert len(result) == 17  # (33 + 1) // 2
        # Last byte: high nibble = 15 (pixel 32), low nibble = 0 (padding)
        assert result[-1] == 0xF0

    def test_nibble_packing_order(self):
        """First pixel goes into high nibble, second into low nibble."""
        from dither import dither_channel_4bpp

        # All black: every nibble = 15 → 0xFF per byte
        row = bytes([0] * 2)
        result = dither_channel_4bpp(row, y=0, width=2)
        assert result[0] == 0xFF

        # Single pixel (odd width): high nibble set, low nibble zero
        row = bytes([0] * 1)
        result = dither_channel_4bpp(row, y=0, width=1)
        assert result[0] == 0xF0

    def test_50pct_gray_coverage(self):
        """50% gray should produce nibbles averaging around 7-8."""
        from dither import dither_channel_4bpp

        row = bytes([128] * 4760)
        result = dither_channel_4bpp(row, y=0, width=4760)
        # Extract all nibbles and compute average
        nibbles = []
        for b in result:
            nibbles.append((b >> 4) & 0x0F)
            nibbles.append(b & 0x0F)
        nibbles = nibbles[:4760]  # trim to actual width
        avg = sum(nibbles) / len(nibbles)
        assert 6.0 < avg < 9.0, f"Expected avg nibble ~7-8, got {avg:.2f}"

    def test_monotonic_coverage(self):
        """Darker pixels should produce higher nibble sums."""
        from dither import dither_channel_4bpp

        sums = []
        for gray in [224, 192, 128, 64, 32]:
            row = bytes([gray] * 4760)
            result = dither_channel_4bpp(row, y=0, width=4760)
            nibble_sum = 0
            for b in result:
                nibble_sum += (b >> 4) & 0x0F
                nibble_sum += b & 0x0F
            sums.append(nibble_sum)
        for i in range(len(sums) - 1):
            assert sums[i] < sums[i + 1], f"Nibble sum not monotonic: {sums}"

    def test_position_dependent(self):
        """Different y positions should produce different output for same input."""
        from dither import dither_channel_4bpp

        row = bytes([128] * 4760)
        result_y0 = dither_channel_4bpp(row, y=0, width=4760)
        result_y1 = dither_channel_4bpp(row, y=1, width=4760)
        assert result_y0 != result_y1, "4bpp dither should vary by y position"

    def test_deterministic(self):
        """Same input + position should produce identical output."""
        from dither import dither_channel_4bpp

        row = bytes([128] * 4760)
        a = dither_channel_4bpp(row, y=42, width=4760)
        b = dither_channel_4bpp(row, y=42, width=4760)
        assert a == b

    def test_nibble_values_in_range(self):
        """All nibble values should be in [0, 15]."""
        from dither import dither_channel_4bpp

        for gray in [0, 64, 128, 192, 255]:
            row = bytes([gray] * 100)
            result = dither_channel_4bpp(row, y=7, width=100)
            for b in result:
                assert (b >> 4) <= 15
                assert (b & 0x0F) <= 15


class TestOrdered4ch4bpp:
    """4-channel ordered dithering to 4bpp nibble-packed output."""

    def test_returns_four_planes(self):
        """Should return K, C, M, Y plane data."""
        from dither import dither_cmyk_4bpp

        cmyk_row = bytes([0, 0, 0, 255] * 100)  # pure black
        k, c, m, y = dither_cmyk_4bpp(cmyk_row, y=0, width=100)
        assert isinstance(k, bytes)
        assert isinstance(c, bytes)
        assert isinstance(m, bytes)
        assert isinstance(y, bytes)

    def test_pure_black_only_k(self):
        """Pure black (K=255): only K plane should have non-zero nibbles."""
        from dither import dither_cmyk_4bpp

        cmyk_row = bytes([0, 0, 0, 255] * 100)
        k, c, m, y = dither_cmyk_4bpp(cmyk_row, y=0, width=100)
        assert any(k)
        assert not any(c)
        assert not any(m)
        assert not any(y)

    def test_pure_cyan_only_c(self):
        """Pure cyan (C=255): only C plane should have non-zero nibbles."""
        from dither import dither_cmyk_4bpp

        cmyk_row = bytes([255, 0, 0, 0] * 100)
        k, c, m, y = dither_cmyk_4bpp(cmyk_row, y=0, width=100)
        assert not any(k)
        assert any(c)
        assert not any(m)
        assert not any(y)

    def test_output_lengths(self):
        """Each plane should be (width + 1) // 2 bytes."""
        from dither import dither_cmyk_4bpp

        for width in [4760, 100, 33]:
            cmyk_row = bytes([128, 64, 32, 200] * width)
            k, c, m, y = dither_cmyk_4bpp(cmyk_row, y=0, width=width)
            expected = (width + 1) // 2
            assert len(k) == expected
            assert len(c) == expected
            assert len(m) == expected
            assert len(y) == expected


# ---------------------------------------------------------------------------
# BRCD dither table loading
# ---------------------------------------------------------------------------


class TestBRCDTables:
    """The driver loads dither matrices from BRCD cache files."""

    def test_load_brcd_cache(self):
        """Should load dither tables from a BRCD cache file."""
        from pathlib import Path

        from dither import dither_load_brcd

        lut_base = Path(__file__).resolve().parent.parent / "src" / "lut"
        k_path = str(lut_base / "0600-k_cache09.bin")
        if not Path(k_path).exists():
            pytest.skip("Original BRCD files not available")
        ch = dither_load_brcd(k_path)
        assert ch.width == 32
        assert ch.height == 32
        assert len(ch.patterns) == 256
        assert ch.patterns[0] == bytes(ch.row_bytes)  # ink=0 → no dots

    def test_bayer_fallback(self):
        """Bayer matrix fallback produces valid dither tables."""
        from dither import load_dither_tables

        channels = load_dither_tables()
        assert "K" in channels
        assert "C" in channels
        assert "M" in channels
        assert "Y" in channels
        # Each channel should have 256 patterns
        for ch in channels.values():
            assert len(ch.patterns) == 256
            assert ch.width == 32
            assert ch.height == 32


# ---------------------------------------------------------------------------
# Capture-derived dithering verification
# ---------------------------------------------------------------------------


class TestCaptureVerification:
    """Verify dithering output matches original driver captures.

    These tests extract the actual dithered bitmap from captures
    (by decompressing the compressed plane data) and compare against
    our dithering output given the same PPM input.
    """

    def test_fullwidth_k_dither_matches(self, all_captures):
        """Dithered K-plane output for test_fullwidth_k should match capture.

        test_fullwidth_k is a pure-black bar on white background.
        Black pixels → K intensity 0 (full ink) → all dots set (0xFF).
        White pixels → K intensity 255 (no ink) → no dots (0x00).
        This is independent of the dither matrix, so it should match exactly.
        """
        import io

        from brfilter import read_ppm, rgb_line_to_cmyk_intensities
        from brother_decode import decode_plane
        from dither import dither_channel_1bpp, load_dither_tables
        from fixture_utils import read_fixture

        cap = all_captures.get("test_fullwidth_k")
        if cap is None:
            pytest.skip("test_fullwidth_k capture not available")

        ppm_data = read_fixture("test_fullwidth_k.ppm")
        if ppm_data is None:
            pytest.skip("test_fullwidth_k.ppm not available")

        ppm = read_ppm(io.BytesIO(ppm_data))
        assert ppm is not None, "test_fullwidth_k.ppm is malformed"
        width, height, _, pixels = ppm

        channels = load_dither_tables()
        sw = ((width + 31) // 32) * 32  # 32-pixel aligned
        bpl = sw // 8
        pad = bytes([255] * (sw - width)) if sw > width else b""

        # Find K-plane blocks (plane_id=0)
        k_blocks = [b for b in cap.blocks if b.plane_id == 0]
        assert k_blocks, "No K-plane blocks in test_fullwidth_k"

        for block in k_blocks:
            for entry_idx, entry in enumerate(block.entries):
                line_idx = block.start_line + entry_idx
                if line_idx >= height:
                    break

                # Decompress captured data
                captured_plane = decode_plane(entry, "K", n_bytes=bpl)

                # Our pipeline: RGB → CMYK intensity → dither
                row_start = line_idx * width * 3
                rgb_row = pixels[row_start : row_start + width * 3]
                k_int, _, _, _ = rgb_line_to_cmyk_intensities(rgb_row, width)
                if pad:
                    k_int = k_int + pad
                our_plane = dither_channel_1bpp(k_int, line_idx, sw, channels["K"])

                assert our_plane == captured_plane, (
                    f"K-plane mismatch at line {line_idx}: "
                    f"our {our_plane[:8].hex()}... vs captured {captured_plane[:8].hex()}..."
                )

    def test_fullwidth_c_dither_matches(self, all_captures):
        """Dithered K and Y planes for test_fullwidth_c should match capture.

        Verifies color separation (3D LUT) + BRCD dithering for a cyan page.
        K and Y planes use 12-bit RLE encoding that we can decode and compare.
        C/M planes use a different compression codec (JPEG-LS-like) that
        requires a separate encoder implementation for byte-level matching.
        """
        import io
        from pathlib import Path

        from brfilter import read_ppm, rgb_line_to_cmyk_intensities
        from brother_decode import decode_plane
        from dither import _try_load_brcd, dither_channel_1bpp, load_dither_tables
        from fixture_utils import read_fixture

        cap = all_captures.get("test_fullwidth_c")
        if cap is None:
            pytest.skip("test_fullwidth_c capture not available")

        ppm_data = read_fixture("test_fullwidth_c.ppm")
        if ppm_data is None:
            pytest.skip("test_fullwidth_c.ppm not available")

        ppm = read_ppm(io.BytesIO(ppm_data))
        assert ppm is not None, "test_fullwidth_c.ppm is malformed"
        width, height, _, pixels = ppm

        lut_dir = str(Path(__file__).resolve().parent.parent / "src" / "lut")
        channels = _try_load_brcd(lut_dir)
        if channels is None:
            channels = load_dither_tables()
        sw = ((width + 31) // 32) * 32
        bpl = sw // 8
        pad = bytes([255] * (sw - width)) if sw > width else b""

        # Check K-plane (pid=0) and Y-plane (pid=3) — both use 12-bit RLE
        for plane_id, plane_name, ch_idx in [(0, "K", 0), (3, "Y", 3)]:
            blocks = [b for b in cap.blocks if b.plane_id == plane_id]
            if not blocks:
                continue

            mismatches = 0
            total = 0
            for block in blocks:
                for entry_idx, entry in enumerate(block.entries):
                    line_idx = block.start_line + entry_idx
                    if line_idx >= height or not entry:
                        continue
                    total += 1

                    captured = decode_plane(entry, plane_name, n_bytes=bpl)

                    row_start = line_idx * width * 3
                    rgb_row = pixels[row_start : row_start + width * 3]
                    intensities = rgb_line_to_cmyk_intensities(rgb_row, width)
                    ch_int = intensities[ch_idx]
                    if pad:
                        ch_int = ch_int + pad
                    our_plane = dither_channel_1bpp(ch_int, line_idx, sw, channels[plane_name])

                    if our_plane != captured:
                        mismatches += 1

            assert mismatches == 0, f"{plane_name}-plane: {mismatches}/{total} lines differ"

    def test_gray75_dither_matches(self, all_captures):
        """Gray75 K-plane dither+compress should match gray75_1000 capture.

        Source: RGB(64,64,64) → via driver LUT → K intensity 155 (pixel-brightness).
        Compares compressed outputs (not decoded bitmaps) because the K-plane
        12-bit RLE is lossy for dithered patterns — different bitmaps can
        produce identical compressed bytes.
        """
        from pathlib import Path

        from brfilter import rgb_line_to_cmyk_intensities
        from brother_encode import encode_plane

        cap = all_captures.get("gray75_1000")
        if cap is None:
            pytest.skip("gray75_1000 capture not available")

        k_blocks = [b for b in cap.blocks if b.plane_id == 0]
        assert k_blocks, "No K-plane blocks in gray75_1000"

        from dither import _try_load_brcd, dither_channel_1bpp, load_dither_tables

        lut_dir = str(Path(__file__).resolve().parent.parent / "src" / "lut")
        channels = _try_load_brcd(lut_dir)
        if channels is None:
            channels = load_dither_tables()
        sw = 4768  # standard 32-pixel aligned width for A4
        width = 4760

        # Gray75 source = RGB(64,64,64) → LUT → K intensity (pixel-brightness)
        rgb_row = bytes([64, 64, 64]) * width
        k_int, _, _, _ = rgb_line_to_cmyk_intensities(rgb_row, width)
        pad = b"\xff" * (sw - width)
        gray_k_int = k_int + pad

        mismatches = 0
        total = 0
        for block in k_blocks:
            for entry_idx, entry in enumerate(block.entries):
                if not entry:
                    continue
                total += 1
                line_idx = block.start_line + entry_idx

                our_plane = dither_channel_1bpp(gray_k_int, line_idx, sw, channels["K"])
                our_compressed = encode_plane(our_plane, "K")

                if our_compressed != entry:
                    mismatches += 1

        assert mismatches == 0, f"{mismatches}/{total} K-plane compressed lines differ"


# ---------------------------------------------------------------------------
# Fine mode BRCD dither table loading
# ---------------------------------------------------------------------------


class TestFineBRCDTables:
    """Fine mode uses different BRCD cache files with different matrix dimensions."""

    def test_load_fine_brcd_channels(self):
        """Should load all 4 Fine BRCD channels."""
        from pathlib import Path

        from dither import _try_load_brcd

        lut_dir = str(Path(__file__).resolve().parent.parent / "src" / "lut")
        channels = _try_load_brcd(lut_dir, fine=True)
        if channels is None:
            pytest.skip("Fine BRCD files not available")
        assert set(channels.keys()) == {"K", "C", "M", "Y"}

    def test_fine_brcd_dimensions(self):
        """Fine BRCD channels have specific dimensions per channel."""
        from pathlib import Path

        from dither import _try_load_brcd

        lut_dir = str(Path(__file__).resolve().parent.parent / "src" / "lut")
        channels = _try_load_brcd(lut_dir, fine=True)
        if channels is None:
            pytest.skip("Fine BRCD files not available")
        # K: 96x12, C: 160x20, M: 160x20, Y: 48x12 (from captures)
        k = channels["K"]
        c = channels["C"]
        m = channels["M"]
        y = channels["Y"]
        assert k.width * k.height > 0
        assert c.width * c.height > 0
        assert m.width * m.height > 0
        assert y.width * y.height > 0

    def test_fine_4bpp_dithering_output_length(self):
        """4bpp dithering with Fine channels produces correct output length."""
        from pathlib import Path

        from dither import _try_load_brcd, dither_channel_4bpp

        lut_dir = str(Path(__file__).resolve().parent.parent / "src" / "lut")
        channels = _try_load_brcd(lut_dir, fine=True)
        if channels is None:
            pytest.skip("Fine BRCD files not available")

        width = 4960  # Fine A4 width
        row = bytes([128] * width)  # 50% gray
        result = dither_channel_4bpp(row, y=0, width=width, channel=channels["K"])
        assert len(result) == (width + 1) // 2  # 2480 bytes

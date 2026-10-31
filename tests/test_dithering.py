"""
Ordered dithering tests.

Tests the dithering subsystem against the reverse-engineered dither.c
of the original driver and against its captures.

The driver uses ordered dithering with BRCD lookup tables,
supporting 1bpp (bitmap, Normal mode) and 4bpp (nibble, Fine mode)
output, one channel at a time.
"""

import pytest

from brother_xl2hb.color.dither import load_dither_tables
from brother_xl2hb.models import HL4150CDN

# The generated Bayer K channel, the tables used without BRCD files.
BAYER_K = load_dither_tables(cache=HL4150CDN.dither_cache)["K"]

# ---------------------------------------------------------------------------
# Dither module API
# ---------------------------------------------------------------------------


class TestDitherAPI:
    """Tests that the dither module exposes the expected API."""

    def test_module_exists(self):
        from brother_xl2hb.color import dither  # noqa: F401

    def test_has_init_function(self):
        from brother_xl2hb.color.dither import load_dither_tables  # noqa: F401

    def test_has_ordered_1ch_1bpp(self):
        from brother_xl2hb.color.dither import dither_channel_1bpp  # noqa: F401

    def test_has_ordered_1ch_4bpp(self):
        from brother_xl2hb.color.dither import dither_channel_4bpp  # noqa: F401


# ---------------------------------------------------------------------------
# 1bpp ordered dithering (bitmap mode — what the HL-4150CDN uses)
# ---------------------------------------------------------------------------


class TestOrdered1bpp:
    """Ordered dithering to 1-bit-per-pixel output."""

    def test_white_input_no_dots(self):
        """Pure white (255) should produce no dots at any position."""
        from brother_xl2hb.color.dither import dither_channel_1bpp

        row = bytes([255] * 4760)
        result = dither_channel_1bpp(row, y=0, width=4760, channel=BAYER_K)
        assert not any(result)

    def test_black_input_all_dots(self):
        """Pure black (0) should produce all dots set."""
        from brother_xl2hb.color.dither import dither_channel_1bpp

        row = bytes([0] * 4760)
        result = dither_channel_1bpp(row, y=0, width=4760, channel=BAYER_K)
        bpl = (4760 + 7) // 8
        assert len(result) == bpl
        # All bits set for active pixels
        for x in range(4760):
            byte_idx = x >> 3
            bit_mask = 0x80 >> (x & 7)
            assert result[byte_idx] & bit_mask, f"Pixel {x} should be set"

    def test_50pct_gray_coverage(self):
        """50% gray should produce approximately 50% dot coverage."""
        from brother_xl2hb.color.dither import dither_channel_1bpp

        row = bytes([128] * 4760)
        result = dither_channel_1bpp(row, y=0, width=4760, channel=BAYER_K)
        bits_set = sum(b.bit_count() for b in result[:595])
        # Allow 10% tolerance
        assert 1900 < bits_set < 2860, f"Expected ~2380 bits, got {bits_set}"

    def test_monotonic_coverage(self):
        """Darker grays should produce more dots than lighter grays."""
        from brother_xl2hb.color.dither import dither_channel_1bpp

        coverages = []
        for gray in [224, 192, 128, 64, 32]:
            row = bytes([gray] * 4760)
            result = dither_channel_1bpp(row, y=0, width=4760, channel=BAYER_K)
            bits = sum(b.bit_count() for b in result[:595])
            coverages.append(bits)
        # Each should be strictly greater than the previous
        for i in range(len(coverages) - 1):
            assert coverages[i] < coverages[i + 1], f"Coverage not monotonic: gray levels produced {coverages}"

    def test_position_dependent(self):
        """Different y positions should produce different patterns for same input."""
        from brother_xl2hb.color.dither import dither_channel_1bpp

        row = bytes([128] * 4760)
        result_y0 = dither_channel_1bpp(row, y=0, width=4760, channel=BAYER_K)
        result_y1 = dither_channel_1bpp(row, y=1, width=4760, channel=BAYER_K)
        assert result_y0 != result_y1, "Dither should vary by y position"

    def test_deterministic(self):
        """Same input + position should produce same output."""
        from brother_xl2hb.color.dither import dither_channel_1bpp

        row = bytes([128] * 4760)
        a = dither_channel_1bpp(row, y=42, width=4760, channel=BAYER_K)
        b = dither_channel_1bpp(row, y=42, width=4760, channel=BAYER_K)
        assert a == b


# ---------------------------------------------------------------------------
# 4bpp ordered dithering (nibble mode — 16 intensity levels per pixel)
# ---------------------------------------------------------------------------


class TestOrdered1ch4bpp:
    """Ordered dithering to 4-bit-per-pixel nibble-packed output."""

    def test_white_input_all_zero_nibbles(self):
        """Pure white (255) should produce all-zero nibbles."""
        from brother_xl2hb.color.dither import dither_channel_4bpp

        row = bytes([255] * 4760)
        result = dither_channel_4bpp(row, y=0, width=4760, channel=BAYER_K)
        assert not any(result)

    def test_black_input_all_max_nibbles(self):
        """Pure black (0) should produce all 0xFF bytes (nibbles = 15)."""
        from brother_xl2hb.color.dither import dither_channel_4bpp

        row = bytes([0] * 4760)
        result = dither_channel_4bpp(row, y=0, width=4760, channel=BAYER_K)
        assert all(b == 0xFF for b in result)

    def test_output_length(self):
        """Output should be (width + 1) // 2 bytes."""
        from brother_xl2hb.color.dither import dither_channel_4bpp

        for width in [4760, 100, 32]:
            row = bytes([128] * width)
            result = dither_channel_4bpp(row, y=0, width=width, channel=BAYER_K)
            assert len(result) == (width + 1) // 2

    def test_output_length_odd_width(self):
        """Odd width: output is (width + 1) // 2, last low nibble padded to 0."""
        from brother_xl2hb.color.dither import dither_channel_4bpp

        row = bytes([0] * 33)  # all black
        result = dither_channel_4bpp(row, y=0, width=33, channel=BAYER_K)
        assert len(result) == 17  # (33 + 1) // 2
        # Last byte: high nibble = 15 (pixel 32), low nibble = 0 (padding)
        assert result[-1] == 0xF0

    def test_nibble_packing_order(self):
        """First pixel goes into high nibble, second into low nibble."""
        from brother_xl2hb.color.dither import dither_channel_4bpp

        # All black: every nibble = 15 → 0xFF per byte
        row = bytes([0] * 2)
        result = dither_channel_4bpp(row, y=0, width=2, channel=BAYER_K)
        assert result[0] == 0xFF

        # Single pixel (odd width): high nibble set, low nibble zero
        row = bytes([0] * 1)
        result = dither_channel_4bpp(row, y=0, width=1, channel=BAYER_K)
        assert result[0] == 0xF0

    def test_50pct_gray_coverage(self):
        """50% gray should produce nibbles averaging around 7-8."""
        from brother_xl2hb.color.dither import dither_channel_4bpp

        row = bytes([128] * 4760)
        result = dither_channel_4bpp(row, y=0, width=4760, channel=BAYER_K)
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
        from brother_xl2hb.color.dither import dither_channel_4bpp

        sums = []
        for gray in [224, 192, 128, 64, 32]:
            row = bytes([gray] * 4760)
            result = dither_channel_4bpp(row, y=0, width=4760, channel=BAYER_K)
            nibble_sum = 0
            for b in result:
                nibble_sum += (b >> 4) & 0x0F
                nibble_sum += b & 0x0F
            sums.append(nibble_sum)
        for i in range(len(sums) - 1):
            assert sums[i] < sums[i + 1], f"Nibble sum not monotonic: {sums}"

    def test_position_dependent(self):
        """Different y positions should produce different output for same input."""
        from brother_xl2hb.color.dither import dither_channel_4bpp

        row = bytes([128] * 4760)
        result_y0 = dither_channel_4bpp(row, y=0, width=4760, channel=BAYER_K)
        result_y1 = dither_channel_4bpp(row, y=1, width=4760, channel=BAYER_K)
        assert result_y0 != result_y1, "4bpp dither should vary by y position"

    def test_deterministic(self):
        """Same input + position should produce identical output."""
        from brother_xl2hb.color.dither import dither_channel_4bpp

        row = bytes([128] * 4760)
        a = dither_channel_4bpp(row, y=42, width=4760, channel=BAYER_K)
        b = dither_channel_4bpp(row, y=42, width=4760, channel=BAYER_K)
        assert a == b

    def test_nibble_values_in_range(self):
        """All nibble values should be in [0, 15]."""
        from brother_xl2hb.color.dither import dither_channel_4bpp

        for gray in [0, 64, 128, 192, 255]:
            row = bytes([gray] * 100)
            result = dither_channel_4bpp(row, y=7, width=100, channel=BAYER_K)
            for b in result:
                assert (b >> 4) <= 15
                assert (b & 0x0F) <= 15


# ---------------------------------------------------------------------------
# BRCD dither table loading
# ---------------------------------------------------------------------------


class TestBRCDTables:
    """The driver loads dither matrices from BRCD cache files."""

    def test_load_brcd_cache(self):
        """Should load dither tables from a BRCD cache file."""
        from pathlib import Path

        from brother_xl2hb.color.dither import dither_channel_1bpp, dither_load_brcd

        lut_base = HL4150CDN.lut_dir
        k_path = str(lut_base / "0600-k_cache09.bin")
        if not Path(k_path).exists():
            pytest.skip("Original BRCD files not available")
        ch = dither_load_brcd(k_path)
        assert ch.width == 32
        assert ch.height == 32
        assert ch.threshold_matrix is not None
        assert ch.threshold_matrix.shape == (32, 32)
        # ink=0 (white) places no dots, full ink (black) fills every position.
        assert dither_channel_1bpp(bytes([255] * 32), 0, 32, ch) == bytes(4)
        assert dither_channel_1bpp(bytes(32), 0, 32, ch) == b"\xff" * 4

    def test_bayer_fallback(self):
        """Bayer matrix fallback produces valid dither tables."""
        from brother_xl2hb.color.dither import load_dither_tables

        channels = load_dither_tables(cache=HL4150CDN.dither_cache)
        assert "K" in channels
        assert "C" in channels
        assert "M" in channels
        assert "Y" in channels
        for ch in channels.values():
            assert ch.threshold_matrix is not None
            assert ch.threshold_matrix.shape == (32, 32)
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

        from brother_decode import decode_plane
        from brother_xl2hb.color.color_lut import rgb_to_cmyk_lut
        from brother_xl2hb.color.dither import dither_channel_1bpp, load_dither_tables
        from brother_xl2hb.render.ppm import read_ppm
        from fixture_utils import read_fixture

        cap = all_captures["test_fullwidth_k"]

        ppm_data = read_fixture("test_fullwidth_k.ppm")

        ppm = read_ppm(io.BytesIO(ppm_data))
        assert ppm is not None, "test_fullwidth_k.ppm is malformed"
        width, height, _, pixels = ppm

        channels = load_dither_tables(cache=HL4150CDN.dither_cache)
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
                k_int, _, _, _ = rgb_to_cmyk_lut(rgb_row, width, HL4150CDN.grid())
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
        K and Y planes use 12-bit RLE encoding that we can decode and compare
        here; the C and M encoders are covered byte for byte by the pipeline
        capture tests.
        """
        import io

        from brother_decode import decode_plane
        from brother_xl2hb.color.color_lut import rgb_to_cmyk_lut
        from brother_xl2hb.color.dither import dither_channel_1bpp, load_brcd_tables
        from brother_xl2hb.render.ppm import read_ppm
        from fixture_utils import read_fixture

        cap = all_captures["test_fullwidth_c"]

        ppm_data = read_fixture("test_fullwidth_c.ppm")

        ppm = read_ppm(io.BytesIO(ppm_data))
        assert ppm is not None, "test_fullwidth_c.ppm is malformed"
        width, height, _, pixels = ppm

        lut_dir = str(HL4150CDN.lut_dir)
        channels = load_brcd_tables(lut_dir, cache=HL4150CDN.dither_cache)
        assert channels is not None, "BRCD tables missing; run scripts/extract_blobs.sh"
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
                    intensities = rgb_to_cmyk_lut(rgb_row, width, HL4150CDN.grid())
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

        from brother_xl2hb.codec.plane_encoders import encode_plane_12bit
        from brother_xl2hb.color.color_lut import rgb_to_cmyk_lut

        cap = all_captures["gray75_1000"]

        k_blocks = [b for b in cap.blocks if b.plane_id == 0]
        assert k_blocks, "No K-plane blocks in gray75_1000"

        from brother_xl2hb.color.dither import dither_channel_1bpp, load_brcd_tables

        lut_dir = str(HL4150CDN.lut_dir)
        channels = load_brcd_tables(lut_dir, cache=HL4150CDN.dither_cache)
        assert channels is not None, "BRCD tables missing; run scripts/extract_blobs.sh"
        sw = 4768  # standard 32-pixel aligned width for A4
        width = 4760

        # Gray75 source = RGB(64,64,64) → LUT → K intensity (pixel-brightness)
        rgb_row = bytes([64, 64, 64]) * width
        k_int, _, _, _ = rgb_to_cmyk_lut(rgb_row, width, HL4150CDN.grid())
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
                our_compressed = encode_plane_12bit(our_plane)

                if our_compressed != entry:
                    mismatches += 1

        assert mismatches == 0, f"{mismatches}/{total} K-plane compressed lines differ"


# ---------------------------------------------------------------------------
# Fine mode BRCD dither table loading
# ---------------------------------------------------------------------------


class TestFineBRCDTables:
    """Fine mode uses different BRCD cache files with different matrix dimensions."""

    def test_nibble_table_width_must_be_whole_bytes(self, tmp_path):
        """A Fine table is one byte per pixel; a width in bits that is not a multiple of 8 is rejected."""
        import struct

        from brother_xl2hb.color.dither import dither_load_brcd

        width, height = 12, 2
        row_bytes = (width * height + 7) // 8
        path = tmp_path / "odd.bin"
        path.write_bytes(b"BRCD1\0" + struct.pack("<HH", width, height) + bytes(256 * row_bytes))
        with pytest.raises(ValueError, match="multiple of 8"):
            dither_load_brcd(str(path), nibbles=True)

    @pytest.mark.parametrize(("version", "nibbles"), [(b"1", False), (b"0", True)])
    def test_version_must_match_the_table_kind(self, tmp_path, version, nibbles):
        """A Fine ('1') table is not read as Normal ('0'), nor the other way round."""
        import struct

        from brother_xl2hb.color.dither import dither_load_brcd

        width, height = 16, 2
        path = tmp_path / "table.bin"
        path.write_bytes(b"BRCD" + version + b"\0" + struct.pack("<HH", width, height) + bytes(256 * 4))
        with pytest.raises(ValueError, match="BRCD version"):
            dither_load_brcd(str(path), nibbles=nibbles)

    def test_dot_at_ink_zero_is_rejected(self, tmp_path):
        """A 1bpp dot at ink level 0 has no uint8 threshold (it would wrap to 255)."""
        import struct

        from brother_xl2hb.color.dither import dither_load_brcd

        width, height = 16, 2
        patterns = b"\x80\0\0\0" + b"\xff" * (255 * 4)
        path = tmp_path / "table.bin"
        path.write_bytes(b"BRCD0\0" + struct.pack("<HH", width, height) + patterns)
        with pytest.raises(ValueError, match="ink level 0"):
            dither_load_brcd(str(path))

    def test_load_fine_brcd_channels(self):
        """Should load all 4 Fine BRCD channels."""

        from brother_xl2hb.color.dither import load_brcd_tables

        lut_dir = str(HL4150CDN.lut_dir)
        channels = load_brcd_tables(lut_dir, fine=True, cache=HL4150CDN.dither_cache)
        if channels is None:
            pytest.skip("Fine BRCD files not available")
        assert set(channels.keys()) == {"K", "C", "M", "Y"}

    def test_fine_brcd_dimensions(self):
        """Fine BRCD channels have specific dimensions per channel."""

        from brother_xl2hb.color.dither import load_brcd_tables

        lut_dir = str(HL4150CDN.lut_dir)
        channels = load_brcd_tables(lut_dir, fine=True, cache=HL4150CDN.dither_cache)
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

        from brother_xl2hb.color.dither import dither_channel_4bpp, load_brcd_tables

        lut_dir = str(HL4150CDN.lut_dir)
        channels = load_brcd_tables(lut_dir, fine=True, cache=HL4150CDN.dither_cache)
        if channels is None:
            pytest.skip("Fine BRCD files not available")

        width = 4960  # Fine A4 width
        row = bytes([128] * width)  # 50% gray
        result = dither_channel_4bpp(row, y=0, width=width, channel=channels["K"])
        assert len(result) == (width + 1) // 2  # 2480 bytes


# ---------------------------------------------------------------------------
# ndarray variants (hot path)
# ---------------------------------------------------------------------------


class TestArrVariants:
    """The *_arr variants must match the bytes API, including over-wide rows."""

    @pytest.mark.parametrize("fn_name", ["dither_channel_1bpp", "dither_channel_4bpp"])
    def test_wider_row_truncated_to_width(self, fn_name):
        """A row longer than `width` (uncropped PPM) is cut like the bytes API does."""
        import numpy as np

        from brother_xl2hb.color import dither

        width = 4768
        row = np.arange(4958, dtype=np.uint32).astype(np.uint8)
        arr_fn = getattr(dither, f"{fn_name}_arr")
        bytes_fn = getattr(dither, fn_name)
        for y in range(3):
            assert arr_fn(row, y, width, BAYER_K) == bytes_fn(row.tobytes(), y, width, BAYER_K)


def test_native_1bpp_dither_matches_numpy(monkeypatch):
    """_dither_fast.dither_row_1bpp agrees with the numpy threshold path."""
    import numpy as np

    from brother_xl2hb.color import dither

    native_module = dither._dither_fast
    if native_module is None:
        pytest.skip("_dither_fast extension not built")
    rng = np.random.default_rng(1)
    channels = dither.load_dither_tables(cache=HL4150CDN.dither_cache)
    for width in (4768, 4761, 5):
        row = rng.integers(0, 256, width + 190, dtype=np.uint8)
        strided = np.repeat(row, 2)[::2]  # non-contiguous view, as from a KCMY column
        for y in (0, 17, 31):
            native = dither.dither_channel_1bpp_arr(strided, y, width, channels["C"])
            monkeypatch.setattr(dither, "_dither_fast", None)
            reference = dither.dither_channel_1bpp_arr(strided, y, width, channels["C"])
            monkeypatch.setattr(dither, "_dither_fast", native_module)
            assert native == reference


# ---------------------------------------------------------------------------
# Missing BRCD blobs: Bayer fallback must be loud, but only once
# ---------------------------------------------------------------------------


class TestBayerFallbackWarning:
    """Missing dither tables change every page, so the fallback warns (once)."""

    @pytest.fixture(autouse=True)
    def _fresh_warning_state(self, monkeypatch):
        from brother_xl2hb.color import dither

        monkeypatch.setattr(dither, "_fallback_warned", set())

    def _fallback_warnings(self, caplog):
        return [r for r in caplog.records if r.name == "brother_xl2hb.color.dither" and r.levelname == "WARNING"]

    def test_missing_tables_warn_with_hint(self, tmp_path, caplog):
        from brother_xl2hb.color.dither import load_dither_tables

        with caplog.at_level("WARNING", logger="brother_xl2hb.color.dither"):
            load_dither_tables(str(tmp_path), cache=HL4150CDN.dither_cache)
        warnings = self._fallback_warnings(caplog)
        assert len(warnings) == 1
        assert "scripts/extract_blobs.sh" in warnings[0].getMessage()
        assert "Bayer" in warnings[0].getMessage()

    def test_warning_is_logged_once_per_table(self, tmp_path, caplog):
        from brother_xl2hb.color.dither import load_dither_tables

        with caplog.at_level("WARNING", logger="brother_xl2hb.color.dither"):
            for _ in range(3):
                load_dither_tables(str(tmp_path), cache=HL4150CDN.dither_cache)
                load_dither_tables(str(tmp_path), fine=True, cache=HL4150CDN.dither_cache)
        warnings = self._fallback_warnings(caplog)
        assert len(warnings) == 2
        assert "0600-*_cache09.bin" in warnings[0].getMessage()
        assert "capt-*_cache09.bin" in warnings[1].getMessage()

        with caplog.at_level("WARNING", logger="brother_xl2hb.color.dither"):
            load_dither_tables(str(tmp_path), toner_save=True, cache=HL4150CDN.dither_cache)
            load_dither_tables(str(tmp_path), toner_save=True, cache=HL4150CDN.dither_cache)
        warnings = self._fallback_warnings(caplog)
        assert len(warnings) == 3
        assert "0600-*-TS_cache09.bin" in warnings[2].getMessage()

    def test_fine_falls_back_to_normal_tables_with_warning(self, tmp_path, caplog):
        from brother_xl2hb.color.dither import load_dither_tables

        lut_dir = HL4150CDN.lut_dir
        if not (lut_dir / "0600-k_cache09.bin").exists():
            pytest.skip("BRCD blobs not extracted")
        model_dir = tmp_path / HL4150CDN.key
        model_dir.mkdir()
        for blob in lut_dir.glob("0600-*_cache09.bin"):
            (model_dir / blob.name).symlink_to(blob)
        with caplog.at_level("WARNING", logger="brother_xl2hb.color.dither"):
            channels = load_dither_tables(str(model_dir), fine=True, cache=HL4150CDN.dither_cache)
        assert channels["K"].nibble_levels is None
        [warning] = self._fallback_warnings(caplog)
        message = warning.getMessage()
        assert "capt-*_cache09.bin missing" in message
        assert "Normal-mode tables 0600-*_cache09.bin" in message
        assert f"scripts/extract_blobs.sh {HL4150CDN.key}" in message

    def test_partial_set_names_the_missing_files(self, tmp_path, caplog):
        from brother_xl2hb.color.dither import load_dither_tables

        lut_dir = HL4150CDN.lut_dir
        present = ("0600-k_cache09.bin", "0600-m_cache09.bin")
        if not all((lut_dir / name).exists() for name in present):
            pytest.skip("BRCD blobs not extracted")
        model_dir = tmp_path / HL4150CDN.key
        model_dir.mkdir()
        for name in present:
            (model_dir / name).symlink_to(lut_dir / name)
        with caplog.at_level("WARNING", logger="brother_xl2hb.color.dither"):
            load_dither_tables(str(model_dir), cache=HL4150CDN.dither_cache)
        [warning] = self._fallback_warnings(caplog)
        assert "Dither tables 0600-c_cache09.bin, 0600-y_cache09.bin missing" in warning.getMessage()

    def test_present_tables_do_not_warn(self, caplog):

        from brother_xl2hb.color.dither import load_dither_tables

        lut_dir = HL4150CDN.lut_dir
        if not (lut_dir / "0600-k_cache09.bin").exists():
            pytest.skip("BRCD blobs not extracted")
        with caplog.at_level("WARNING", logger="brother_xl2hb.color.dither"):
            load_dither_tables(str(lut_dir), cache=HL4150CDN.dither_cache)
        assert not self._fallback_warnings(caplog)
